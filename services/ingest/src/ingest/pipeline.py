"""One document end to end: parse -> SQL tables -> describe images -> chunk -> embed -> index.

The document's owner (its registry row) decides where its files go (paths.py), the user_id of its Qdrant
points, and which earlier versions it may supersede.
"""

import asyncio
import dataclasses
import hashlib
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from rag_common.contracts import DocStatus, DocumentInfo, TableSchema
from rag_common.index import ChunkPayload, point_id
from rag_common.logging import log_fields

from ingest import chunkstore
from ingest.chunking import (
    TokenCounter,
    context_header,
    group_by_section,
    image_chunks,
    prose_chunks,
    row_chunks,
    sheet_summary,
    summary_chunk,
    table_chunks,
)
from ingest.config import Settings
from ingest.describe import describe_image, summarize
from ingest.elements import Chunk, Element, ParsedDocument
from ingest.indexer import Indexer
from ingest.isolation import IsolatedRunner
from ingest.llmcache import JobCache, LlmCache
from ingest.parsers import parse_file
from ingest.paths import (
    asset_dir,
    chunk_path,
    remove_artifacts,
    remove_empty_owner_dirs,
    remove_owner,
    remove_worker_artifacts,
    source_path,
    table_path,
)
from ingest.registry import Registry
from ingest.tables import TableSource, column_warnings, write_tables

log = logging.getLogger("ingest.pipeline")
DropPoints = Callable[[str, str], Awaitable[None]]  # (owner, doc_id): delete the document's points
# The version of what builds a document's index. Bump it when a parsing or chunking fix leaves documents
# indexed before it missing content: the worker re-queues every ready document of a lower version at start.
# 1: before versions were kept. 2: text after a figure or table in the same section is no longer dropped.
INDEX_VERSION = 2


class RunLost(Exception):
    """The run no longer holds its document (deleted, its account purged, or taken back): stop at once."""


async def delete_document(settings: Settings, registry: Registry, drop_points: DropPoints, doc: DocumentInfo,
                          claim: str | None) -> bool:
    """Remove a deleting document: points, then files, then the row. Returns True when the row is gone.
    Without a claim (a run holds the row) only the points go; the run removes the files and the row when it
    sees the deletion, within one lease renewal. With a claim, the claim is renewed (a fresh lease) before
    each step that removes data, and the work stops as soon as the claim is lost: once its lease ran out
    someone else may have taken the row over, and after them a new upload of the same file may own the
    folder. The files are followed by the document's cached model replies that no other document of the owner
    uses (llmcache.py). A failure raises and leaves the row deleting (hidden); the worker's sweep finishes it
    later."""
    owner = _owner(doc)
    if claim is not None and not registry.renew_claim(doc.doc_id, claim):
        return False
    await drop_points(owner, doc.doc_id)
    if claim is None or not registry.renew_claim(doc.doc_id, claim):
        return False
    remove_artifacts(settings, owner, doc.doc_id)
    if not registry.renew_claim(doc.doc_id, claim):
        return False
    await asyncio.to_thread(LlmCache.from_settings(settings).release_document, owner, doc.doc_id)
    return registry.finish_deleting(doc.doc_id, claim)


def _owner(doc: DocumentInfo) -> str:
    if not doc.owner_id:
        raise ValueError(f"document {doc.doc_id} has no owner")
    return doc.owner_id


def _now() -> float:
    return time.monotonic()


@dataclass
class Pipeline:
    settings: Settings
    registry: Registry
    indexer: Indexer
    vision: httpx.AsyncClient
    counter: TokenCounter
    runner: IsolatedRunner
    cache: LlmCache | None = None  # the LLM cache for jobs; None, or no model identity: every reply from the model
    _threads: set[threading.Event] = field(default_factory=set, init=False, repr=False)

    async def in_thread(self, fn: Callable, /, *args, **kwargs):
        """asyncio.to_thread that is remembered: cancelling the run does not stop the thread, which may still
        write a file, so whoever cleans up after a cancelled run waits for it first (settle)."""
        finished = threading.Event()

        def run():
            try:
                return fn(*args, **kwargs)
            finally:
                finished.set()

        self._threads.add(finished)
        try:
            return await asyncio.to_thread(run)
        finally:
            if finished.is_set():
                self._threads.discard(finished)

    async def settle(self) -> None:
        """Wait until every thread a cancelled run started has returned."""
        for finished in list(self._threads):
            await asyncio.to_thread(finished.wait)
            self._threads.discard(finished)

    async def parse(self, doc: DocumentInfo, ext: str) -> ParsedDocument:
        """Every format is parsed in the isolated child, with the timeout and crash detection."""
        path = source_path(self.settings, _owner(doc), doc.doc_id, ext)
        artifacts = self.settings.docling_artifacts
        artifacts = artifacts if artifacts and Path(artifacts).exists() else None
        return await self.in_thread(
            self.runner.run, parse_file, str(path), ext, doc.name, artifacts, self.settings.parse_threads,
            timeout=self.settings.parse_timeout_s,
        )

    async def describe(self, doc: DocumentInfo, parsed: ParsedDocument, cache: JobCache | None = None) -> None:
        """Describe each distinct image once; cap the number per document. A description the LLM cache holds
        costs no model call."""
        images = [e for e in parsed.elements if e.kind == "image"]
        seen: set[str] = set()
        keep: list[Element] = []
        for el in images:
            digest = hashlib.sha256(el.image).hexdigest()
            if digest not in seen and len(keep) < self.settings.max_figures_per_doc:
                seen.add(digest)
                keep.append(el)
        dropped = len(images) - len(keep)
        if dropped:
            parsed.warnings.append(f"{dropped} duplicate or excess figures were not described")
        kept = {id(e) for e in keep}
        parsed.elements = [e for e in parsed.elements if e.kind != "image" or id(e) in kept]
        truncated_at: list[str] = []
        failed_at: list[str] = []
        skipped_at: list[str] = []
        failures_in_a_row, started, stop_reason = 0, _now(), ""
        for i, el in enumerate(keep, 1):
            where = f"p. {el.page}" if el.page else f"#{i}"
            if not el.standalone and not stop_reason:  # the circuit breaker: the vision model is down or too slow
                if failures_in_a_row >= self.settings.figure_failure_limit:
                    stop_reason = f"{failures_in_a_row} figures in a row failed"
                elif _now() - started > self.settings.figure_deadline_s:
                    stop_reason = f"describing took longer than {self.settings.figure_deadline_s:.0f} s"
                if stop_reason:
                    log.error("figure descriptions stopped; the rest are searchable by caption only",
                              extra={"fields": {"doc_id": doc.doc_id, "reason": stop_reason, "left": len(keep) - i + 1}})
            if stop_reason and not el.standalone:
                skipped_at.append(where)
                continue
            self.step(doc, f"describing image {i}/{len(keep)}")
            budget = self.settings.image_description_tokens if el.standalone else self.settings.figure_description_tokens
            try:
                text, truncated = await describe_image(
                    self.vision, self.settings.vision_settings, el.image, doc.name, el.text, el.section, el.standalone,
                    budget, cache=cache,
                )
            except Exception as exc:
                if el.standalone:  # an image file is its description: without it there is nothing to index
                    raise
                log.warning("describing a figure failed after a retry; indexing the rest",
                            extra={"fields": {"doc_id": doc.doc_id, "figure": where, "error": str(exc)[:300]}})
                failed_at.append(where)
                failures_in_a_row += 1
                continue
            failures_in_a_row = 0
            if truncated:
                truncated_at.append(where)
            el.text = f"{el.text}\n\n{text}".strip() if el.text else text
        if skipped_at:
            parsed.warnings.append(
                f"figure descriptions stopped ({stop_reason}): {len(skipped_at)} of {len(keep)} figures were not "
                f"described and are searchable by caption only ({', '.join(skipped_at[:10])}"
                f"{', ...' if len(skipped_at) > 10 else ''}); reindex to try again "
                "(INGEST_FIGURE_FAILURE_LIMIT, INGEST_FIGURE_DEADLINE_S)"
            )
        if failed_at:
            parsed.warnings.append(
                f"{len(failed_at)} of {len(keep)} figures could not be described and are searchable by caption only "
                f"({', '.join(failed_at[:10])}{', ...' if len(failed_at) > 10 else ''}); reindex to try again"
            )
        if truncated_at:
            parsed.warnings.append(
                f"{len(truncated_at)} of {len(keep)} image descriptions reached the token limit "
                f"({', '.join(truncated_at[:10])}{', ...' if len(truncated_at) > 10 else ''})"
            )

    def table_sources(self, parsed: ParsedDocument) -> list[TableSource]:
        cfg = self.settings.chunking
        sources = []
        for sheet in parsed.sheets:
            small = len(sheet.frame) <= 60 and len(table_chunks(sheet.frame, "", "", self.counter, cfg)) == 1
            sources.append(TableSource(sheet.name, f'sheet "{sheet.name}"', sheet.name, sheet.frame, row_filter=not small,
                                       title=sheet.title, notes=sheet.notes, headers=sheet.headers, units=sheet.units,
                                       layout=sheet.layout or None, raw=sheet.raw or None))
        for el in parsed.elements:
            if el.kind == "table" and len(el.frame) >= self.settings.min_pdf_table_rows and el.frame.shape[1] >= 2:
                sources.append(
                    TableSource(f"p{el.page}_table{el.table_index}", f"page {el.page}, table {el.table_index}",
                                None, el.frame, row_filter=False)
                )
        return sources

    def chunk(self, doc: DocumentInfo, parsed: ParsedDocument, schemas: list[TableSchema]) -> list[Chunk]:
        cfg = self.settings.chunking
        chunks: list[Chunk] = []
        for group in group_by_section(parsed.elements):
            el = group[0]
            if el.kind == "text":
                chunks += prose_chunks(group, doc.name, self.counter, cfg)
            elif el.kind == "table":
                header = context_header(doc.name, f"Table {el.table_index}", f"Section: {el.section}" if el.section else "")
                chunks += table_chunks(
                    el.frame, header, el.text, self.counter, cfg,
                    section=el.section, page=el.page, page_end=el.page, table_index=el.table_index,
                )
            else:
                asset = hashlib.sha256(el.image).hexdigest()[:16] + ".png"
                chunks += image_chunks(el, asset, doc.name, self.counter, cfg)

        # table_sources lists the sheets first, in order, and write_tables keeps that order: pair them by position
        # (by name, a copied sheet "Sales (2)" and a second table of "Sales" could take each other's schema)
        for sheet, schema in zip(parsed.sheets, schemas[:len(parsed.sheets)], strict=True):
            chunks.append(summary_chunk(sheet, sheet_summary(sheet, schema), doc.name, self.counter))
            if not schema.row_filter:
                header = context_header(doc.name, f"Sheet: {sheet.name}")
                chunks += table_chunks(sheet.frame, header, "", self.counter, cfg, sheet=sheet.name, table=schema.table)
                continue
            limit = self.settings.max_rows_indexed
            if len(sheet.frame) > limit:
                parsed.warnings.append(
                    f'Sheet "{sheet.name}": first {limit:,} of {len(sheet.frame):,} rows indexed for search; SQL sees all rows'
                )
            capped = dataclasses.replace(sheet, frame=sheet.frame.iloc[:limit])
            chunks += row_chunks(capped, schema.table, doc.name, self.counter, cfg)
        return chunks

    async def summary_chunk(
        self, doc: DocumentInfo, chunks: list[Chunk], schemas: list[TableSchema], warnings: list[str],
        cache: JobCache | None = None,
    ) -> Chunk | None:
        """Material = headings, table schemas and the opening ~2,500 tokens; appended last so
        existing chunk indexes (and evaluation labels) do not shift."""
        sections = list(dict.fromkeys(c.section for c in chunks if c.section))[:30]
        parts = [f"Sections: {'; '.join(sections)}"] if sections else []
        parts += [f"Table {t.table} ({t.source}, {t.rows} rows): {', '.join(c.name for c in t.columns)}" for t in schemas[:15]]
        used = 0
        for c in chunks:
            if c.kind in ("text", "table", "image", "sheet_summary") and used < 2500:
                parts.append(c.text)
                used += c.n_tokens
        try:
            text, cut = await summarize(self.vision, self.settings.vision_settings, doc.name, doc.doc_type.value,
                                        "\n\n".join(parts)[:16000], self.settings.summary_tokens,
                                        self.settings.summary_retry_tokens, cache=cache)
        except Exception as exc:  # the summary helps overview questions; indexing goes on without it
            log.warning("document summary failed after a retry", extra={"fields": {"doc_id": doc.doc_id,
                                                                                    "error": str(exc)[:300]}})
            warnings.append(f"document summary failed ({type(exc).__name__}); overview questions may miss it; "
                            "reindex to try again")
            return None
        if not text:
            return None
        if cut:  # kept, and marked: part of a summary still helps overview questions
            warnings.append("document summary reached its token limit even after a retry and was kept incomplete "
                            "(INGEST_SUMMARY_TOKENS, INGEST_SUMMARY_RETRY_TOKENS)")
            text += "\n(This summary was cut at its length limit.)"
        header = context_header(doc.name, "Document summary")
        return Chunk(kind="summary", text=text, search_text=f"{header}\n{text}", n_tokens=self.counter.count(text))

    def step(self, doc: DocumentInfo, text: str) -> None:
        """Show progress for the run; RunLost when the run no longer holds the document."""
        if not self.registry.progress(doc.doc_id, doc.run_id, text):
            raise RunLost(doc.doc_id)

    def _job(self, doc: DocumentInfo) -> JobCache | None:
        """The run's use of the LLM cache; None when it is off (INGEST_LLM_CACHE=false, or no model identity)."""
        if self.cache is None or not self.cache.model:
            return None
        try:
            return self.cache.job(_owner(doc), doc.doc_id, run=self.in_thread,
                                  max_errors=self.settings.llm_cache_max_errors)
        except Exception as exc:  # the cache never fails a job: this run asks the model for everything
            log_fields(log, "llm cache unusable for this document; asking the model", logging.WARNING,
                       doc_id=doc.doc_id, error=f"{type(exc).__name__}: {exc}"[:300])
            return None

    async def process(self, doc: DocumentInfo) -> list[DocumentInfo]:
        """Index one claimed document (doc.run_id is the run); return the earlier versions it superseded (same
        file name), already hidden and still to be removed (remove_superseded). RunLost as soon as the run finds
        it no longer holds the document. The run's LLM cache counts are logged at its end, also when it fails."""
        job = self._job(doc)
        try:
            return await self._index(doc, job)
        finally:
            if job is not None:
                job.log_summary()

    async def _index(self, doc: DocumentInfo, job: JobCache | None) -> list[DocumentInfo]:
        s = self.settings
        owner = _owner(doc)
        ext = self.registry.ext(doc.doc_id)
        self.step(doc, "parsing")
        parsed = await self.parse(doc, ext)

        self.step(doc, "writing tables")
        sources = self.table_sources(parsed)
        schemas = (
            await self.in_thread(write_tables, table_path(s, owner, doc.doc_id), sources) if sources else []
        )
        parsed.warnings.extend(column_warnings(schemas))  # dates stored as numbers, mixed values, lost cells
        await self.describe(doc, parsed, job)

        self.step(doc, "chunking")
        chunks = await self.in_thread(self.chunk, doc, parsed, schemas)  # CPU-bound: keep the loop free
        if not chunks:
            raise ValueError("no indexable content: " + "; ".join(parsed.warnings or ["empty document"]))
        if doc.doc_type.value != "image":  # an image's description already is its summary
            self.step(doc, "summarizing")
            summary = await self.summary_chunk(doc, chunks, schemas, parsed.warnings, job)
            if summary:
                chunks.append(summary)

        self.step(doc, f"embedding {len(chunks)} chunks")
        text_vectors = await self.indexer.embed_texts([c.search_text for c in chunks])
        with_images = [(i, c) for i, c in enumerate(chunks) if c.image]
        image_vectors = dict(
            zip([i for i, _ in with_images], await self.indexer.embed_images([c.image for _, c in with_images]), strict=True)
        )
        for _, c in with_images:
            path = asset_dir(s, owner, doc.doc_id) / c.asset
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(c.image)

        payloads = [self._payload(doc, i, c) for i, c in enumerate(chunks)]
        self.step(doc, "indexing")
        await self.indexer.replace(owner, doc.doc_id, payloads, text_vectors, image_vectors)
        await self.in_thread(
            chunkstore.write,
            chunk_path(s, owner, doc.doc_id),
            [{"id": point_id(doc.doc_id, p.chunk_index), **p.model_dump(exclude={"search_text"})} for p in payloads],
        )

        # A re-uploaded file with new content replaces its earlier versions (same owner, same name) only now
        # that the new version is searchable, in the transaction that marks it ready: never neither, never both.
        done = self.registry.complete(doc.doc_id, doc.run_id, len(chunks), len(with_images), parsed.warnings,
                                      schemas, supersede=s.replace_same_name, index_version=INDEX_VERSION)
        if done is None:
            raise RunLost(doc.doc_id)  # deleted while processing, its account purged, or taken back
        log_fields(log, "document indexed", doc_id=doc.doc_id, chunks=len(chunks), images=len(with_images),
                   tables=len(schemas), warnings=len(parsed.warnings), superseded=len(done.superseded))
        return done.superseded

    async def remove_superseded(self, old: list[DocumentInfo]) -> None:
        """Finish the deletion of superseded versions: points and files, then the row. Never raises: the new
        version is ready whatever happens here, and a version left deleting is finished by the sweep."""
        for doc in old:
            try:
                await delete_document(self.settings, self.registry, self.indexer.delete, doc, doc.run_id)
            except Exception:
                log.exception("removing a superseded version failed; the sweep retries",
                              extra={"fields": {"doc_id": doc.doc_id}})

    async def abandon(self, doc: DocumentInfo) -> str:
        """The run lost its document: deal with what it wrote, by what became of the row. Returns which case.
        - account deleted: every point, folder and cached model reply of the owner goes, including folders the
          job re-created and replies it cached after the purge removed them;
        - deleting under this run (deleted while it ran): points, files, cached replies, then the row;
        - gone (finished by someone else): the points and files this run wrote, and the document's cached
          replies; the upload's file is not the run's;
        - anything else (another run, or a new upload of the same file, holds the id): nothing is touched,
          the files are not this run's."""
        owner, s = _owner(doc), self.settings
        if self.registry.is_deleted(owner):
            await self.indexer.delete(owner, doc.doc_id)
            remove_owner(s, owner)
            await asyncio.to_thread(LlmCache.from_settings(s).purge_owner, owner)
            case = "account deleted"
        else:
            row = self.registry.get_for_worker(doc.doc_id)
            if row is not None and row.status == DocStatus.DELETING and row.run_id == doc.run_id:
                await delete_document(s, self.registry, self.indexer.delete, row, doc.run_id)
                case = "deleted"
            elif row is None:
                await self.indexer.delete(owner, doc.doc_id)
                remove_worker_artifacts(s, owner, doc.doc_id)
                await asyncio.to_thread(LlmCache.from_settings(s).release_document, owner, doc.doc_id)
                case = "gone"
            else:
                return "held elsewhere"
            remove_empty_owner_dirs(s, owner)
        log_fields(log, "cleaned up after a run that lost its document", doc_id=doc.doc_id, case=case)
        return case

    @staticmethod
    def _payload(doc: DocumentInfo, index: int, c: Chunk) -> ChunkPayload:
        return ChunkPayload(
            doc_id=doc.doc_id,
            doc_name=doc.name,
            doc_type=doc.doc_type.value,
            kind=c.kind,
            chunk_index=index,
            text=c.text,
            search_text=c.search_text,
            text_sha=hashlib.sha256(c.text.encode()).hexdigest()[:16],
            n_tokens=c.n_tokens,
            section=c.section,
            page=c.page,
            page_end=c.page_end,
            table_index=c.table_index,
            sheet=c.sheet,
            table=c.table,
            row_start=c.row_start,
            row_end=c.row_end,
            asset=c.asset,
            user_id=doc.owner_id,
        )
