"""ingest-worker: claims queued documents from the registry and runs the pipeline.

The worker holds no signing key and receives no identity: the owner of each job is in the registry row it
claims, and is used for file paths, the Qdrant payload and the audit trail.

One job is one run (registry.py): while it runs, the worker renews the run's lease every few seconds. When a
renewal or any other write of the run finds the row no longer processing under it (the user deleted the
document, the account was purged), the run stops at once, the parser child included, waits for the job's
threads, and cleans up what is its to clean (Pipeline.abandon). A sweep, on a timer next to the jobs, takes
back jobs whose lease ran out and finishes deletions nobody holds. At start the worker takes back at once
what its own earlier process held (its id: INGEST_WORKER_ID, the host name by default), and re-queues the
ready documents an older index version built (pipeline.INDEX_VERSION). The LLM cache (llmcache.py) is checked
and bounded at start, and the sweep removes cached replies of documents that no longer exist."""

import asyncio
import contextlib
import logging
import os
import signal
import socket
import time

from rag_common.audit import AuditClient
from rag_common.contracts import DocumentInfo
from rag_common.http import make_client
from rag_common.index import ensure_collection, qdrant_client
from rag_common.logging import configure_logging, log_fields

from ingest.chunking import TokenCounter
from ingest.config import get_settings
from ingest.indexer import Indexer
from ingest.isolation import IsolatedRunner
from ingest.llmcache import LlmCache, model_identity
from ingest.pipeline import INDEX_VERSION, Pipeline, RunLost, delete_document
from ingest.registry import Registry

log = logging.getLogger("ingest.worker")
HEARTBEAT_S = 10  # the worker.heartbeat file, and the lease renewal of the running job


async def _heartbeat(path, stop: asyncio.Event) -> None:
    while not stop.is_set():
        path.write_text(str(time.time()))
        try:
            await asyncio.wait_for(stop.wait(), HEARTBEAT_S)
        except TimeoutError:
            pass


async def _hold(registry: Registry, doc: DocumentInfo, every_s: float) -> None:
    """Renew the run's lease until the run loses its row; then return. A renewal that fails (the database
    busy for longer than its timeout) is retried: only the row's answer ends the run."""
    while True:
        try:
            if not registry.renew(doc.doc_id, doc.run_id):
                return
        except Exception:
            log.exception("renewing the job's lease failed; retrying", extra={"fields": {"doc_id": doc.doc_id}})
        await asyncio.sleep(every_s)


def _stop_parsing(pipeline: Pipeline) -> None:
    """A cancelled job's parse keeps its thread and the parser child busy: kill the child (a new one starts
    with the next job)."""
    close = getattr(getattr(pipeline, "runner", None), "close", None)
    if close:
        close()


async def process_until_stopped(pipeline: Pipeline, registry: Registry, doc: DocumentInfo, stop: asyncio.Event,
                                renew_s: float = HEARTBEAT_S) -> list[DocumentInfo] | None:
    """Process one document, or on shutdown put it back in the queue without counting the attempt.
    Returns the versions it replaced, or None when it was stopped. RunLost when the run lost its row."""
    work = asyncio.create_task(pipeline.process(doc))
    halt = asyncio.create_task(stop.wait())
    hold = asyncio.create_task(_hold(registry, doc, renew_s))
    try:
        await asyncio.wait({work, halt, hold}, return_when=asyncio.FIRST_COMPLETED)
        if work.done():
            return work.result()
        work.cancel()
        _stop_parsing(pipeline)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await work
        # Keep renewing the claim until writers have stopped, including on shutdown. Releasing it
        # earlier lets another worker write the same files beside an uncancellable old thread.
        settle = getattr(pipeline, "settle", None)
        if settle:
            await settle()
        if stop.is_set() and registry.release(doc.doc_id, doc.run_id):
            log_fields(log, "stopping: document back in the queue", doc_id=doc.doc_id)
            return None
        raise RunLost(doc.doc_id)
    finally:
        halt.cancel()
        hold.cancel()
        await asyncio.gather(halt, hold, return_exceptions=True)


async def abandon_quietly(pipeline: Pipeline, doc: DocumentInfo) -> None:
    """Pipeline.abandon, logged instead of raised: the worker goes on with the next job either way (a deletion
    left unfinished is finished by the sweep)."""
    try:
        await pipeline.abandon(doc)
    except Exception:
        log.exception("cleaning up after a deleted document failed", extra={"fields": {"doc_id": doc.doc_id}})


async def run_job(pipeline: Pipeline, registry: Registry, audit: AuditClient, doc: DocumentInfo,
                  stop: asyncio.Event) -> bool:
    """Process one claimed document and record the outcome; False when the worker was stopped meanwhile."""
    started = time.perf_counter()
    log_fields(log, "processing", doc_id=doc.doc_id, name=doc.name)
    try:
        replaced = await process_until_stopped(pipeline, registry, doc, stop)
        if replaced is None:
            return False
        seconds = round(time.perf_counter() - started, 1)
        for old in replaced:
            await audit.record("document.superseded", old.doc_id, user_id=doc.owner_id, name=old.name,
                               replaced_by=doc.doc_id)
        log_fields(log, "done", doc_id=doc.doc_id, seconds=seconds)
        info = registry.get_for_worker(doc.doc_id)
        if info:
            await audit.record("document.ready", doc.doc_id, user_id=doc.owner_id, name=info.name,
                               chunks=info.n_chunks, images=info.n_images, tables=[t.table for t in info.tables],
                               warnings=info.warnings, seconds=seconds)
        await pipeline.remove_superseded(replaced)  # separately: a failure here never touches the new version
    except RunLost:  # deleted (or its account was) while it ran: what the job wrote must not stay behind
        await abandon_quietly(pipeline, doc)
    except Exception as exc:
        log.exception("document failed", extra={"fields": {"doc_id": doc.doc_id}})
        error = f"{type(exc).__name__}: {exc}"
        if registry.fail(doc.doc_id, doc.run_id, error):
            await audit.record("document.failed", doc.doc_id, user_id=doc.owner_id, name=doc.name,
                               error=error[:2000])
        else:
            await abandon_quietly(pipeline, doc)
    return True


async def sweep(pipeline: Pipeline, registry: Registry, max_attempts: int, mine: bool = False) -> None:
    """Take back jobs whose run stopped renewing its lease, and finish deletions nobody holds (a run that died
    after its document was deleted, a clean-up that failed). With mine (the first sweep after a start), also
    what this worker's earlier process held. Then the LLM cache: the uses of documents the registry no longer
    has (a run that cached replies after its document was deleted, and died), and the entries nobody uses.
    Errors are logged; the next sweep retries."""
    try:
        if taken := registry.recover(max_attempts, mine=mine):
            log_fields(log, "took back interrupted jobs", jobs=taken)
        for doc in registry.deleting(mine=mine):
            claim = registry.claim_deleting(doc.doc_id, mine=mine)
            if claim is not None and await delete_document(pipeline.settings, registry, pipeline.indexer.delete,
                                                           doc, claim):
                log_fields(log, "finished a deletion", doc_id=doc.doc_id)
    except Exception:
        log.exception("sweep failed; retried at the next one")
    try:
        if removed := await asyncio.to_thread(LlmCache.from_settings(pipeline.settings).collect, registry.owned_ids):
            log_fields(log, "removed cached model replies of deleted documents", rows=removed)
    except Exception:
        log.exception("llm cache sweep failed; retried at the next one")


async def sweeper(pipeline: Pipeline, registry: Registry, max_attempts: int, every_s: float,
                  stop: asyncio.Event) -> None:
    """Lease-aware cleanup on a timer, also while a job runs. Own-holder recovery runs before claims."""
    while not stop.is_set():
        await sweep(pipeline, registry, max_attempts)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), every_s)


async def open_cache(cache: LlmCache, settings, vision) -> None:
    """The LLM cache at start: checked (a damaged file starts again) and bounded also when it is off, so what it
    holds stays bounded and purgeable; it serves jobs (cache.model) only when on and the model is known. A
    cache that cannot be opened stays off until the next start (logged): jobs ask the model."""
    try:
        entries, size = await asyncio.to_thread(cache.open, settings.llm_cache)
    except Exception as exc:
        log_fields(log, "llm cache could not be opened; off until the worker restarts", logging.ERROR,
                   error=f"{type(exc).__name__}: {exc}"[:300])
        return
    if not settings.llm_cache:
        log_fields(log, "llm cache off (INGEST_LLM_CACHE=false): every reply comes from the model", entries=entries)
        return
    cache.model = await model_identity(settings, vision)
    if cache.model:
        log_fields(log, "llm cache on", entries=entries, mb=round(size / 2**20, 1), max_mb=settings.llm_cache_max_mb,
                   max_entries=settings.llm_cache_max_entries, model=cache.model)


async def _wait_for_models(indexer: Indexer, stop: asyncio.Event):
    while not stop.is_set():
        try:
            return await indexer.info()
        except Exception as exc:  # models still loading
            log_fields(log, "waiting for models service", error=str(exc))
            await asyncio.sleep(5)
    return None


async def run() -> None:
    s = get_settings()
    configure_logging(s.log_level)
    for path in (s.files_dir, s.assets_dir, s.tables_dir, s.chunks_dir):
        path.mkdir(parents=True, exist_ok=True)
    registry = Registry(s.registry_path, lease_s=s.run_lease_s, holder=s.worker_id or socket.gethostname())
    registry.recover(s.max_attempts, mine=True)  # what this worker held before it stopped is taken back now
    if stale := registry.requeue_stale(INDEX_VERSION):  # a parsing or chunking fix: rebuild what it missed
        log_fields(log, "re-queued documents an older index version built", logging.WARNING, documents=stale,
                   index_version=INDEX_VERSION)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows dev runs
            pass

    qdrant = qdrant_client(s.qdrant_url, s.qdrant_api_key_file, timeout=int(s.qdrant_timeout_s))
    models = make_client(s.models_url, timeout=s.models_timeout_s)
    vision = make_client(s.vision_url, timeout=s.vision_timeout_s)
    indexer = Indexer(qdrant, models, s.embed_batch, s.image_embed_batch, s.upsert_batch, s.retry_backoff_s)
    os.environ["INGEST_RAW_TABLE_MAX_ROWS"] = str(s.raw_table_max_rows)  # the parse child reads it (layout.py)
    runner = IsolatedRunner(max_cache_mb=s.parser_cache_mb)
    audit = AuditClient(s.audit_url, "ingest-worker")
    heartbeat = asyncio.create_task(_heartbeat(s.data_dir / "worker.heartbeat", stop))
    sweeping = None
    try:
        info = await _wait_for_models(indexer, stop)
        if info is None:
            return
        await ensure_collection(qdrant, info)
        cache = LlmCache.from_settings(s)
        await open_cache(cache, s, vision)
        pipeline = Pipeline(s, registry, indexer, vision, TokenCounter(s.tokenizer_file), runner, cache)
        # Own-holder recovery assumes an earlier process is dead. Finish it before claiming any new
        # work; a scheduled startup sweep could otherwise reclaim this process's first fresh job.
        await sweep(pipeline, registry, s.max_attempts, mine=True)
        log.info("worker ready")
        sweeping = asyncio.create_task(sweeper(pipeline, registry, s.max_attempts, s.sweep_interval_s, stop))
        while not stop.is_set():
            doc = registry.claim_next()
            if doc is None:
                try:
                    await asyncio.wait_for(stop.wait(), s.poll_interval_s)
                except TimeoutError:
                    pass
                continue
            if not await run_job(pipeline, registry, audit, doc, stop):
                break
    finally:
        stop.set()
        await heartbeat
        if sweeping is not None:
            with contextlib.suppress(Exception):
                await sweeping
        runner.close()
        await audit.aclose()
        await models.aclose()
        await vision.aclose()
        await qdrant.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
