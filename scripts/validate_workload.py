"""Exercise a running stack with synthetic documents in an isolated, disposable account.

Run with Python 3.12 and httpx/openpyxl/psutil installed (the ingest development environment works):
    python scripts/validate_workload.py --macos --output /tmp/workload.json

Needs operator access to this project's Docker services. Never uses an existing user's account,
documents or chats. Passwords/tokens stay in memory; the report contains only synthetic measurements.
The minimum duration includes ingestion; --minutes is not a long-term memory leak certification.
"""

import argparse
import csv
import json
import os
import re
import secrets
import statistics
import subprocess
import tempfile
import threading
import time
from datetime import date
from pathlib import Path

import httpx
import psutil
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]

# Runs inside rag, using the same signed SQL client and generation selection as a real question.
SQL_CHECK = r'''
import asyncio,json,sys
from rag.config import Settings
from rag.clients import SqlClient
from rag_common.contracts import DocumentInfo
from rag_common.http import make_client
from rag_common.identity import Identity,Signer,current_identity
from rag_common.snapshots import published_snapshots
body=json.loads(sys.stdin.readline())
async def main():
    s=Settings()
    token=current_identity.set(Identity(body['owner'],'user','auth'))
    try:
        async with make_client(s.sql_url,timeout=60,signer=Signer.from_file(s.signing_key_file,'rag')) as http:
            with published_snapshots([DocumentInfo(**body['document'])]):
                result=await SqlClient(http).query(body['document']['doc_id'],body['sql'])
                print('VALIDATION_JSON:'+result.model_dump_json())
    finally:
        current_identity.reset(token)
asyncio.run(main())
'''

PYTHON_CHECK = r'''
import asyncio,json,sys
from rag.config import Settings
from rag.clients import SqlClient,SandboxClient
from rag_common.contracts import DocumentInfo
from rag_common.http import make_client
from rag_common.identity import Identity,Signer,current_identity
from rag_common.snapshots import published_snapshots
body=json.loads(sys.stdin.readline())
async def main():
    s=Settings()
    if not s.sandbox_url: raise RuntimeError('enable the Python sandbox before using --python')
    token=current_identity.set(Identity(body['owner'],'user','auth'))
    try:
        async with make_client(s.sql_url,timeout=120,signer=Signer.from_file(s.signing_key_file,'rag')) as http:
            with published_snapshots([DocumentInfo(**body['document'])]):
                data=await SqlClient(http).tables_file(body['document']['doc_id'],128*1024*1024)
        async with make_client(s.sandbox_url,timeout=600) as http:
            reply=await SandboxClient(http).execute(body['code'],60,[('data/tables.duckdb',data)])
            if reply.get('exit_code')!=0 or reply.get('timed_out'):
                raise RuntimeError('Python snapshot check failed')
            print('VALIDATION_JSON:'+json.dumps(json.loads(reply['stdout'])))
    finally:
        current_identity.reset(token)
asyncio.run(main())
'''

CLEANUP = r'''
import asyncio,json,sys
from auth.config import Settings
from auth.store import AuthStore
from auth.cli import _services,_record
from rag_common.identity import OPERATOR_ID
body=json.loads(sys.stdin.readline())
async def main():
    s=Settings(); store=AuthStore(s.db_path)
    user=store.get_user_by_name(body['username'])
    if user is None: return
    if not (user.username.startswith('validation-') and user.user_id==body['owner'] and user.role=='user'):
        raise RuntimeError('refusing to delete an account outside this validation run')
    store.apply(user.user_id,'delete',actor_id=OPERATOR_ID)
    result=await (await _services(s)).purge(user.user_id,OPERATOR_ID)
    assert result.ok, 'synthetic account purge did not finish'
    assert store.delete_user(user.user_id)
    await _record(s,'admin.user_deleted',OPERATOR_ID,user.user_id,by='workload-validation')
asyncio.run(main())
'''


def pdf_fixture(path, pages):
    """A deterministic, text-bearing PDF without a fixture-generation dependency."""
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    kids = []
    for page in range(1, pages + 1):
        page_id, content_id = len(objects) + 1, len(objects) + 2
        kids.append(f'{page_id} 0 R')
        lines = [f'Synthetic operations report - page {page}',
                 'Period | Region | Units | Revenue', f'2026 | North | {page} | {page * 12}',
                 'Repeated header: public synthetic validation data.']
        lines += [f'Paragraph {i}: warehouse inventory shipping and accounting checks for page {page}.'
                  for i in range(1, 22)]
        if page == pages:
            lines.append('FINALPAGEZEBRA: the final page is present and searchable.')
        commands = ['BT /F1 10 Tf 40 755 Td 14 TL']
        for line in lines:
            line = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            commands += [f'({line}) Tj', 'T*']
        stream = ('\n'.join(commands) + '\nET').encode('ascii')
        objects += [f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
                    f'/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>'.encode(),
                    f'<< /Length {len(stream)} >>\nstream\n'.encode() + stream + b'\nendstream']
    objects[1] = f'<< /Type /Pages /Kids [{" ".join(kids)}] /Count {pages} >>'.encode()
    result = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result += f'{index} 0 obj\n'.encode() + obj + b'\nendobj\n'
    start = len(result)
    result += f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode()
    for offset in offsets[1:]:
        result += f'{offset:010d} 00000 n \n'.encode()
    result += f'trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n'.encode()
    path.write_bytes(result)


def fixtures(folder, rows, pages):
    with (folder / 'large-orders.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'region', 'note'])
        writer.writerows((i, 3 * i + 7, 'North' if i % 2 else 'South', 'routine shipment')
                         for i in range(1, rows + 1))
    with (folder / 'messy-unicode.csv').open('w', encoding='utf-16', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['id', 'amount', 'region', 'note'])
        writer.writerows((f'{i:08}', f'{i % 900},25', 'München' if i % 2 else 'São Paulo',
                          'quoted; value\nsecond line' if i % 101 == 0 else None)
                         for i in range(1, min(rows, 50000) + 1))
    wb = Workbook()
    ws = wb.active
    ws.title = 'Quarterly sales'
    ws.merge_cells('A1:F1')
    ws['A1'] = 'Synthetic regional sales — amounts in USD'
    ws.append(['Notes: identifiers have leading zeros; blank amounts are intentional.'])
    ws.append([])
    ws.append(['Order ID', 'Date', 'Region', 'Units', 'Amount', 'Notes'])
    for i in range(1, min(rows, 12000) + 1):
        if i % 3000 == 0:
            ws.append(['Order ID', 'Date', 'Region', 'Units', 'Amount', 'Notes'])
        ws.append([f'{i:08}', date(2026, i % 12 + 1, i % 27 + 1), 'North' if i % 2 else 'South',
                   i % 20, None if i % 113 == 0 else i / 10,
                   'line one\nline two; naïve café' if i % 97 == 0 else 'standard'])
        ws.cell(ws.max_row, 2).number_format = 'yyyy-mm-dd'
        ws.cell(ws.max_row, 5).number_format = '"$"#,##0.00'
    ws.append(['TOTAL', None, None, '=SUM(D5:D12008)', '=SUM(E5:E12008)', 'Uncached formulas are intentional'])
    wb.save(folder / 'messy-workbook.xlsx')
    pdf_fixture(folder / 'long-report.pdf', pages)
    paragraph = ('Synthetic handbook: inventory reconciliation compares recorded stock with counted stock. '
                 'A discrepancy needs an investigation, an approved correction, and a dated audit record.\n\n')
    (folder / 'long-handbook.txt').write_text(paragraph * 6000 + '\nHANDBOOKTAILOTTER: final Unicode note café.\n')
    (folder / 'broken-report.pdf').write_bytes(b'%PDF-1.4\nThis deliberately truncated fixture has no objects.\n')


def memory_bytes(value):
    match = re.fullmatch(r'([\d.]+)\s*([KMGT]?i?B)', value.strip())
    if not match:
        raise ValueError('unknown Docker memory unit')
    unit = match[2]
    factor = 1024 if 'i' in unit else 1000
    power = 'BKMGT'.index(unit[0]) if unit[0] != 'B' else 0
    return int(float(match[1]) * factor ** power)


def healthy_service(before, after):
    if after['health'] != 'healthy' or after['restarts'] != before['restarts']:
        return False
    old_counter, new_counter = before.get('child_oom_kills'), after.get('child_oom_kills')
    if old_counter is not None and new_counter is not None:
        return new_counter == old_counter
    # A historical main-process flag is harmless if no new restart occurred. For services with
    # parser/sandbox children, an unavailable counter cannot clear a flag that is still set.
    if 'child_oom_kills' in after and after['oom_killed']:
        return False
    return not after['oom_killed'] or before['oom_killed']


class Stack:
    def __init__(self, macos):
        self.macos = macos
        if macos:
            from macos import compose_args, environment
            self.command, self.env = compose_args([]), environment()
        else:
            self.command, self.env = ['docker', 'compose', '-f', str(ROOT / 'compose.yaml')], os.environ.copy()

    def call(self, *args, stdin=None, timeout=120):
        result = subprocess.run([*self.command, *args], input=stdin, text=True, capture_output=True,
                                cwd=ROOT, env=self.env, timeout=timeout)
        if result.returncode:
            # Do not echo command input/output: an operator command can contain credentials.
            raise RuntimeError(f'Docker operation {args[0]} failed (exit {result.returncode})')
        return result.stdout

    def python(self, service, program, body):
        return self.call('exec', '-T', service, 'python', '-c', program, stdin=json.dumps(body) + '\n')

    def states(self, ids):
        fmt = '{{index .Config.Labels "com.docker.compose.service"}} {{.State.OOMKilled}} {{.RestartCount}} {{if .State.Health}}{{.State.Health.Status}}{{end}}'
        result = subprocess.run(['docker', 'inspect', '--format', fmt, *ids], capture_output=True, text=True, check=True)
        states = {}
        for container, line in zip(ids, result.stdout.splitlines(), strict=True):
            p = line.split()
            states[p[0]] = {'oom_killed': p[1] == 'true', 'restarts': int(p[2]), 'health': p[3] if len(p) > 3 else ''}
            if p[0] in ('ingest-worker', 'sandbox'):
                # Docker's OOMKilled flag can miss a parser/sandbox child killed while its parent survives.
                events = subprocess.run(['docker', 'exec', container, 'cat', '/sys/fs/cgroup/memory.events'],
                                        capture_output=True, text=True, timeout=10)
                counters = dict(row.split() for row in events.stdout.splitlines()) if events.returncode == 0 else {}
                states[p[0]]['child_oom_kills'] = int(counters['oom_kill']) if 'oom_kill' in counters else None
        return states


class Sampler:
    def __init__(self, stack):
        self.ids = stack.call('ps', '-q').split()
        self.before = stack.states(self.ids)
        self.samples, self.errors = [], 0
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.stage = 'baseline'
        self.groups = set()
        self.native_pids = set()
        record = ROOT / '.runtime' / 'macos-background.json'
        if record.is_file():
            self.groups = set(json.loads(record.read_text()).get('process_groups', {}).values())
        if stack.macos:
            # The documented foreground launch has no process-group record. Read only listener PIDs;
            # do not collect command lines, environment variables or credentials from other processes.
            for port in (18080, 18083):
                try:
                    listeners = subprocess.run(['lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t'],
                                               capture_output=True, text=True, timeout=10)
                    self.native_pids.update(int(pid) for pid in listeners.stdout.split())
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(['docker', 'stats', '--no-stream', '--format', '{{json .}}', *self.ids],
                                        capture_output=True, text=True, timeout=15, check=True)
                containers = {}
                for line in result.stdout.splitlines():
                    row = json.loads(line)
                    # Names themselves are not written; resolve by the ID list and scoped inspect.
                    containers[row['ID']] = memory_bytes(row['MemUsage'].split('/')[0])
                rss = 0
                if self.groups or self.native_pids:
                    processes = subprocess.run(['ps', '-axo', 'pid=,pgid=,rss='], capture_output=True, text=True, check=True)
                    rss = sum(int(r.split()[2]) * 1024 for r in processes.stdout.splitlines()
                              if r.split() and (int(r.split()[0]) in self.native_pids or int(r.split()[1]) in self.groups))
                self.samples.append({'elapsed_s': round(time.monotonic() - self.started, 1), 'stage': self.stage,
                                     'containers': containers, 'native_process_rss_bytes': rss,
                                     'host_available_bytes': psutil.virtual_memory().available,
                                     'host_swap_used_bytes': psutil.swap_memory().used})
            except Exception:
                self.errors += 1
            self.stop.wait(5)

    def finish(self, stack):
        self.stop.set()
        self.thread.join(20)
        after = stack.states(self.ids)
        # Container IDs are useful during sampling, but are unnecessary in a public result.
        names = stack.call('ps', '--format', '{{.ID}} {{.Service}}').splitlines()
        mapping = dict(line.split() for line in names)
        for sample in self.samples:
            sample['containers'] = {next((v for k, v in mapping.items() if k.startswith(key)), 'unknown'): n
                                    for key, n in sample['containers'].items()}
        peaks = {service: max(sample['containers'].get(service, 0) for sample in self.samples)
                 for service in after} if self.samples else {}
        return {'interval_s': 5, 'errors': self.errors, 'before': self.before, 'after': after,
                'native_rss_observed': bool(self.groups or self.native_pids),
                'container_peak_bytes': peaks, 'samples': self.samples,
                'limits': 'Docker working set and native process RSS; RSS excludes some Metal allocations. '
                          'Finite synthetic workload; no assertion of leak freedom or support for every document.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--macos', action='store_true')
    parser.add_argument('--python', action='store_true', help='also check old/new Python data downloads; requires the sandbox')
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--rows', type=int, default=250000)
    parser.add_argument('--pages', type=int, default=100)
    parser.add_argument('--minutes', type=float, default=10)
    parser.add_argument('--document-timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rows <= 1000000 or not 1 <= args.pages <= 500 or not 0 <= args.minutes <= 120:
        parser.error('rows: 1..1000000, pages: 1..500, minutes: 0..120')
    stack = Stack(args.macos)
    report = {'schema': 1, 'rows': args.rows, 'pages': args.pages, 'documents': [], 'checks': {}, 'cleanup': False}
    username, owner = 'validation-' + secrets.token_hex(8), None
    sampler = Sampler(stack)
    sampler.thread.start()
    started = time.monotonic()
    error = None
    try:
        created = stack.call('exec', '-T', 'auth', 'auth-admin', 'create-user', '--username', username,
                             '--role', 'user', '--password-stdin', stdin=secrets.token_urlsafe(36) + '\n')
        owner = re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', created)[0]
        token = stack.call('exec', '-T', 'auth', 'auth-admin', 'create-token', '--username', username,
                           '--name', 'synthetic-validation', '--days', '1').strip().splitlines()[-1]
        with tempfile.TemporaryDirectory(prefix='rag-workload-') as tmp, httpx.Client(
                base_url=args.url, headers={'Authorization': 'Bearer ' + token}, timeout=120) as client:
            folder = Path(tmp)
            fixtures(folder, args.rows, args.pages)

            def document(doc_id):
                return client.get('/api/documents/' + doc_id).raise_for_status().json()

            def search(doc, term):
                # Grep intentionally ignores ordinary words unless quoted as an exact phrase.
                return client.post('/api/search', json={'question': '"' + term + '"', 'doc_ids': [doc['doc_id']],
                    'retrievers': ['grep'], 'rerank': False, 'coverage': False, 'mmr': False,
                    'threshold': False, 'top_k': 5}).raise_for_status().json()['results']

            def sql(doc, statement):
                out = stack.python('rag', SQL_CHECK, {'owner': owner, 'document': doc, 'sql': statement})
                return json.loads(next(line.split('VALIDATION_JSON:', 1)[1] for line in out.splitlines()
                                       if line.startswith('VALIDATION_JSON:')))

            ready = []
            for name in ('large-orders.csv', 'messy-unicode.csv', 'messy-workbook.xlsx', 'long-report.pdf',
                         'long-handbook.txt', 'broken-report.pdf'):
                sampler.stage = name
                stage_start = time.monotonic()
                path = folder / name
                print('Indexing synthetic fixture: ' + name, flush=True)
                with path.open('rb') as file:
                    response = client.post('/api/documents', files={'files': (name, file)}).raise_for_status().json()[0]
                assert not response.get('error'), 'fixture upload rejected: ' + name
                doc_id = response['document']['doc_id']
                while True:
                    doc = document(doc_id)
                    if doc['status'] in ('ready', 'failed'):
                        break
                    assert time.monotonic() - stage_start < args.document_timeout, 'fixture indexing timed out: ' + name
                    time.sleep(3)
                row = {'name': name, 'bytes': path.stat().st_size, 'status': doc['status'], 'chunks': doc['n_chunks'],
                       'elapsed_s': round(time.monotonic() - stage_start, 1),
                       'tables': [{'name': t['table'], 'rows': t['rows']} for t in doc['tables']],
                       'warning_count': len(doc['warnings'])}
                report['documents'].append(row)
                assert doc['status'] == ('failed' if name == 'broken-report.pdf' else 'ready'), 'unexpected result: ' + name
                if doc['status'] == 'ready':
                    ready.append(doc)
                if name == 'large-orders.csv':
                    table = doc['tables'][0]['table']
                    result = sql(doc, f'SELECT count(*),sum(x),sum(y) FROM "{table}"')
                    total = args.rows * (args.rows + 1) // 2
                    assert result['rows'] == [[args.rows, total, 3 * total + 7 * args.rows]], 'large CSV aggregate mismatch'
                    report['checks']['large_csv_exact_aggregates'] = True
                if name in ('long-report.pdf', 'long-handbook.txt'):
                    term = 'FINALPAGEZEBRA' if name.endswith('.pdf') else 'HANDBOOKTAILOTTER'
                    hits = search(doc, term)
                    assert hits and any(term in hit['text'] for hit in hits), 'end-of-document content missing'
                    report['checks'][name + '_tail_search'] = True

            # Re-index the large table while repeatedly reading its published generation.
            old = ready[0]
            sampler.stage = 'reindex-with-readers'
            print('Reindexing the large CSV while checking concurrent searches.', flush=True)
            client.post('/api/documents/' + old['doc_id'] + '/reindex').raise_for_status()
            counts = {'old': 0, 'new': 0}
            reindex_start = time.monotonic()
            new = None
            while True:
                hits = search(old, 'routine shipment')
                assert hits and len({hit['generation'] for hit in hits}) == 1, 'mixed or missing snapshot'
                current = document(old['doc_id'])
                generation = hits[0]['generation']
                assert generation in (old['generation'], current['generation']), 'unexpected published generation'
                counts['old' if generation == old['generation'] else 'new'] += 1
                if current['status'] == 'ready' and current['generation'] != old['generation']:
                    new = current
                    break
                assert current['status'] != 'failed', 'reindex failed'
                assert time.monotonic() - reindex_start < args.document_timeout, 'reindex timed out'
                time.sleep(2)
            assert all(h['generation'] == new['generation'] for h in search(new, 'routine shipment'))
            table = old['tables'][0]['table']
            assert sql(old, f'SELECT count(*) FROM "{table}"')['rows'] == [[args.rows]]
            assert sql(new, f'SELECT count(*) FROM "{table}"')['rows'] == [[args.rows]]
            if args.python:
                print('Checking old and new table generations with isolated pandas runs.', flush=True)
                code = ("import duckdb,json\ncon=duckdb.connect('data/tables.duckdb',read_only=True)\n"
                        + "df=con.sql(" + repr(f'SELECT x,y FROM "{table}"') + ").df()\n"
                        + "print(json.dumps({'rows':len(df),'sum_x':int(df.x.sum()),'sum_y':int(df.y.sum())}))")
                for snapshot in (old, new):
                    out = stack.python('rag', PYTHON_CHECK, {'owner': owner, 'document': snapshot, 'code': code})
                    value = json.loads(next(line.split('VALIDATION_JSON:', 1)[1] for line in out.splitlines()
                                            if line.startswith('VALIDATION_JSON:')))
                    total = args.rows * (args.rows + 1) // 2
                    assert value == {'rows': args.rows, 'sum_x': total, 'sum_y': 3 * total + 7 * args.rows}
                report['checks']['old_and_new_python_snapshots'] = True
            report['checks']['concurrent_reindex'] = counts | {'old_and_new_sql_readable': True,
                'elapsed_s': round(time.monotonic() - reindex_start, 1)}

            # Repeat a large PDF after the spreadsheet/embedding workload, where retained native
            # parser caches previously exhausted the worker's 4 GiB container.
            old_pdf = next(doc for doc in ready if doc['name'] == 'long-report.pdf')
            sampler.stage = 'repeat-pdf'
            print('Repeating the large PDF after the spreadsheet workload.', flush=True)
            repeated_start = time.monotonic()
            client.post('/api/documents/' + old_pdf['doc_id'] + '/reindex').raise_for_status()
            while True:
                repeated = document(old_pdf['doc_id'])
                assert repeated['status'] != 'failed', 'repeated PDF indexing failed'
                if repeated['status'] == 'ready' and repeated['generation'] != old_pdf['generation']:
                    break
                assert time.monotonic() - repeated_start < args.document_timeout, 'repeated PDF timed out'
                time.sleep(3)
            assert any('FINALPAGEZEBRA' in hit['text'] for hit in search(repeated, 'FINALPAGEZEBRA'))
            client.get('/api/documents/' + old_pdf['doc_id'] + '/chunks', params={
                'generation': old_pdf['generation'], 'limit': 100}).raise_for_status()
            report['checks']['repeated_pdf'] = {'passed': True, 'elapsed_s': round(time.monotonic() - repeated_start, 1)}

            sampler.stage = 'repeated-reads'
            print('Repeating searches and chunk reads to check sustained memory use.', flush=True)
            read_latencies = []
            cycles = 0
            while cycles < 30 or time.monotonic() - started < args.minutes * 60:
                for doc in ready:
                    begin = time.monotonic()
                    term = 'FINALPAGEZEBRA' if doc['name'].endswith('.pdf') else 'North'
                    search(doc, term)
                    client.get('/api/documents/' + doc['doc_id'] + '/chunks', params={
                        'generation': doc['generation'], 'offset': 0, 'limit': 100}).raise_for_status()
                    read_latencies.append(time.monotonic() - begin)
                cycles += 1
                time.sleep(1)
            report['checks']['repeated_reads'] = {'cycles': cycles, 'requests': len(read_latencies) * 2,
                'median_pair_s': round(statistics.median(read_latencies), 3),
                'max_pair_s': round(max(read_latencies), 3)}
            sampler.stage = 'cooldown'
            print('Workload complete; observing memory recovery for 30 seconds.', flush=True)
            time.sleep(30)
    except BaseException as exc:
        # Error strings can contain response bodies or credentials. Retain only the class in the report.
        report['failure_type'] = type(exc).__name__
        error = exc
    finally:
        if owner:
            try:
                sampler.stage = 'cleanup'
                stack.python('auth', CLEANUP, {'username': username, 'owner': owner})
                report['cleanup'] = True
            except Exception:
                report['cleanup'] = False
                print('Cleanup needs operator attention for the synthetic account: ' + username, flush=True)
        report['elapsed_s'] = round(time.monotonic() - started, 1)
        report['memory'] = sampler.finish(stack)
        report['passed'] = (error is None and report['cleanup'] and bool(report['memory']['samples'])
                            and report['memory']['errors'] == 0
                            and all(healthy_service(report['memory']['before'][name], state)
                                    for name, state in report['memory']['after'].items()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    if error:
        raise error
    if not report['passed']:
        raise SystemExit('Workload checks or memory monitoring failed; inspect the report.')
    print('Synthetic workload passed; temporary account and documents removed.', flush=True)


if __name__ == '__main__':
    main()
