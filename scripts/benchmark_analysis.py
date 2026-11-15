"""Compare installed analysis engines inside the live Python sandbox, without models or user data.

    uv run --python 3.12 scripts/benchmark_analysis.py --macos --output /tmp/analysis-benchmark.json

Uses the same rag -> sandbox path as run_python. Fresh child per engine/workload/size; reads
identical Parquet inputs, validates identical results, reports repeated query times, startup,
and Linux process peak RSS. Synthetic data, warm filesystem cache; not an LLM/search benchmark.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

from check_sandbox_isolation import Stack

# Each child imports only its own engine. Integer results avoid floating-point comparison ambiguity.
CHILD = r'''
import json, resource, statistics, sys, time
started = time.perf_counter()
engine, task, repeats = sys.argv[1], sys.argv[2], int(sys.argv[3])
if engine == 'duckdb':
    import duckdb
    con = duckdb.connect(config={'threads': 2, 'memory_limit': '512MB'})
    queries = {
        'filter': "SELECT count(*), coalesce(sum(id),0) FROM 'orders.parquet' WHERE customer_id=42 AND qty>=2",
        'text_match': "SELECT count(*), coalesce(sum(qty),0) FROM 'orders.parquet' WHERE contains(note,'late delivery')",
        'group': "SELECT region,sum(qty*price_cents) FROM 'orders.parquet' GROUP BY region ORDER BY region",
        'join_group': "SELECT segment,sum(qty*price_cents) FROM 'orders.parquet' JOIN 'customers.parquet' USING(customer_id) GROUP BY segment ORDER BY segment",
    }
    def query():
        return con.sql(queries[task]).fetchall()
elif engine == 'polars':
    import polars as pl
    def query():
        q = pl.scan_parquet('orders.parquet')
        if task == 'filter':
            q = q.filter((pl.col('customer_id')==42) & (pl.col('qty')>=2)).select(pl.len(),pl.col('id').sum())
        elif task == 'text_match':
            q = q.filter(pl.col('note').str.contains('late delivery',literal=True)).select(pl.len(),pl.col('qty').sum())
        else:
            key = 'region'
            if task == 'join_group':
                q = q.join(pl.scan_parquet('customers.parquet'),on='customer_id')
                key = 'segment'
            q = q.group_by(key).agg((pl.col('qty')*pl.col('price_cents')).sum().alias('total')).sort(key)
        return q.collect().rows()
else:
    import pandas as pd
    def query():
        if task == 'filter':
            q = pd.read_parquet('orders.parquet',columns=['id','qty'],filters=[('customer_id','==',42)])
            q = q.loc[q.qty>=2]
            return [(len(q),q.id.sum())]
        if task == 'text_match':
            q = pd.read_parquet('orders.parquet',columns=['note','qty'])
            q = q.loc[q.note.str.contains('late delivery',regex=False)]
            return [(len(q),q.qty.sum())]
        key = 'region' if task == 'group' else 'segment'
        cols = ['region','qty','price_cents'] if task == 'group' else ['customer_id','qty','price_cents']
        q = pd.read_parquet('orders.parquet',columns=cols)
        if task == 'join_group':
            q = q.merge(pd.read_parquet('customers.parquet'),on='customer_id',validate='many_to_one')
        q['total'] = q.qty*q.price_cents
        return list(q.groupby(key,sort=True).total.sum().items())
import_s = time.perf_counter()-started
times=[]
for _ in range(repeats):
    t = time.perf_counter()
    result = query()
    times.append(time.perf_counter()-t)
print(json.dumps({'engine':engine,'task':task,'seconds':times,'median_s':statistics.median(times),
                  'import_s':import_s,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  'result':[[int(x) for x in row] for row in result]}))
'''

DRIVER = r'''
import importlib.metadata, json, os, pathlib, platform, random, subprocess, sys, time
import duckdb
sizes = __SIZES__
repeats = __REPEATS__
child = __CHILD__
results=[]
con=duckdb.connect(config={'threads':2,'memory_limit':'512MB'})
con.sql("COPY (SELECT i AS customer_id,i%7 AS segment FROM range(10000) t(i)) TO 'customers.parquet' (FORMAT PARQUET)")
for n in sizes:
    con.sql(f"""COPY (SELECT i AS id,i%10000 AS customer_id,i%8 AS region,i%20+1 AS qty,
                  i%3000+50 AS price_cents,CASE WHEN i%101=0 THEN 'late delivery'
                  WHEN i%103=0 THEN 'battery overheating' ELSE 'routine shipment' END AS note
                  FROM range({n}) t(i)) TO 'orders.parquet' (FORMAT PARQUET)""")
    for task in ('filter','text_match','group','join_group'):
        baseline=None
        engines=['duckdb','polars','pandas']
        random.Random(f'{n}:{task}').shuffle(engines)
        for engine in engines:
            t=time.perf_counter()
            out=subprocess.run([sys.executable,'-I','-B','-u','-c',child,engine,task,str(repeats)],
                               capture_output=True,text=True,timeout=90)
            if out.returncode:
                raise RuntimeError(f'{engine} {task} failed: {out.stderr[-2000:]}')
            row=json.loads(out.stdout)
            row.update(rows=n,process_s=time.perf_counter()-t)
            if baseline is None:
                baseline=row['result']
            assert row['result']==baseline, f"Result mismatch: {n} {task} {engine}"
            results.append(row)
    pathlib.Path('orders.parquet').unlink()
print(json.dumps({'versions':{p:importlib.metadata.version(p) for p in ('duckdb','polars','pandas','pyarrow')},
                  'platform':platform.platform(),'machine':platform.machine(),
                  'memory_limit_bytes':pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip(),
                  'cpu_max':pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                  'threads':{k:os.getenv(k) for k in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                  'repeats':repeats,'results':results}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--macos', action='store_true', help='use both Mac Compose files and launcher environment')
    parser.add_argument('--rows', nargs='+', type=int, default=[1000, 100000, 1000000])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('/tmp/local-rag-analysis-benchmark.json'))
    args = parser.parse_args()
    if not args.rows or any(n < 100 or n > 2_000_000 for n in args.rows) or not 1 <= args.repeats <= 10:
        parser.error('rows must be 100..2000000; repeats must be 1..10')
    stack = Stack(str(Path(__file__).resolve().parents[1]))
    if args.macos:
        import macos
        import os
        os.environ.update(macos.environment())
        stack.compose = macos.compose_args([])
    code = DRIVER.replace('__SIZES__', repr(args.rows)).replace('__REPEATS__', repr(args.repeats)).replace('__CHILD__', repr(CHILD))
    reply = stack.execute(code, timeout_s=580)
    if reply.get('exit_code') != 0 or reply.get('error') or reply.get('timed_out'):
        sys.exit(json.dumps(reply, indent=2))
    try:
        report = json.loads(reply['stdout'])
    except (KeyError, ValueError):
        sys.exit(json.dumps(reply, indent=2))
    report['host'] = platform.platform()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('rows      workload      engine    median ms   peak MiB')
    for r in report['results']:
        print(f"{r['rows']:9} {r['task']:13} {r['engine']:8} {r['median_s']*1000:10.2f} {r['peak_rss_mib']:10.1f}")
    print(f'All engine results matched. Full measurements: {args.output}')


if __name__ == '__main__':
    main()
