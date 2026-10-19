"""Reproduce the v5 guidance by survival experiment and LKH reference runs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tsplib_parser import parse_tsp
from src.component_study import (CONFIG,run_population,run_ils_control,warmup,length)

ADDITIONAL = dict(eil101=629,a280=2579,rd400=15281,u574=36905,
                  rat575=6773,d657=48912,rat783=8806,pr1002=259045,pcb1173=56892)
ORIGINAL = dict(berlin52=7542,ch150=6528,d493=35002,eil76=538,kroA100=21282,
                lin318=42029,pcb442=50778,pr439=107217,pr76=108159,rat99=1211,tsp225=3916)
OUT = ROOT/'results'/'v5'
DATA = ROOT/'data'/'v5'
METHODS = ['Guided','Neutral','Guided-crowding','Neutral-crowding','ILS-800','ILS-work']


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    def download(name):
        target = DATA/f'{name}.tsp'
        url = f'https://raw.githubusercontent.com/rhgrant10/tsplib95/master/archives/problems/tsp/{name}.tsp'
        if not target.exists():
            with urllib.request.urlopen(url,timeout=60) as response:
                target.write_bytes(response.read())
        parse_tsp(target)
        return name,url
    with ThreadPoolExecutor(max_workers=5) as pool:
        sources = dict(pool.map(download,ADDITIONAL))
    for geometry in ['uniform','clustered']:
        for n in [100,250,500,1000]:
            seed = 286400+n+(10000 if geometry=='clustered' else 0)
            rng = np.random.default_rng(seed)
            if geometry=='uniform':
                xy = rng.uniform(0,10000,(n,2))
            else:
                centers = rng.uniform(1500,8500,(5,2))
                xy = centers[rng.integers(5,size=n)]+rng.normal(0,300,(n,2))
            name = f'{geometry}{n}'
            lines = [f'NAME : {name}','TYPE : TSP',f'DIMENSION : {n}',
                     'EDGE_WEIGHT_TYPE : EUC_2D','NODE_COORD_SECTION']
            lines += [f'{i+1} {x:.8f} {y:.8f}' for i,(x,y) in enumerate(xy)]
            (DATA/f'{name}.tsp').write_text('\n'.join(lines)+'\nEOF\n')
    manifest = []
    for p in sorted(list((ROOT/'data').glob('*.tsp'))+list(DATA.glob('*.tsp'))):
        instance = parse_tsp(p)
        name = p.stem
        manifest.append(dict(instance=name,dimension=instance.dimension,
             path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
             suite='TSPLIB' if name in ORIGINAL or name in ADDITIONAL else 'synthetic',
             optimum={**ORIGINAL,**ADDITIONAL}.get(name),
             source=sources.get(name,'TSPLIB bundled in v4' if name in ORIGINAL else 'fixed synthetic generator in this script')))
    (OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
    protocol = dict(version='5.0.0',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        main_config=CONFIG,methods=METHODS,seeds=list(range(10001,10031)),
        instance_count=len(manifest),primary_unit='instance mean over 30 paired runs',
        primary_contrasts=['Guided vs Neutral','Guided-crowding vs Guided',
                           'Neutral-crowding vs Neutral','Guided vs ILS-800','Guided vs ILS-work'],
        primary_endpoint='relative excess to documented reference length',
        multiplicity='Holm across five paired instance-level Wilcoxon tests',
        effect_sizes='median paired difference and paired rank-biserial correlation',
        bootstrap='10000 paired instance resamples, seed 92614',
        reference='TSPLIB published optimum, synthetic best of five seeded LKH runs',
        timing='per-process CPU and elapsed wall time; JIT warmup excluded',
        work_control='ILS-work matches Guided 2-opt delta trials including common initialization; complete-candidate overshoot retained and disclosed',
        notes='Prospective extension of an already inspected v4 study; not a preregistered trial. No post-outcome parameter selection.')
    path = OUT/'protocol.json'
    if not path.exists():
        path.write_text(json.dumps(protocol,indent=2))
    print(f'Prepared {len(manifest)} instances',flush=True)


def block(task):
    item,seed,cache = task
    target = Path(cache)/f"{item['instance']}_{seed}.json"
    if target.exists():
        return item['instance'],seed,'cached'
    warmup()
    ins = parse_tsp(ROOT/item['path'])
    guided = run_population(ins.dist,seed,True,False)
    records = {'Guided':guided}
    order = ['Neutral','Guided-crowding','Neutral-crowding','ILS-800','ILS-work']
    np.random.default_rng(seed+7241).shuffle(order)
    for method in order:
        if method=='ILS-800':
            records[method] = run_ils_control(ins.dist,seed)
        elif method=='ILS-work':
            records[method] = run_ils_control(ins.dist,seed,guided['delta_trials'])
        else:
            records[method] = run_population(ins.dist,seed,method.startswith('Guided'),method.endswith('crowding'))
    for method,r in records.items():
        if sorted(r['best_tour'])!=list(range(ins.dimension)):
            raise ValueError('Infeasible tour')
        if int(length(np.asarray(r['best_tour']),ins.dist))!=r['best_length']:
            raise ValueError('Tour length mismatch')
        r.update(instance=item['instance'],dimension=ins.dimension,seed=seed,method=method)
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(records))
    temp.replace(target)
    return item['instance'],seed,'done'


def reference_block(item):
    import elkai
    target = OUT/'reference_runs'/f"{item['instance']}.json"
    if target.exists():
        return item['instance'],'cached'
    ins = parse_tsp(ROOT/item['path'])
    problem = (ROOT/item['path']).read_text()
    records = []
    for seed in range(1,6):
        params = f'PROBLEM_FILE = :stdin:\nRUNS = 1\nSEED = {seed}\nTRACE_LEVEL = 0\n'
        start_cpu,start_wall = time.process_time(),time.perf_counter()
        solution = elkai._elkai.solve_problem(params,problem)
        t = np.asarray(solution,dtype=np.int32)-1
        if sorted(t)!=list(range(ins.dimension)):
            raise ValueError('LKH returned infeasible tour')
        records.append(dict(instance=item['instance'],seed=seed,best_length=int(length(t,ins.dist)),
            best_tour=t.tolist(),cpu_seconds=time.process_time()-start_cpu,
            wall_seconds=time.perf_counter()-start_wall,parameters=params))
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(records))
    return item['instance'],'done'


def enrich_block(task):
    item,seed,cache = task
    target = Path(cache)/f"{item['instance']}_{seed}.json"
    records = json.loads(target.read_text())
    if 'q10_count' in records['Guided']:
        return item['instance'],seed,'cached'
    warmup()
    ins = parse_tsp(ROOT/item['path'])
    for method in METHODS[:4]:
        original = records[method]
        replay = run_population(ins.dist,seed,method.startswith('Guided'),method.endswith('crowding'))
        for field in ['best_length','best_tour','delta_trials','candidate_count','initial_best']:
            if original[field]!=replay[field]:
                raise AssertionError(f'Replay mismatch: {item["instance"]} {seed} {method} {field}')
        for row,new_row in zip(original['trace'],replay['trace']):
            for field in ['edge_diversity','neff','clone_neff']:
                if not np.isclose(row[field],new_row[field],rtol=0,atol=1e-12):
                    raise AssertionError('Diagnostic replay mismatch')
            row.update({k:v for k,v in new_row.items() if k.startswith('q')})
        original.update({k:v for k,v in replay.items() if k.startswith('q')})
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(records))
    temp.replace(target)
    return item['instance'],seed,'verified and enriched'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--references',action='store_true')
    p.add_argument('--enrich',action='store_true')
    p.add_argument('--runs',type=int,default=30)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--cache',default=str(OUT/'raw_runs'))
    p.add_argument('--instances',nargs='*')
    args = p.parse_args()
    if args.prepare:
        prepare()
        return
    manifest = json.loads((OUT/'dataset_manifest.json').read_text())
    if args.instances:
        manifest = [x for x in manifest if x['instance'] in args.instances]
    Path(args.cache).mkdir(parents=True,exist_ok=True)
    env = dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
        cpu_count=os.cpu_count(),workers=args.workers,
        packages={k:importlib.metadata.version(k) for k in ['numpy','numba','scipy','elkai']})
    env['thread_limits'] = {k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']}
    env_file = 'environment_reference.json' if args.references else ('environment_replay.json' if args.enrich else 'environment_experiment.json')
    (OUT/env_file).write_text(json.dumps(env,indent=2))
    warmup()
    tasks = manifest if args.references else [(x,s,args.cache) for x in manifest for s in range(10001,10001+args.runs)]
    fn = reference_block if args.references else (enrich_block if args.enrich else block)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fn,t) for t in tasks]
        for k,f in enumerate(as_completed(futures),1):
            result = f.result()
            if k%5==0 or k==len(tasks):
                print(json.dumps(dict(completed=k,total=len(tasks),latest=result,
                    elapsed_seconds=round(time.perf_counter()-start,1))),flush=True)


if __name__=='__main__':
    main()
