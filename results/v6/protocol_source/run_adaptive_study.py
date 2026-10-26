"""Run the protocol fixed before the v6 intervention experiment."""
from __future__ import annotations
import argparse,json,hashlib,sys,time,platform,os,importlib.metadata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.tsplib_extended import parse_extended
from src.adaptive_study import run_population_extended,run_descent_ils,warmup_extended

OUT=ROOT/'results/v6';DATA=ROOT/'data/v6'
POLICIES={'Guided':'indexed','Crowding':'crowding','Gated':'quality',
          'Adaptive-class':'adaptive_class','Adaptive-slot':'adaptive_slot'}
EXTRA=dict(att48=10628,att532=27686,gr96=55209,gr137=69853,gr202=40160,
           gr229=134602,gr431=171414,gr666=294358,gr120=6942,si175=21407,
           si535=48450,pcb3038=137694)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/'results/v5/dataset_manifest.json').read_text())
    manifest=[]
    for item in old:
        item=dict(item);item.update(original=True,edge_weight_type='EUC_2D')
        manifest.append(item)
    for name,optimum in EXTRA.items():
        p=DATA/(name+'.tsp');ins=parse_extended(p)
        manifest.append(dict(instance=name,dimension=ins.dimension,path=str(p.relative_to(ROOT)),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),suite='TSPLIB',optimum=optimum,
            original=False,edge_weight_type=ins.edge_weight_type,
            source='https://raw.githubusercontent.com/rhgrant10/tsplib95/master/archives/problems/tsp/'+name+'.tsp'))
    for geometry in ['uniform','clustered']:
        for n in [100,250,500,1000]:
            for replica in [2,3]:
                seed=586400+n+1000*replica+(10000 if geometry=='clustered' else 0)
                rng=np.random.default_rng(seed)
                if geometry=='uniform':xy=rng.uniform(0,10000,(n,2))
                else:
                    centers=rng.uniform(1500,8500,(5,2))
                    xy=centers[rng.integers(5,size=n)]+rng.normal(0,300,(n,2))
                name=f'{geometry}{n}_s{replica}';p=DATA/(name+'.tsp')
                content=[f'NAME : {name}','TYPE : TSP',f'DIMENSION : {n}',
                    'EDGE_WEIGHT_TYPE : EUC_2D','NODE_COORD_SECTION']
                content +=[f'{i+1} {x:.8f} {y:.8f}' for i,(x,y) in enumerate(xy)]
                p.write_text('\n'.join(content)+'\nEOF\n')
                manifest.append(dict(instance=name,dimension=n,path=str(p.relative_to(ROOT)),
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),suite='synthetic',optimum=None,
                    original=False,edge_weight_type='EUC_2D',generator_seed=seed,geometry=geometry,
                    source='fixed seeded generator in experiments/run_adaptive_study.py'))
    assert len(manifest)==56
    (OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
    protocol=dict(version='6.0.0',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        status='Exploratory intervention extension informed by inspected v5 results; not preregistered.',
        policies=POLICIES,control='ILS-descent',population=16,trigger=4,
        quality_gate='For uphill crowding only: candidate <= 1.05 * retained incumbent at iteration start. Improving replacements always eligible. No population-wide quality guarantee.',
        iteration_design={'50':'28 original instances, new seeds','250':'all 56 instances, new seeds'},
        seeds=list(range(20001,20031)),main_runs=15120,
        guidance={'beta':6.,'gamma':4.},other_parameters='Identical to v5',
        primary_contrasts=['Gated vs Crowding','Adaptive-class vs Gated',
            'Adaptive-class vs Adaptive-slot','Adaptive-class vs Guided','Guided vs ILS-descent'],
        primary_family='10 tests: five paired instance-mean contrasts at each of two horizons, Holm corrected together',
        secondary='56-instance long-horizon family stratified by original EUC, added TSPLIB, and generated instances; descriptive effects and separate exploratory correction',
        uncertainty='10000 paired instance bootstrap samples; exact Wilcoxon for nonzero untied differences, otherwise Pratt ranks with 50000 seeded sign permutations',
        stopping='All population policies generate 16*iterations candidates; ILS-descent matches Guided delta trials exactly including common initialization',
        measurement='Slot and class influence evaluated each iteration for every population policy; traces every 10 iterations; timing includes online diagnostics',
        exclusions='No post-outcome selection of thresholds or horizons; no parameter tuning on extension outputs. Symmetric instances only because diagnostics use undirected edges.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/adaptive_study.py','src/tsplib_extended.py', 'experiments/run_adaptive_study.py']})
    p=OUT/'protocol.json'
    if not p.exists():p.write_text(json.dumps(protocol,indent=2))
    print('Prepared 56 instances and locked v6 protocol',flush=True)


def block(task):
    item,seed,horizon=task
    target=OUT/'raw_runs'/f"{item['instance']}_{seed}_{horizon}.json"
    if target.exists():return item['instance'],seed,horizon,'cached'
    warmup_extended();ins=parse_extended(ROOT/item['path'])
    records={};base=run_population_extended(ins.dist,seed,'indexed',horizon)
    records['Guided']=base
    methods=[m for m in POLICIES if m!='Guided']+['ILS-descent']
    np.random.default_rng(seed+9123+horizon).shuffle(methods)
    for method in methods:
        records[method]=(run_descent_ils(ins.dist,seed,base['delta_trials']) if method=='ILS-descent'
            else run_population_extended(ins.dist,seed,POLICIES[method],horizon))
    for method,r in records.items():
        t=np.asarray(r['best_tour'])
        if not np.array_equal(np.sort(t),np.arange(ins.dimension)):raise ValueError('Tour permutation')
        if int(ins.dist[t,np.roll(t,-1)].astype(np.int64).sum())!=r['best_length']:raise ValueError('Independent length')
        if r['initial_best']!=base['initial_best']:raise ValueError('Common start')
        r.update(instance=item['instance'],dimension=ins.dimension,seed=seed,method=method,horizon=horizon)
    if records['ILS-descent']['delta_trials']!=base['delta_trials']:raise ValueError('Exact budget')
    tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(records));tmp.replace(target)
    return item['instance'],seed,horizon,'done'


def reference(item):
    import elkai
    name=item['instance'];target=OUT/'reference_runs'/(name+'.json')
    if target.exists():return name,'cached'
    if item['original']:
        target.write_bytes((ROOT/'results/v5/reference_runs'/(name+'.json')).read_bytes())
        return name,'reused v5'
    ins=parse_extended(ROOT/item['path']);problem=(ROOT/item['path']).read_text();records=[]
    for seed in range(1,6):
        params=f'PROBLEM_FILE = :stdin:\nRUNS = 1\nSEED = {seed}\nTRACE_LEVEL = 0\n'
        cpu,wall=time.process_time(),time.perf_counter()
        tour=np.asarray(elkai._elkai.solve_problem(params,problem),dtype=np.int32)-1
        if not np.array_equal(np.sort(tour),np.arange(ins.dimension)):raise ValueError('LKH feasibility')
        records.append(dict(instance=name,seed=seed,best_tour=tour.tolist(),
            best_length=int(ins.dist[tour,np.roll(tour,-1)].astype(np.int64).sum()),parameters=params,
            cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall))
    target.write_text(json.dumps(records));return name,'new reference'


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    p.add_argument('--references',action='store_true');p.add_argument('--workers',type=int,default=6)
    p.add_argument('--instances',nargs='*');p.add_argument('--seeds',type=int,default=30)
    a=p.parse_args()
    if a.prepare:prepare();return
    manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    if a.instances:manifest=[x for x in manifest if x['instance'] in a.instances]
    for q in ['raw_runs','reference_runs']:(OUT/q).mkdir(parents=True,exist_ok=True)
    warmup_extended()
    tasks=manifest if a.references else [(x,s,h) for x in manifest for s in range(20001,20001+a.seeds)
        for h in ([50,250] if x['original'] else [250])]
    (OUT/('environment_reference.json' if a.references else 'environment_experiment.json')).write_text(json.dumps(dict(
        python=sys.version,platform=platform.platform(),workers=a.workers,
        packages={x:importlib.metadata.version(x) for x in ['numpy','numba','scipy','elkai']},
        threads={x:os.environ.get(x) for x in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']}),indent=2))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(reference if a.references else block,t) for t in tasks]
        for k,f in enumerate(as_completed(jobs),1):
            result=f.result()
            if k%10==0 or k==len(tasks):print(json.dumps(dict(completed=k,total=len(tasks),
                latest=result,elapsed=round(time.perf_counter()-start,1))),flush=True)


if __name__=='__main__':main()
