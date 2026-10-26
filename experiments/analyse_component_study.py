"""Validate and analyse all v5 tours; build the paper's numeric source tables."""
from __future__ import annotations
import gzip
import io
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,rankdata,friedmanchisquare,PermutationMethod

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tsplib_parser import parse_tsp
from src.component_study import length
from experiments.run_component_study import METHODS
OUT=ROOT/'results'/'v5'
CONTRASTS=[('Guided','Neutral'),('Guided-crowding','Guided'),
           ('Neutral-crowding','Neutral'),('Guided','ILS-800'),('Guided','ILS-work')]


def holm(values):
    p=np.asarray(values,dtype=float)
    order=np.argsort(p)
    adjusted=np.minimum(1.,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    result=np.empty_like(p)
    result[order]=adjusted
    return result


def interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(92614)
    samples=values[rng.integers(len(values),size=(10000,len(values)))]
    lo,hi=np.quantile(np.median(samples,axis=1),[.025,.975])
    return float(lo),float(hi)


def paired_stats(a,b,label):
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    # Stable rounding avoids false distinctions caused solely by arithmetic.
    d=np.round(d,12)
    ranks=rankdata(np.abs(d))
    signed_mass=ranks[d!=0].sum()
    rbc=0. if signed_mass==0 else (ranks[d<0].sum()-ranks[d>0].sum())/signed_mass
    if np.all(d==0):
        stat,p=0.,1.
        p_method='all ties'
    else:
        untied=bool(np.all(d!=0) and len(np.unique(np.abs(d)))==len(d))
        p_method='exact signed-rank' if untied else '50000 sign permutations'
        result=wilcoxon(d,zero_method='pratt',alternative='two-sided',
            method='exact' if untied else PermutationMethod(n_resamples=50000,rng=np.random.default_rng(86123)))
        stat,p=float(result.statistic),float(result.pvalue)
    lo,hi=interval(d)
    return dict(comparison=label,n_instances=len(d),mean_delta=float(d.mean()),
        median_delta=float(np.median(d)),median_ci_low=lo,median_ci_high=hi,
        wins=int(np.sum(d<0)),ties=int(np.sum(d==0)),losses=int(np.sum(d>0)),
        statistic=stat,p_raw=p,p_method=p_method,rank_biserial_favour_first=float(rbc))


def write_gzip(path,records):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as z:
            for row in records:
                z.write((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode())


def write_csv(path,frame,compress=False):
    """Serialize a complete table before atomically publishing checked bytes."""
    payload=frame.to_csv(index=False).encode('utf-8')
    if compress:
        payload=gzip.compress(payload,mtime=0)
    temporary=path.with_name(path.name+'.tmp')
    written=temporary.write_bytes(payload)
    if written!=len(payload) or temporary.read_bytes()!=payload:
        raise IOError('Incomplete CSV export: '+str(path))
    temporary.replace(path)


def load_records():
    raw=sorted((OUT/'raw_runs').glob('*.json'))
    if raw:
        return [r for p in raw for r in json.loads(p.read_text()).values()]
    with gzip.open(OUT/'run_records.jsonl.gz','rt') as f:
        return [json.loads(line) for line in f]


def load_references():
    raw=sorted((OUT/'reference_runs').glob('*.json'))
    if raw:
        return [r for p in raw for r in json.loads(p.read_text())]
    with gzip.open(OUT/'reference_records.jsonl.gz','rt') as f:
        return [json.loads(line) for line in f]


def main():
    manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    records=load_records()
    references=load_references()
    assert len(records)==28*30*6,(len(records),'incomplete main runs')
    assert len(references)==28*5,(len(references),'incomplete references')
    for item in manifest:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    distances={x['instance']:parse_tsp(ROOT/x['path']).dist for x in manifest}
    all_records=records+references
    for r in all_records:
        d=distances[r['instance']]
        assert sorted(r['best_tour'])==list(range(len(d)))
        tour=np.asarray(r['best_tour'],dtype=np.int64)
        # Independent NumPy implementation, rather than the search evaluator.
        assert int(d[tour,np.roll(tour,-1)].sum(dtype=np.int64))==r['best_length']
    lookup={x['instance']:x for x in manifest}
    ref_df=pd.DataFrame([{k:v for k,v in r.items() if k not in ['best_tour','parameters']} for r in references])
    for item in manifest:
        ref=ref_df[ref_df.instance==item['instance']]
        item['reference_length']=item['optimum'] or int(ref.best_length.min())
        item['reference_kind']='published optimum' if item['optimum'] else 'best feasible LKH reference'
        item['lkh_best']=int(ref.best_length.min())
        item['lkh_mean']=float(ref.best_length.mean())
        item['lkh_cpu_mean']=float(ref.cpu_seconds.mean())
        if item['optimum']:
            assert ref.best_length.min()>=item['optimum']
    rows,traces=[],[]
    for r in records:
        item=lookup[r['instance']]
        ref=item['reference_length']
        assert r['best_length']<=r['initial_best']
        if item['optimum']:
            assert r['best_length']>=item['optimum']
        row={k:v for k,v in r.items() if k not in ['trace','best_tour','guidance','crowding']}
        row.update(suite=item['suite'],reference_length=ref,reference_kind=item['reference_kind'],
            excess_pct=100*(r['best_length']-ref)/ref,
            improvement_from_start_pct=100*(r['initial_best']-r['best_length'])/r['initial_best'])
        rows.append(row)
        for t in r['trace']:
            traces.append(dict(instance=r['instance'],seed=r['seed'],method=r['method'],**t))
        if r['method'] in METHODS[:4]:
            assert 'q10_count' in r,'secondary replay incomplete'
            assert 1-1e-9<=r['clone_neff']<=r['neff']+1e-9<=16+1e-8
            assert 0<=r['edge_diversity']<=1
            assert np.isclose(np.log(r['neff']/r['clone_neff']),r['entropy_inflation'])
            assert r['trace'][-1]['iteration']==50
            assert r['trace'][-1]['best_length']==r['best_length']
    df=pd.DataFrame(rows).sort_values(['instance','method','seed'])
    assert len(df[['instance','method','seed']].drop_duplicates())==len(df)
    assert (df.groupby(['instance','method']).size()==30).all()
    assert (df.groupby(['instance','seed']).initial_best.nunique()==1).all()
    trials=df.pivot(index=['instance','seed'],columns='method',values='delta_trials')
    overshoot=(trials['ILS-work']-trials['Guided'])/trials['Guided']
    assert (overshoot>=0).all()
    summary=df.groupby(['instance','suite','dimension','reference_length','method'],sort=True).agg(
        runs=('best_length','size'),mean_length=('best_length','mean'),sd_length=('best_length','std'),
        median_length=('best_length','median'),best_length=('best_length','min'),worst_length=('best_length','max'),
        mean_excess=('excess_pct','mean'),sd_excess=('excess_pct','std'),median_excess=('excess_pct','median'),
        success_1pct=('excess_pct',lambda x:float((x<=1.).mean())),
        cpu_mean=('cpu_seconds','mean'),cpu_sd=('cpu_seconds','std'),wall_mean=('wall_seconds','mean'),
        delta_trials_mean=('delta_trials','mean'),candidate_count_mean=('candidate_count','mean'),
        initial_best_mean=('initial_best','mean'),mean_improvement=('improvement_from_start_pct','mean'),
        neff_mean=('neff','mean'),clone_neff_mean=('clone_neff','mean'),
        edge_diversity_mean=('edge_diversity','mean'),unique_tours_mean=('unique_tours','mean'),
        q05_count_mean=('q05_count','mean'),q05_edge_diversity_mean=('q05_edge_diversity','mean'),
        q10_count_mean=('q10_count','mean'),q10_edge_diversity_mean=('q10_edge_diversity','mean'),
        q20_count_mean=('q20_count','mean'),q20_edge_diversity_mean=('q20_edge_diversity','mean')).reset_index()
    wide=summary.pivot(index='instance',columns='method',values='mean_excess')
    results=[paired_stats(wide[a],wide[b],f'{a} vs {b}') for a,b in CONTRASTS]
    adjusted=holm([r['p_raw'] for r in results])
    for r,p in zip(results,adjusted):r['p_holm']=float(p)
    tests=pd.DataFrame(results)
    strata=[]
    for suite in ['TSPLIB','synthetic']:
        subset=summary[summary.suite==suite].pivot(index='instance',columns='method',values='mean_excess')
        for a,b in CONTRASTS:
            row=paired_stats(subset[a],subset[b],f'{a} vs {b}')
            row['suite']=suite
            strata.append(row)
    # Stratum tests are sensitivity analyses, corrected as one ten-test family.
    for r,p in zip(strata,holm([r['p_raw'] for r in strata])):r['p_holm']=float(p)
    factorial_values={
        'Guidance averaged across survival rules':((wide['Guided']-wide['Neutral'])+(wide['Guided-crowding']-wide['Neutral-crowding']))/2,
        'Crowding averaged across guidance settings':((wide['Guided-crowding']-wide['Guided'])+(wide['Neutral-crowding']-wide['Neutral']))/2,
        'Guidance by crowding interaction':(wide['Guided-crowding']-wide['Guided'])-(wide['Neutral-crowding']-wide['Neutral'])}
    factorial=[paired_stats(d,np.zeros(len(d)),label) for label,d in factorial_values.items()]
    for r,p in zip(factorial,holm([r['p_raw'] for r in factorial])):r['p_holm']=float(p)
    ranks=pd.DataFrame(rankdata(wide[METHODS],axis=1),index=wide.index,columns=METHODS)
    friedman=friedmanchisquare(*[wide[k] for k in METHODS])
    method_summary=summary.groupby('method').agg(mean_excess=('mean_excess','mean'),
        median_instance_excess=('mean_excess','median'),mean_cpu=('cpu_mean','mean'),
        neff=('neff_mean','mean'),clone_neff=('clone_neff_mean','mean'),
        edge_diversity=('edge_diversity_mean','mean'),unique_tours=('unique_tours_mean','mean'),
        q10_count=('q10_count_mean','mean'),q10_edge_diversity=('q10_edge_diversity_mean','mean'),
        success_1pct=('success_1pct','mean'),delta_trials=('delta_trials_mean','mean')).reset_index()
    method_summary['mean_rank']=method_summary.method.map(ranks.mean().to_dict())
    for suite in ['TSPLIB','synthetic']:
        means=summary[summary.suite==suite].groupby('method').mean_excess.mean()
        method_summary[f'{suite}_mean_excess']=method_summary.method.map(means.to_dict())
    datasets=pd.DataFrame(manifest)
    validation=dict(unique_main_runs=len(records),reference_runs=len(references),
        total_saved_tours_checked=len(all_records),instance_count=len(manifest),
        runs_per_condition=30,mean_work_overshoot_fraction=float(overshoot.mean()),
        maximum_work_overshoot_fraction=float(overshoot.max()),
        friedman_statistic=float(friedman.statistic),friedman_p=float(friedman.pvalue),
        tests='Exact two-sided Wilcoxon signed-rank for nonzero untied differences; otherwise Pratt ranks with 50000 sign permutations; Holm family of five',
        ci='Median of paired instance differences, percentile bootstrap over instances, 10000 resamples',
        secondary_replays=3360,
        timing='Original performance runs exclude secondary 5/10/20 percent diversity audit; original traces and clone diagnostics included')
    for name,data in [('per_run',df),('performance_summary',summary),('paired_tests',tests),
                       ('stratum_tests',pd.DataFrame(strata)),('factorial_tests',pd.DataFrame(factorial)),
                       ('method_summary',method_summary),('dataset_summary',datasets),('reference_runs',ref_df),
                       ('diagnostic_traces',pd.DataFrame(traces)),('instance_ranks',ranks.reset_index())]:
        write_csv(OUT/f'{name}.csv',data)
        if name in ['per_run','diagnostic_traces']:
            write_csv(OUT/f'{name}.csv.gz',data,compress=True)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    write_gzip(OUT/'run_records.jsonl.gz',sorted(records,key=lambda r:(r['instance'],r['method'],r['seed'])))
    write_gzip(OUT/'reference_records.jsonl.gz',sorted(references,key=lambda r:(r['instance'],r['seed'])))
    print(method_summary.round(4).to_string(index=False))
    print(tests.round(6).to_string(index=False))
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
