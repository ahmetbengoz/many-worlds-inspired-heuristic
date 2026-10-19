"""Independently verify v6 tours and report fixed exploratory contrasts."""
from pathlib import Path
import sys,json,gzip,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.tsplib_extended import parse_extended
from experiments.analyse_component_study import paired_stats,holm,write_gzip

OUT=ROOT/'results/v6'
METHODS=['Guided','Crowding','Gated','Adaptive-class','Adaptive-slot','ILS-descent']
CONTRASTS=[('Gated','Crowding'),('Adaptive-class','Gated'),
 ('Adaptive-class','Adaptive-slot'),('Adaptive-class','Guided'),('Guided','ILS-descent')]


def records_from(raw_dir,gzip_name,blocks):
    paths=sorted(raw_dir.glob('*.json'))
    if paths:
        if blocks:return [r for p in paths for r in json.loads(p.read_text()).values()]
        return [r for p in paths for r in json.loads(p.read_text())]
    with gzip.open(OUT/gzip_name,'rt') as f:return [json.loads(x) for x in f]


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name in ['src/adaptive_study.py','src/tsplib_extended.py']:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==protocol['source_sha256'][name],name
    snapshot=ROOT/protocol['runner_snapshot_at_execution']
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==protocol['source_sha256']['experiments/run_adaptive_study.py']
    manifest=json.loads((OUT/'dataset_manifest.json').read_text());lookup={x['instance']:x for x in manifest}
    records=records_from(OUT/'raw_runs','run_records.jsonl.gz',True)
    refs=records_from(OUT/'reference_runs','reference_records.jsonl.gz',False)
    assert len(records)==15120,len(records)
    assert len(refs)==280,len(refs)
    matrices={}
    for x in manifest:
        p=ROOT/x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
        matrices[x['instance']]=parse_extended(p).dist
    for r in records+refs:
        d=matrices[r['instance']];t=np.asarray(r['best_tour'],dtype=np.int64)
        assert np.array_equal(np.sort(t),np.arange(len(d)))
        assert int(d[t,np.roll(t,-1)].sum(dtype=np.int64))==r['best_length']
        optimum=lookup[r['instance']]['optimum']
        if optimum:assert r['best_length']>=optimum,(r['instance'],r['best_length'],optimum)
    ref_df=pd.DataFrame([{k:v for k,v in r.items() if k not in ('best_tour','parameters')} for r in refs])
    for x in manifest:
        rr=ref_df[ref_df.instance==x['instance']]
        x['reference_length']=x['optimum'] or int(rr.best_length.min())
        x['reference_kind']='published optimum' if x['optimum'] else 'best feasible LKH reference'
        x['stratum']=('original TSPLIB' if x['original'] else 'added TSPLIB') if x['suite']=='TSPLIB' else 'generated'
    rows=[];traces=[]
    for r in records:
        x=lookup[r['instance']];ref=x['reference_length']
        row={k:v for k,v in r.items() if k not in ('best_tour','trace','guidance','policy')}
        row.update(suite=x['suite'],stratum=x['stratum'],original=x['original'],
            edge_weight_type=x['edge_weight_type'],reference_length=ref,
            reference_kind=x['reference_kind'],excess_pct=100.*(r['best_length']-ref)/ref,
            improvement_from_start_pct=100.*(r['initial_best']-r['best_length'])/r['initial_best'])
        if r['method']!='ILS-descent':
            assert r['candidate_count']==16*r['horizon']
            assert 0<=r['active_iterations']<=r['horizon']
            assert 0<=r['gate_rejections']<=r['uphill_proposals']
            row['active_fraction']=r['active_iterations']/r['horizon']
            row['gate_fraction']=r['gate_rejections']/max(1,r['uphill_proposals'])
            assert 1.-1e-8<=r['clone_neff']<=r['neff']+1e-8<=16.+1e-7
            assert 0<=r['edge_diversity']<=1
            assert r['trace'][-1]['iteration']==r['horizon']
        assert r['best_length']<=r['initial_best']
        assert r['trace'][-1]['best_length']==r['best_length']
        assert all(a['best_length']>=b['best_length'] for a,b in zip(r['trace'],r['trace'][1:]))
        rows.append(row)
        traces.extend(dict(instance=r['instance'],seed=r['seed'],method=r['method'],horizon=r['horizon'],**t) for t in r['trace'])
    df=pd.DataFrame(rows).sort_values(['horizon','instance','method','seed'])
    assert len(df[['instance','horizon','method','seed']].drop_duplicates())==15120
    assert (df.groupby(['instance','horizon','method']).size()==30).all()
    assert (df.groupby(['instance','seed']).initial_best.nunique()==1).all()
    for h,n in [(50,28),(250,56)]:assert df[df.horizon==h].instance.nunique()==n
    # The population paths are nested: extending T must not change their first
    # 50 iterations. Check this before making any horizon comparison.
    short={(r['instance'],r['seed'],r['method']):r for r in records if r['horizon']==50 and r['method']!='ILS-descent'}
    for r in records:
        key=(r['instance'],r['seed'],r['method'])
        if r['horizon']==250 and key in short:
            at50=next(t for t in r['trace'] if t['iteration']==50)
            assert at50['best_length']==short[key]['best_length']
            for field in ['neff','clone_neff','edge_diversity']:
                assert np.isclose(at50[field],short[key][field],rtol=0,atol=1e-12)
    budgets=df.pivot(index=['instance','horizon','seed'],columns='method',values='delta_trials')
    assert (budgets['Guided']==budgets['ILS-descent']).all()
    agg=dict(runs=('best_length','size'),mean_length=('best_length','mean'),sd_length=('best_length','std'),
        mean_excess=('excess_pct','mean'),sd_excess=('excess_pct','std'),median_excess=('excess_pct','median'),
        cpu_mean=('cpu_seconds','mean'),wall_mean=('wall_seconds','mean'),delta_trials_mean=('delta_trials','mean'),
        candidate_count_mean=('candidate_count','mean'),restart_candidates_mean=('restart_candidates','mean'),
        initial_best_mean=('initial_best','mean'),neff_mean=('neff','mean'),clone_neff_mean=('clone_neff','mean'),
        edge_diversity_mean=('edge_diversity','mean'),unique_tours_mean=('unique_tours','mean'),
        q10_count_mean=('q10_count','mean'),q10_edge_diversity_mean=('q10_edge_diversity','mean'),
        active_fraction_mean=('active_fraction','mean'),gate_fraction_mean=('gate_fraction','mean'),
        gate_rejections_mean=('gate_rejections','mean'),accepted_candidates_mean=('accepted_candidates','mean'),
        converged_descents_mean=('converged_descents','mean'),budget_truncated_descents_mean=('budget_truncated_descents','mean'))
    summary=df.groupby(['horizon','instance','suite','stratum','original','edge_weight_type','dimension','reference_length','method']).agg(**agg).reset_index()
    tests=[]
    for h in [50,250]:
        wide=summary[summary.horizon==h].pivot(index='instance',columns='method',values='mean_excess')
        for a,b in CONTRASTS:
            row=paired_stats(wide[a],wide[b],a+' vs '+b);row['horizon']=h;tests.append(row)
    for r,p in zip(tests,holm([x['p_raw'] for x in tests])):r['p_holm']=float(p)
    strata=[]
    for group in ['original TSPLIB','added TSPLIB','generated']:
        wide=summary[(summary.horizon==250)&(summary.stratum==group)].pivot(index='instance',columns='method',values='mean_excess')
        for a,b in CONTRASTS:
            row=paired_stats(wide[a],wide[b],a+' vs '+b);row['stratum']=group;strata.append(row)
    for r,p in zip(strata,holm([x['p_raw'] for x in strata])):r['p_holm']=float(p)
    cols=['mean_excess','cpu_mean','delta_trials_mean','candidate_count_mean','restart_candidates_mean',
        'neff_mean','clone_neff_mean','edge_diversity_mean','unique_tours_mean','q10_count_mean',
        'q10_edge_diversity_mean','active_fraction_mean','gate_fraction_mean','gate_rejections_mean',
        'converged_descents_mean','budget_truncated_descents_mean']
    methods=summary.groupby(['horizon','method'])[cols].mean().reset_index()
    medians=summary.groupby(['horizon','method']).mean_excess.median()
    methods['median_instance_excess']=[float(medians.loc[h,m]) for h,m in zip(methods.horizon,methods.method)]
    methods['n_instances']=methods.horizon.map({50:28,250:56})
    matched=summary[summary.original].groupby(['horizon','method'])[cols].mean().reset_index()
    suites=summary[summary.horizon==250].groupby(['stratum','method'])[cols].mean().reset_index()
    traces_df=pd.DataFrame(traces)
    trace_keys=['horizon','instance','method','iteration']
    # ILS may repeat its final snapshot when a candidate-count checkpoint is
    # also terminal. Keep one observation per seed/state for readable means.
    snapshots=traces_df.drop_duplicates(trace_keys+['seed'],keep='last')
    trace_metrics=[c for c in snapshots.select_dtypes(include='number').columns
                   if c not in trace_keys+['seed']]
    trace_means=snapshots.groupby(trace_keys).agg(
        runs_at_snapshot=('seed','nunique'),
        **{c:(c,'mean') for c in trace_metrics}).reset_index()
    guided=df[df.method=='Guided'].copy()
    guided['class_only_criterion']=(guided.clone_neff<4)&(guided.neff>=4)
    guided['slot_criterion']=guided.neff<4
    guided['class_criterion']=guided.clone_neff<4
    trigger_audit=guided.groupby(['horizon','instance'])[['class_only_criterion','slot_criterion','class_criterion']].mean().reset_index()
    validation=dict(unique_main_runs=15120,reference_tours_checked=280,unique_inputs=56,
        total_tours_checked_in_extension=15400,cumulative_unique_main_runs=20160,
        cumulative_unique_reference_runs=280,cumulative_unique_tours_checked=20440,
        new_search_reference_runs=140,reused_reference_runs=140,paired_runs_per_condition=30,
        horizons={'50':28,'250':56},strong_ils_exact_budget=True,
        common_initial_best_verified=True,nested_population_horizons_verified=True,
        all_saved_tours_independently_recomputed=True,search_module_hashes_verified=True,
        primary_holm_family_size=10,secondary_stratum_family_size=15,
        note='The two horizons share initialization but are analysed in separate paired instance blocks. v5 records and the extension remain distinct experiments.')
    for name,data in [('per_run',df),('performance_summary',summary),('method_summary',methods),
       ('matched_horizon_summary',matched),('stratum_summary',suites),('paired_tests',pd.DataFrame(tests)),
       ('stratum_tests',pd.DataFrame(strata)),('dataset_summary',pd.DataFrame(manifest)),
       ('reference_runs',ref_df),('diagnostic_traces',traces_df),('trace_means',trace_means),
       ('trigger_audit',trigger_audit)]:
        data.to_csv(OUT/(name+'.csv'),index=False)
        if name in ['per_run','diagnostic_traces']:data.to_csv(OUT/(name+'.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
    write_gzip(OUT/'run_records.jsonl.gz',records)
    write_gzip(OUT/'reference_records.jsonl.gz',refs)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    print(methods.to_string(index=False));print(pd.DataFrame(tests).to_string(index=False));print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
