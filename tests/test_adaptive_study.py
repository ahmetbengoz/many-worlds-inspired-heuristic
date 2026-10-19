import numpy as np
from src.component_study import run_population, candidate_lists, length
from src.adaptive_study import (run_population_extended,quality_admissible,
    descent_with_budget,run_descent_ils)


def matrix():
    xy=np.random.default_rng(319).uniform(0,1000,(24,2))
    return np.floor(np.linalg.norm(xy[:,None]-xy[None,:],axis=2)+.5).astype(np.int32)


def test_retained_policies_reproduce_v5_tours():
    d=matrix()
    for crowd,policy in [(False,'indexed'),(True,'crowding')]:
        old=run_population(d,239,True,crowd)
        new=run_population_extended(d,239,policy)
        for key in ['best_tour','best_length','initial_best','delta_trials']:
            assert old[key]==new[key]


def test_quality_gate_keeps_improvements_and_restricts_uphill():
    assert quality_admissible(180,200,100)
    assert quality_admissible(105,104,100)
    assert not quality_admissible(106,104,100)


def test_descent_budget_and_convergence():
    d=matrix();t=np.random.default_rng(3).permutation(len(d)).astype(np.int32)
    c=candidate_lists(d,12)
    bounded,tr,m,pa,done=descent_with_budget(t,d,c,7)
    assert tr==7 and not done
    assert sorted(bounded)==list(range(len(d)))
    assert length(bounded,d)<=length(t,d)
    optimum,tr,m,pa,done=descent_with_budget(t,d,c,100000)
    assert done
    repeated,tr2,m2,_,done2=descent_with_budget(optimum,d,c,100000)
    assert done2 and m2==0 and np.array_equal(optimum,repeated)


def test_stronger_ils_exact_work_and_common_start():
    d=matrix();g=run_population_extended(d,112,'indexed')
    r=run_descent_ils(d,112,g['delta_trials'])
    assert r['delta_trials']==g['delta_trials']
    assert r['initial_best']==g['initial_best']
    assert r['best_length']==length(np.asarray(r['best_tour']),d)
