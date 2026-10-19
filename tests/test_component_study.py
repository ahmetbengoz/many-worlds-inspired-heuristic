import numpy as np
from src.component_study import (bounded_two_opt, double_bridge, diagnostics,
                                 run_population, run_ils_control, warmup)
from src.operators import candidate_lists, two_opt_candidate_descent, tour_length


def edges(t):
    return {tuple(sorted((int(a),int(b)))) for a,b in zip(t,np.roll(t,-1))}


def test_four_edge_double_bridge_and_permutation():
    for n in (8,16,52,100):
        t = np.arange(n,dtype=np.int32)
        for seed in range(20):
            new = double_bridge(t,np.random.default_rng(seed))
            assert sorted(new)==list(range(n))
            assert len(edges(t)-edges(new))==4
            assert len(edges(new)-edges(t))==4


def test_compiled_descent_preserves_legacy_deterministic_scan():
    rng = np.random.default_rng(99)
    for n in (8,30,100):
        xy = rng.normal(size=(n,2))*1000
        dist = np.floor(np.linalg.norm(xy[:,None]-xy[None,:],axis=2)+.5).astype(np.int32)
        cand = candidate_lists(dist,12)
        for _ in range(5):
            t = rng.permutation(n).astype(np.int32)
            old, delta = two_opt_candidate_descent(t,dist,cand,1,10)
            new, d, trials, moves = bounded_two_opt(t,dist,cand,1,10)
            assert np.array_equal(old,new)
            assert d==delta==tour_length(new,dist)-tour_length(t,dist)
            assert trials>=moves and moves<=10


def test_clone_aggregation_ignores_rotation_and_reversal():
    t = np.arange(8,dtype=np.int32)
    pop = np.asarray([t,np.roll(t,3),t[::-1],np.roll(t[::-1],2)])
    m = diagnostics(pop,np.ones(4),6.)
    assert m['unique_tours']==1
    assert np.isclose(m['clone_neff'],1.)
    assert np.isclose(m['neff'],4.)
    assert np.isclose(m['edge_diversity'],0.)
    assert np.isclose(m['entropy_inflation'],np.log(4.))


def test_controls_share_initial_state_and_return_actual_best_tour():
    warmup()
    rng = np.random.default_rng(8)
    xy = rng.normal(size=(30,2))*1000
    d = np.floor(np.linalg.norm(xy[:,None]-xy[None,:],axis=2)+.5).astype(np.int32)
    out = [run_population(d,14,g,c) for g,c in [(1,0),(0,0),(1,1),(0,1)]]
    out += [run_ils_control(d,14),run_ils_control(d,14,target_trials=out[0]['delta_trials'])]
    assert len({r['initial_best'] for r in out})==1
    for r in out:
        assert r['best_length']==tour_length(np.asarray(r['best_tour']),d)
        assert r['best_length']<=r['initial_best']
        assert all(a['best_length']>=b['best_length'] for a,b in zip(r['trace'],r['trace'][1:]))
    for r in out[:4]:
        assert r['trace'][-1]['iteration']==50
        assert r['clone_neff']<=r['neff']+1e-10
    assert out[-1]['delta_trials']>=out[0]['delta_trials']
