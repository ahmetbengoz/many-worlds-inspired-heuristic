"""Controlled guidance x structural-survival study (v5).

The v4 implementation is deliberately retained in mwi_h.py. This module uses
a genuine four-edge double bridge, a common initial population for every
control, complete terminal diagnostics, and counted 2-opt trials. Crowding is
a diagnostic control, not a claimed new optimiser.
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit
from .operators import (nearest_neighbor_tour, random_tour, order_crossover,
                        guided_position_move, swap_mutation, insertion_mutation,
                        candidate_lists)
from .metrics import influence_weights
from .mwi_h import directed_transition_kernel

CONFIG = dict(population_size=16, iterations=50, beta=6., gamma=4.,
              candidate_k=12, ls_passes=1, ls_moves=10, init_moves=25,
              restart_patience=20, local_samples=4, elite_count=2)


@njit(cache=True)
def bounded_two_opt(tour, dist, candidates, max_passes=1, max_moves=10):
    """Same deterministic scan as v4, additionally return trial/move counts."""
    n = len(tour)
    work = tour.copy()
    total_delta = 0
    moves = 0
    trials = 0
    if n < 5:
        return work, total_delta, trials, moves
    for _ in range(max_passes):
        improved = False
        pos = np.empty(n, dtype=np.int32)
        for z in range(n):
            pos[work[z]] = z
        for i in range(1, n - 2):
            a, b = work[i-1], work[i]
            for h in range(candidates.shape[1]):
                k = pos[candidates[a, h]]
                if k <= i or k >= n-1:
                    continue
                c, d = work[k], work[(k+1) % n]
                trials += 1
                delta = int(dist[a,c]) + int(dist[b,d]) - int(dist[a,b]) - int(dist[c,d])
                if delta < 0:
                    work[i:k+1] = work[i:k+1][::-1]
                    total_delta += delta
                    moves += 1
                    improved = True
                    for z in range(i, k+1):
                        pos[work[z]] = z
                    break
            if moves >= max_moves:
                return work, total_delta, trials, moves
        if not improved:
            break
    return work, total_delta, trials, moves


@njit(cache=True)
def length(tour, dist):
    value = 0
    for i in range(len(tour)):
        value += int(dist[tour[i], tour[(i+1) % len(tour)]])
    return value


@njit(cache=True)
def neighbours(tour):
    n = len(tour)
    a = np.empty((n, 2), dtype=np.int32)
    for i in range(n):
        a[tour[i], 0] = tour[(i-1) % n]
        a[tour[i], 1] = tour[(i+1) % n]
    return a


@njit(cache=True)
def edge_overlaps(tour, adjacency):
    result = np.zeros(len(adjacency), dtype=np.int32)
    for j in range(len(adjacency)):
        for k in range(len(tour)):
            a, b = tour[k], tour[(k+1) % len(tour)]
            if b == adjacency[j,a,0] or b == adjacency[j,a,1]:
                result[j] += 1
    return result


def double_bridge(tour, rng):
    """A B C D -> A D C B, with four nonempty cyclic segments.

    Choosing three cuts in the fixed tour array gives the four cyclic cuts
    including the closing edge. No old inter-segment edge remains.
    """
    if len(tour) < 8:
        return swap_mutation(tour, rng)
    # The offset construction guarantees four segment lengths of at least two.
    a,b,c = np.sort(rng.choice(len(tour)-5,3,replace=False))+np.array([2,3,4])
    return np.concatenate((tour[:a], tour[c:], tour[b:c], tour[a:b])).astype(np.int32)


def diagnostics(population, lengths, beta):
    p, _ = influence_weights(lengths, beta=beta)
    n = population.shape[1]
    other = np.roll(population, -1, axis=1)
    signatures = np.sort(np.minimum(population, other).astype(np.int64)*n +
                         np.maximum(population, other), axis=1)
    _, classes = np.unique(signatures, axis=0, return_inverse=True)
    masses = np.bincount(classes, weights=p)
    _, counts = np.unique(signatures, return_counts=True)
    m = len(population)
    disagreement = 1. - float(np.sum(counts*(counts-1))) / (m*(m-1)*n)
    h = -float(np.sum(p*np.log(p)))
    hc = -float(np.sum(masses*np.log(masses)))
    result = dict(neff=float(np.exp(h)), clone_neff=float(np.exp(hc)),
                  unique_tours=int(len(masses)), edge_diversity=max(0.,disagreement),
                  entropy_inflation=float(h-hc))
    # Secondary measurement audit: distinguish retained poor tours from diverse
    # tours within a stated excess of the current population's best length.
    # These bands do not affect any search or survivor decision.
    for pct in (5,10,20):
        selected = signatures[lengths <= (1+pct/100)*float(np.min(lengths))]
        count = len(selected)
        _, freq = np.unique(selected,return_counts=True)
        de = 0. if count<2 else 1.-float(np.sum(freq*(freq-1)))/(count*(count-1)*n)
        result.update({f'q{pct:02d}_count':int(count),
                       f'q{pct:02d}_unique':int(len(np.unique(selected,axis=0))),
                       f'q{pct:02d}_edge_diversity':max(0.,de)})
    return result


class Work:
    def __init__(self, dist, cand):
        self.dist, self.cand = dist, cand
        self.trials = self.moves = self.full_evaluations = 0

    def evaluate(self, t):
        self.full_evaluations += 1
        return int(length(t, self.dist))

    def improve(self, t, max_moves=10):
        t, _, trials, moves = bounded_two_opt(t, self.dist, self.cand, 1, max_moves)
        self.trials += int(trials)
        self.moves += int(moves)
        return t

    def sampled(self, tour, rng, samples):
        best = (0, None, None)
        n = len(tour)
        for _ in range(samples):
            i,k = sorted(rng.choice(np.arange(1,n),2,replace=False))
            a,b,c,d = tour[i-1],tour[i],tour[k],tour[(k+1)%n]
            delta = int(self.dist[a,c])+int(self.dist[b,d])-int(self.dist[a,b])-int(self.dist[c,d])
            self.trials += 1
            if delta < best[0]:
                best = (delta,i,k)
        t = tour.copy()
        if best[1] is not None:
            t[best[1]:best[2]+1] = t[best[1]:best[2]+1][::-1]
            self.moves += 1
        return t


def initialise(dist, rng, work):
    n = len(dist)
    pop = []
    for start in rng.choice(n, 8, replace=False):
        t = nearest_neighbor_tour(dist, rng, start=int(start), random_k=1)
        pop.append(work.improve(t,25))
    while len(pop)<12:
        t = nearest_neighbor_tour(dist, rng, random_k=4)
        pop.append(work.improve(t,12))
    while len(pop)<16:
        pop.append(work.sampled(random_tour(n,rng),rng,32))
    pop = np.asarray(pop, dtype=np.int32)
    vals = np.asarray([work.evaluate(t) for t in pop], dtype=float)
    return pop, vals


def run_population(dist, seed, guidance=True, crowding=False, trace_every=10):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    rng = np.random.default_rng(seed)
    cand = candidate_lists(dist,12)
    work = Work(dist,cand)
    pop, vals = initialise(dist,rng,work)
    initial_best = int(vals.min())
    memory, memval = pop.copy(), vals.copy()
    best = pop[int(np.argmin(vals))].copy()
    bestval = initial_best
    last_improve = 0
    beta,gamma = (6.,4.) if guidance else (0.,0.)
    traces = []
    candidate_count = restart_count = 0
    for it in range(50):
        if it % trace_every == 0:
            traces.append(dict(iteration=it,best_length=bestval,**diagnostics(pop,vals,beta)))
        p,fhat = influence_weights(vals,beta=beta)
        kernel = directed_transition_kernel(p,fhat,gamma=gamma)
        elite = np.argsort(vals)[:2]
        new, newval = pop.copy(), vals.copy()
        adjacency = np.asarray([neighbours(t) for t in new]) if crowding else None
        scale = max(1.,float(np.std(vals)))
        for i in range(16):
            j = int(rng.choice(16,p=kernel[i]))
            r = rng.random()
            if r<.42:
                t = order_crossover(pop[i],pop[j],rng)
            elif r<.70:
                t = guided_position_move(pop[i],pop[j],rng)
            elif r<.86:
                t = double_bridge(memory[i],rng)
            else:
                t = double_bridge(best if p[i]<1/16 else pop[i],rng)
            if rng.random()<.30:
                t = swap_mutation(t,rng)
            if rng.random()<.20:
                t = insertion_mutation(t,rng)
            if rng.random()<.20:
                t = work.sampled(t,rng,4)
            t = work.improve(t)
            value = work.evaluate(t)
            candidate_count += 1
            # Standard nearest-structure crowding, deterministic index tie break.
            victim = int(np.argmax(edge_overlaps(t,adjacency))) if crowding else i
            reference = newval[victim] if crowding else vals[i]
            accept = value<=reference
            if not accept:
                prob = min(.18,.5*float(p[j])+.04)*np.exp(-(value-reference)/scale)
                accept = rng.random()<prob
            if accept:
                new[victim],newval[victim] = t,value
                if crowding:
                    adjacency[victim] = neighbours(t)
            if value<memval[i]:
                memory[i],memval[i] = t.copy(),value
        if crowding:
            # Preserve an elite only if its undirected edge class is absent.
            for source in elite:
                if edge_overlaps(pop[source],adjacency).max()==len(dist):
                    continue
                victim = int(np.argmax(newval))
                if vals[source]<newval[victim]:
                    new[victim],newval[victim] = pop[source].copy(),vals[source]
                    adjacency[victim] = neighbours(new[victim])
        else:
            for source,victim in zip(elite,np.argsort(newval)[-2:]):
                if vals[source]<newval[victim]:
                    new[victim],newval[victim] = pop[source].copy(),vals[source]
        if it-last_improve>=20:
            for victim in np.argsort(newval)[-2:]:
                t = work.improve(double_bridge(best,rng))
                new[victim],newval[victim] = t,work.evaluate(t)
                restart_count += 1
            last_improve = it
        pop,vals = new,newval
        idx = int(np.argmin(vals))
        if vals[idx]<bestval:
            best,bestval = pop[idx].copy(),int(vals[idx])
            last_improve = it
    final = diagnostics(pop,vals,beta)
    traces.append(dict(iteration=50,best_length=bestval,**final))
    return dict(best_length=bestval,best_tour=best.tolist(),initial_best=initial_best,
                guidance=guidance,crowding=crowding,**final,trace=traces,
                candidate_count=candidate_count,restart_candidates=restart_count,
                delta_trials=work.trials,accepted_ls_moves=work.moves,
                full_evaluations=work.full_evaluations,
                cpu_seconds=time.process_time()-start_cpu,
                wall_seconds=time.perf_counter()-start_wall)


def run_ils_control(dist, seed, target_trials=None):
    start_cpu,start_wall = time.process_time(),time.perf_counter()
    rng = np.random.default_rng(seed)
    work = Work(dist,candidate_lists(dist,12))
    pop,vals = initialise(dist,rng,work)
    idx = int(np.argmin(vals))
    current = pop[idx].copy()
    value = int(vals[idx])
    best,bestval = current.copy(),value
    initial = bestval
    last_improve = it = restart_count = 0
    trace = [dict(iteration=0,best_length=bestval,delta_trials=work.trials)]
    while (it<800 if target_trials is None else work.trials<target_trials):
        t = double_bridge(current,rng)
        if rng.random()<.30:
            t = swap_mutation(t,rng)
        if rng.random()<.20:
            t = insertion_mutation(t,rng)
        t = work.improve(t)
        v = work.evaluate(t)
        if v<=value:
            current,value = t,v
        if v<bestval:
            best,bestval = t.copy(),v
            last_improve = it
        if it-last_improve>=80:
            current = work.improve(double_bridge(best,rng))
            value = work.evaluate(current)
            restart_count += 1
            last_improve = it
            if value<bestval:
                best,bestval = current.copy(),value
        it += 1
        if it%100==0:
            trace.append(dict(iteration=it,best_length=bestval,delta_trials=work.trials))
        if it>100000:
            raise RuntimeError('Unexpectedly small per-candidate work')
    trace.append(dict(iteration=it,best_length=bestval,delta_trials=work.trials))
    return dict(best_length=bestval,best_tour=best.tolist(),initial_best=initial,
                candidate_count=it,restart_candidates=restart_count,
                delta_trials=work.trials,accepted_ls_moves=work.moves,
                full_evaluations=work.full_evaluations,trace=trace,
                cpu_seconds=time.process_time()-start_cpu,
                wall_seconds=time.perf_counter()-start_wall)


def warmup():
    d = np.ones((8,8),dtype=np.int32)
    np.fill_diagonal(d,0)
    t = np.arange(8,dtype=np.int32)
    bounded_two_opt(t,d,candidate_lists(d,4))
    length(t,d)
    edge_overlaps(t,np.asarray([neighbours(t)]))
