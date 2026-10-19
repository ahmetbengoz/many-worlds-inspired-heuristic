"""Exploratory v6 intervention study; v5 routines remain unchanged.

All population policies share initialization, 16 candidates per iteration and
operator settings. Quality gating limits only uphill crowding acceptance; it
is not a global constraint on the entire population. The two adaptive policies
use a fixed threshold of four effective trajectories.
"""
from __future__ import annotations
import time
import numpy as np
from numba import njit
from .component_study import (Work, initialise, diagnostics, neighbours, edge_overlaps,
    double_bridge, length, influence_weights, directed_transition_kernel,
    candidate_lists, order_crossover, guided_position_move, swap_mutation,
    insertion_mutation)

POLICIES=("indexed","crowding","quality","adaptive_class","adaptive_slot")


def clone_effective_count(population,p):
    n=population.shape[1]
    other=np.roll(population,-1,axis=1)
    signatures=np.sort(np.minimum(population,other).astype(np.int64)*n+np.maximum(population,other),axis=1)
    _,classes=np.unique(signatures,axis=0,return_inverse=True)
    masses=np.bincount(classes,weights=p)
    return float(np.exp(-np.sum(masses*np.log(masses))))


def quality_admissible(value,reference,incumbent):
    """An improving replacement is always eligible; uphill uses a 5% gate."""
    return value<=reference or 100*int(value)<=105*int(incumbent)


def run_population_extended(dist, seed, policy="indexed", iterations=50, trace_every=10):
    if policy not in POLICIES:
        raise ValueError(policy)
    guidance = True
    gated = policy in ("quality", "adaptive_class", "adaptive_slot")
    active_iterations = gate_rejections = uphill_proposals = accepted_candidates = 0
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
    for it in range(iterations):
        if it % trace_every == 0:
            traces.append(dict(iteration=it,best_length=bestval,**diagnostics(pop,vals,beta)))
        p,fhat = influence_weights(vals,beta=beta)
        class_count = clone_effective_count(pop,p)
        slot_count = float(np.exp(-np.sum(p*np.log(p))))
        crowding = policy in ("crowding", "quality") or (policy=="adaptive_class" and class_count<4.) or (policy=="adaptive_slot" and slot_count<4.)
        active_iterations += int(crowding)
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
                uphill_proposals += 1
                if crowding and gated and not quality_admissible(value,reference,bestval):
                    accept = False
                    gate_rejections += 1
            if accept:
                accepted_candidates += 1
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
    traces.append(dict(iteration=iterations,best_length=bestval,**final))
    return dict(best_length=bestval,best_tour=best.tolist(),initial_best=initial_best,
                guidance=guidance,policy=policy,iterations=iterations,**final,trace=traces,
                active_iterations=active_iterations,gate_rejections=gate_rejections,
                uphill_proposals=uphill_proposals,accepted_candidates=accepted_candidates,
                candidate_count=candidate_count,restart_candidates=restart_count,
                delta_trials=work.trials,accepted_ls_moves=work.moves,
                full_evaluations=work.full_evaluations,
                cpu_seconds=time.process_time()-start_cpu,
                wall_seconds=time.perf_counter()-start_wall)


@njit(cache=True)
def descent_with_budget(tour,dist,candidates,available):
    """Repeat the candidate-list scan to convergence or the exact trial cap."""
    t=tour.copy();n=len(t);trials=moves=passes=0
    while True:
        changed=False;pos=np.empty(n,dtype=np.int32)
        for z in range(n):pos[t[z]]=z
        for i in range(1,n-2):
            a,b=t[i-1],t[i]
            for h in range(candidates.shape[1]):
                k=pos[candidates[a,h]]
                if k<=i or k>=n-1:continue
                if trials>=available:return t,trials,moves,passes,False
                c,d=t[k],t[(k+1)%n];trials+=1
                delta=int(dist[a,c])+int(dist[b,d])-int(dist[a,b])-int(dist[c,d])
                if delta<0:
                    t[i:k+1]=t[i:k+1][::-1];moves+=1;changed=True
                    for z in range(i,k+1):pos[t[z]]=z
                    break
        passes+=1
        if not changed:return t,trials,moves,passes,True


def run_descent_ils(dist,seed,target_trials):
    start_cpu,start_wall=time.process_time(),time.perf_counter()
    rng=np.random.default_rng(seed);cand=candidate_lists(dist,12);work=Work(dist,cand)
    pop,vals=initialise(dist,rng,work);idx=int(np.argmin(vals))
    current=pop[idx].copy();value=int(vals[idx]);best=current.copy();bestval=value;initial=value
    completed=truncated=passes=candidates=restarts=stale=0
    if target_trials<work.trials:raise ValueError('Target below shared initialization work')
    trace=[dict(iteration=0,best_length=bestval,delta_trials=work.trials)]
    first=True
    while work.trials<target_trials:
        if first:t=current.copy();first=False
        else:
            t=double_bridge(current,rng)
            if rng.random()<.30:t=swap_mutation(t,rng)
            if rng.random()<.20:t=insertion_mutation(t,rng)
        t,tr,mo,pa,converged=descent_with_budget(t,dist,cand,int(target_trials-work.trials))
        work.trials+=int(tr);work.moves+=int(mo);passes+=int(pa)
        completed+=int(converged);truncated+=int(not converged)
        v=work.evaluate(t);candidates+=1;stale+=1
        if v<=value:current,value=t.copy(),v
        if v<bestval:best,bestval=t.copy(),v;stale=0
        if stale>=80:
            current=double_bridge(best,rng);value=work.evaluate(current);restarts+=1;stale=0
        if candidates%100==0:trace.append(dict(iteration=candidates,best_length=bestval,delta_trials=work.trials))
        if candidates>100000:raise RuntimeError('No progress toward work budget')
    trace.append(dict(iteration=candidates,best_length=bestval,delta_trials=work.trials))
    return dict(best_length=bestval,best_tour=best.tolist(),initial_best=initial,
        delta_trials=work.trials,accepted_ls_moves=work.moves,full_evaluations=work.full_evaluations,
        candidate_count=candidates,restart_candidates=restarts,converged_descents=completed,
        budget_truncated_descents=truncated,descent_passes=passes,trace=trace,
        cpu_seconds=time.process_time()-start_cpu,wall_seconds=time.perf_counter()-start_wall)


def warmup_extended():
    from .component_study import warmup
    warmup()
    d=np.ones((8,8),dtype=np.int32);np.fill_diagonal(d,0)
    descent_with_budget(np.arange(8,dtype=np.int32),d,candidate_lists(d,4),100)

