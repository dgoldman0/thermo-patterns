from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ============================================================
# Bar-Yam multiscale complexity utilities
# ============================================================

def entropy_from_samples(samples: List[Tuple[int, ...]], subset_indices: Tuple[int, ...]) -> float:
    counts = {}
    for s in samples:
        key = tuple(s[i] for i in subset_indices)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log(p)
    return H


def complexity_profile(samples: List[Tuple[int, ...]], n_parts: int) -> Dict[int, float]:
    """
    Bar-Yam multiscale complexity profile C(k) for a finite set of atomic parts.

    samples: list of tuples of length n_parts giving values for each part.
    """
    indices = list(range(n_parts))

    # 1) Entropies H(S) for all nonempty subsets S
    H = {}
    from itertools import combinations
    for r in range(1, n_parts + 1):
        for subset in combinations(indices, r):
            fs = frozenset(subset)
            H[fs] = entropy_from_samples(samples, subset)

    # 2) Information atoms I(S) via Möbius inversion
    I = {}
    for r in range(1, n_parts + 1):
        for subset in combinations(indices, r):
            S = frozenset(subset)
            total = 0.0
            for k in range(1, r + 1):
                for T_tuple in combinations(subset, k):
                    T = frozenset(T_tuple)
                    coeff = (-1) ** (r - k)
                    total += coeff * H[T]
            I[S] = total

    # 3) Complexity profile C(k)
    C = {}
    for k in range(1, n_parts + 1):
        Ck = 0.0
        for S, val in I.items():
            if len(S) >= k:
                Ck += val
        C[k] = Ck
    return C


# ============================================================
# Model parameters: state space + environment + driving
# ============================================================

@dataclass
class ModelParams:
    # Bitstring architecture (same as England-first model)
    n_sites: int = 16
    window_positions: Tuple[Tuple[int, int], ...] = (
        (0, 4),   # A
        (4, 8),   # B
        (8, 12),  # C
    )
    template: Tuple[int, ...] = (1, 1, 0, 1)
    blank: Tuple[int, ...] = (0, 0, 0, 0)

    # Environment: coarse label pattern over three windows (0/1)
    env_patterns: Tuple[Tuple[int, int, int], ...] = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 1),
    )
    env_switch_prob: float = 0.01  # chance per MC step to advance env pattern

    # Local Bar-Yam-inspired potential parameters
    alpha: float = 1.0            # selection strength for Δf_i
    w_match: float = 1.0          # reward when window label matches env label
    w_mismatch: float = -0.2      # mild penalty for mismatch
    w_other: float = -0.5         # penalty when window in "other" state

    # Move proposal probabilities
    p_window_move: float = 0.7    # propose window toggle vs env bit flip
    p_env_bit_flip: float = 0.3

    # Ensemble / simulation sizes
    n_systems: int = 200
    n_steps: int = 50000
    sample_every: int = 200       # record observables every this many ACCEPTED moves


# ============================================================
# State utilities
# ============================================================

def window_sites(params: ModelParams) -> set:
    return {i for (start, end) in params.window_positions for i in range(start, end)}


def init_states(params: ModelParams, rng: random.Random) -> np.ndarray:
    """
    Ensemble initial condition: A is TEMPLATE, B and C are BLANK for each system.
    """
    M = params.n_systems
    N = params.n_sites
    states = np.zeros((M, N), dtype=int)
    for m in range(M):
        for w_idx, (start, end) in enumerate(params.window_positions):
            if w_idx == 0:
                states[m, start:end] = params.template
            else:
                states[m, start:end] = params.blank
    return states


def init_environment(params: ModelParams) -> Tuple[int, Tuple[int, int, int]]:
    """
    Environment starts in the first pattern in env_patterns.
    """
    return 0, params.env_patterns[0]


def block_labels_for_state(state: np.ndarray, params: ModelParams) -> Tuple[int, int, int]:
    """
    0 = BLANK, 1 = TEMPLATE, 2 = other pattern.
    """
    labels = []
    for (start, end) in params.window_positions:
        block = tuple(int(b) for b in state[start:end])
        if block == params.blank:
            labels.append(0)
        elif block == params.template:
            labels.append(1)
        else:
            labels.append(2)
    return tuple(labels)


def count_templates(state: np.ndarray, params: ModelParams) -> int:
    c = 0
    for (start, end) in params.window_positions:
        block = tuple(int(b) for b in state[start:end])
        if block == params.template:
            c += 1
    return c


def mean_templates(states: np.ndarray, params: ModelParams) -> float:
    return float(sum(count_templates(states[m], params) for m in range(states.shape[0]))) / states.shape[0]


def block_labels_from_ensemble(states: np.ndarray, params: ModelParams) -> List[Tuple[int, int, int]]:
    return [block_labels_for_state(states[m], params) for m in range(states.shape[0])]


def bits_in_windowA_from_ensemble(states: np.ndarray, params: ModelParams) -> List[Tuple[int, ...]]:
    start, end = params.window_positions[0]
    return [tuple(int(b) for b in states[m, start:end]) for m in range(states.shape[0])]


# ============================================================
# Local potential f_i(ω_i, E_t)
# ============================================================

def local_potential(state: np.ndarray, env_labels: Tuple[int, int, int], params: ModelParams) -> float:
    """
    Local f_i for system i at state `state` under environment labels `env_labels`.

    Each of the three windows contributes:

        + w_match    when its coarse label (0/1) equals the environment label
        + w_mismatch when labels differ (excluding 'other')
        + w_other    when the window block is an 'other' pattern (label 2)

    The environment sequence env_patterns encodes multiscale structure in time:
    patterns cycle according to env_switch_prob, and systems gain f_i by tracking
    that structure with their own window labels.
    """
    f = 0.0
    sys_labels = block_labels_for_state(state, params)
    for w in range(3):
        s_lab = sys_labels[w]
        e_lab = env_labels[w]
        if s_lab == 2:
            f += params.w_other
        elif s_lab == e_lab:
            f += params.w_match
        else:
            f += params.w_mismatch
    return f


# ============================================================
# Move proposals
# ============================================================

def propose_window_toggle(state: np.ndarray, params: ModelParams, rng: random.Random) -> np.ndarray | None:
    """
    Propose toggling a single window between BLANK and TEMPLATE.
    """
    w_idx = rng.randrange(len(params.window_positions))
    start, end = params.window_positions[w_idx]
    block = tuple(int(b) for b in state[start:end])
    new_state = state.copy()
    if block == params.blank:
        new_state[start:end] = params.template
        return new_state
    elif block == params.template:
        new_state[start:end] = params.blank
        return new_state
    else:
        return None


def propose_env_bit_flip(state: np.ndarray, params: ModelParams, rng: random.Random, window_site_set: set) -> np.ndarray | None:
    """
    Propose flipping one bit outside the replication windows.
    """
    env_indices = [i for i in range(params.n_sites) if i not in window_site_set]
    if not env_indices:
        return None
    idx = rng.choice(env_indices)
    new_state = state.copy()
    new_state[idx] = 1 - new_state[idx]
    return new_state


# ============================================================
# Simulation wrapper
# ============================================================

@dataclass
class Observables:
    steps: List[int]
    env_history: List[Tuple[int, int, int]]
    mean_templates: List[float]
    total_EP_R: List[float]
    C_blocks: Dict[int, List[float]]
    C_bits: Dict[int, List[float]]


def run_local_bar_yam_sim(params: ModelParams, seed: int = 0) -> Tuple[np.ndarray, Observables]:
    rng = random.Random(seed)
    states = init_states(params, rng)
    window_site_set = window_sites(params)
    env_idx, env_labels = init_environment(params)

    total_EP_R = 0.0
    accepted_moves = 0

    # Initial complexity
    block_samples = block_labels_from_ensemble(states, params)
    Cb = complexity_profile(block_samples, n_parts=3)
    bit_samples = bits_in_windowA_from_ensemble(states, params)
    Cbit = complexity_profile(bit_samples, n_parts=4)

    steps = [0]
    env_history = [env_labels]
    mean_t = [mean_templates(states, params)]
    total_EP_R_hist = [0.0]
    C_blocks_series = {k: [v] for k, v in Cb.items()}
    C_bits_series = {k: [v] for k, v in Cbit.items()}

    for step in range(1, params.n_steps + 1):
        # External environment drive: occasionally advance env pattern
        if rng.random() < params.env_switch_prob:
            env_idx = (env_idx + 1) % len(params.env_patterns)
            env_labels = params.env_patterns[env_idx]

        # Choose system and move type
        m = rng.randrange(params.n_systems)
        state_m = states[m]
        if rng.random() < params.p_window_move:
            proposal = propose_window_toggle(state_m, params, rng)
            move_kind = "R"
        else:
            proposal = propose_env_bit_flip(state_m, params, rng, window_site_set)
            move_kind = "N"

        if proposal is None:
            continue

        f_before = local_potential(state_m, env_labels, params)
        f_after = local_potential(proposal, env_labels, params)
        dF = f_after - f_before

        # Metropolis acceptance based on Δf_i
        if dF >= 0:
            accept = True
        else:
            accept = rng.random() < math.exp(params.alpha * dF)

        if not accept:
            continue

        # Commit move
        states[m] = proposal
        accepted_moves += 1

        # EP bookkeeping: only window toggles count as "R" events
        if move_kind == "R":
            total_EP_R += params.alpha * dF

        # Record observables every sample_every accepted moves
        if accepted_moves % params.sample_every == 0:
            block_samples = block_labels_from_ensemble(states, params)
            Cb = complexity_profile(block_samples, n_parts=3)
            bit_samples = bits_in_windowA_from_ensemble(states, params)
            Cbit = complexity_profile(bit_samples, n_parts=4)

            steps.append(accepted_moves)
            env_history.append(env_labels)
            mean_t.append(mean_templates(states, params))
            total_EP_R_hist.append(total_EP_R)
            for k, v in Cb.items():
                C_blocks_series.setdefault(k, []).append(v)
            for k, v in Cbit.items():
                C_bits_series.setdefault(k, []).append(v)

    obs = Observables(
        steps=steps,
        env_history=env_history,
        mean_templates=mean_t,
        total_EP_R=total_EP_R_hist,
        C_blocks=C_blocks_series,
        C_bits=C_bits_series,
    )
    return states, obs


# Simple CLI test
if __name__ == "__main__":
    params = ModelParams()
    final_states, obs = run_local_bar_yam_sim(params, seed=42)
    print("Accepted moves:", obs.steps[-1])
    print("Final env labels:", obs.env_history[-1])
    print("Mean templates final:", obs.mean_templates[-1])
    print("Total EP_R:", obs.total_EP_R[-1])
    print("Final C_blocks:", {k: v[-1] for k, v in obs.C_blocks.items()})
    print("Final C_bits:", {k: v[-1] for k, v in obs.C_bits.items()})
