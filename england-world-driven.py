import math
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Global model configuration
# ============================================================

# Lattice size and template definition
N = 12                      # number of lattice sites (1D ring)
L = 4                       # length of each replication window

TEMPLATE = (1, 1, 0, 1)     # τ pattern: "replicator"
BLANK    = (0, 0, 0, 0)     # β pattern: "blank" / non-replicator

# Three disjoint replication windows (blocks A, B, C)
# Python indices are 0-based, end-exclusive
REPLICATION_WINDOWS = [
    (0, 4),   # block A: sites 0-3
    (4, 8),   # block B: sites 4-7
    (8, 12),  # block C: sites 8-11
]

# Precompute which sites belong to any replication window
WINDOW_SITES = set(i for (start, end) in REPLICATION_WINDOWS for i in range(start, end))

# ------------------------------------------------------------
# Default dynamical parameters
# ------------------------------------------------------------
# These are chosen to be qualitatively consistent with England's bound:
#   ΔS_tot >= ln(g/δ)
# We pick ln(g/δ) = ln(3) -> g/δ ≈ 3 (moderate growth advantage).
SIGMA0_DEFAULT = math.log(3.0)   # EP per replication event (nats), ~1.10

# Base replication rate (dimensionless time units)
K_R_DEFAULT = 0.3

# Noise rate for sites OUTSIDE replication windows (environment)
K_N_ENV_DEFAULT = 0.05


# ============================================================
# State utilities
# ============================================================

def count_templates(state):
    """
    Count how many replication windows currently match the TEMPLATE pattern.
    This is our discrete "replicator number" n(ω).
    """
    count = 0
    for (start, end) in REPLICATION_WINDOWS:
        if tuple(state[start:end]) == TEMPLATE:
            count += 1
    return count


def apply_block(state, window_index, pattern):
    """
    Return a new state where replication window 'window_index'
    is set to 'pattern' (either TEMPLATE or BLANK).
    """
    s = list(state)
    start, end = REPLICATION_WINDOWS[window_index]
    for i in range(L):
        s[start + i] = pattern[i]
    return tuple(s)


def block_labels(state):
    """
    Coarse-grained labels for blocks A, B, C.

    0 = BLANK  (β)
    1 = TEMPLATE (τ)
    2 = other  (should not appear here if we never apply noise inside windows)
    """
    labels = []
    for (start, end) in REPLICATION_WINDOWS:
        block = tuple(state[start:end])
        if block == BLANK:
            labels.append(0)
        elif block == TEMPLATE:
            labels.append(1)
        else:
            labels.append(2)
    return tuple(labels)


# ============================================================
# Transition definitions
# ============================================================

def replication_edges(
    state,
    k_R=K_R_DEFAULT,
    sigma0=SIGMA0_DEFAULT,
    require_existing_template=True,
):
    """
    Define replication transitions R for a given state.

    Each replication window can be in BLANK (β) or TEMPLATE (τ).
    We implement a simple two-state Markov process per window:

        β  <---->  τ

    - β -> τ ("birth" / replication event):
        forward rate  = k_R * exp(+sigma0/2)
        reverse rate  = k_R * exp(-sigma0/2)
        EP increment  = ln(rate_fwd / rate_rev) = sigma0  (for birth events)

    - τ -> β ("death" / decay event):
        forward rate  = k_R * exp(-sigma0/2)
        reverse rate  = k_R * exp(+sigma0/2)
        We do NOT count EP for these events in Σ_R, following the
        theoretical setup where R contains only replication events.

    If require_existing_template is True, β->τ events are only allowed when
    at least one TEMPLATE is already present somewhere in the system
    (autocatalytic requirement).
    """
    edges = []
    n_templates = count_templates(state)

    for w_idx, (start, end) in enumerate(REPLICATION_WINDOWS):
        block = tuple(state[start:end])

        # β -> τ (birth)
        if block == BLANK:
            # optional "must already have at least one replicator" condition
            if require_existing_template and n_templates == 0:
                continue

            new_state = apply_block(state, w_idx, TEMPLATE)
            fwd_rate = k_R * math.exp(+sigma0 / 2.0)
            rev_rate = k_R * math.exp(-sigma0 / 2.0)

            edges.append({
                "from": state,
                "to": new_state,
                "rate": fwd_rate,
                "rev_rate": rev_rate,
                "kind": "R",
                "subkind": "birth",
                "window": w_idx,
            })

        # τ -> β (death)
        elif block == TEMPLATE:
            new_state = apply_block(state, w_idx, BLANK)
            fwd_rate = k_R * math.exp(-sigma0 / 2.0)
            rev_rate = k_R * math.exp(+sigma0 / 2.0)

            edges.append({
                "from": state,
                "to": new_state,
                "rate": fwd_rate,
                "rev_rate": rev_rate,
                "kind": "R",
                "subkind": "death",
                "window": w_idx,
            })

        # If block is "other" (shouldn't happen with current noise design),
        # we do not define replication edges for it.

    return edges


def noise_edges(state, k_N_env=K_N_ENV_DEFAULT):
    """
    Background environmental noise: symmetric bit flips OUTSIDE replication windows.

    - We flip each non-window bit with rate k_N_env.
    - Bits inside replication windows are untouched by noise; they change
      only via explicit replication β↔τ transitions above.

    This provides a simple "environment" with stochastic dynamics that the
    replicator lives in, without constantly scrambling the template blocks.
    """
    edges = []
    for i in range(N):
        # Skip bits that belong to any replication window
        if i in WINDOW_SITES:
            continue

        s = list(state)
        s[i] ^= 1  # flip the bit
        new_state = tuple(s)

        edges.append({
            "from": state,
            "to": new_state,
            "rate": k_N_env,
            "rev_rate": k_N_env,   # symmetric noise: log-rate ratio = 0
            "kind": "N",
            "subkind": "flip",
            "site": i,
        })

    return edges


def get_transitions(
    state,
    k_R=K_R_DEFAULT,
    sigma0=SIGMA0_DEFAULT,
    k_N_env=K_N_ENV_DEFAULT,
    require_existing_template=True,
):
    """
    Aggregate all outgoing transitions from the current state:
      - Replication edges (β ↔ τ in each window)
      - Environmental noise edges (bit flips outside the windows)
    """
    transitions = []
    transitions.extend(
        replication_edges(
            state,
            k_R=k_R,
            sigma0=sigma0,
            require_existing_template=require_existing_template,
        )
    )
    transitions.extend(noise_edges(state, k_N_env=k_N_env))
    return transitions


# ============================================================
# Gillespie simulation
# ============================================================

def gillespie_step(
    state,
    t,
    rng,
    k_R=K_R_DEFAULT,
    sigma0=SIGMA0_DEFAULT,
    k_N_env=K_N_ENV_DEFAULT,
    require_existing_template=True,
):
    """
    Perform one Gillespie step from state at time t.

    Returns:
        new_state : updated microstate
        dt        : time increment
        info      : dict with event details and EP increment dΣ_R for this event
                    (EP is only counted for replication-birth events).
    """
    transitions = get_transitions(
        state,
        k_R=k_R,
        sigma0=sigma0,
        k_N_env=k_N_env,
        require_existing_template=require_existing_template,
    )

    rates = [tr["rate"] for tr in transitions]
    total_rate = sum(rates)

    if total_rate <= 0.0:
        # No outgoing edges => absorbing state
        return state, float("inf"), None

    # Draw waiting time from exponential distribution
    u = rng.random()
    dt = -math.log(u) / total_rate

    # Choose which event occurs
    u2 = rng.random() * total_rate
    accum = 0.0
    chosen = None
    for tr in transitions:
        accum += tr["rate"]
        if u2 <= accum:
            chosen = tr
            break

    new_state = chosen["to"]

    # EP increment from R, restricted to β->τ birth events
    ep_R = 0.0
    if chosen["kind"] == "R" and chosen.get("subkind") == "birth":
        # log(rate_fwd / rate_rev) = sigma0 by construction
        ep_R = math.log(chosen["rate"] / chosen["rev_rate"])

    return new_state, dt, {"event": chosen, "ep_R": ep_R}


def simulate(
    initial_state,
    t_max,
    k_R=K_R_DEFAULT,
    sigma0=SIGMA0_DEFAULT,
    k_N_env=K_N_ENV_DEFAULT,
    require_existing_template=True,
    seed=None,
    sample_dt=0.2,
):
    """
    Simulate the Markov process using the Gillespie algorithm.

    We record at regular sampling intervals:
      - time t
      - microstate ω
      - template count n(ω)
      - cumulative EP from replication births Σ_R
      - cumulative number of replication births |R|
    """
    rng = random.Random(seed)
    t = 0.0
    state = tuple(initial_state)

    # Time series containers
    times = [t]
    states = [state]
    n_templates_series = [count_templates(state)]
    cum_ep_R = [0.0]
    R_event_count = [0]

    ep_R_total = 0.0
    R_events = 0
    next_sample_time = sample_dt

    while t < t_max:
        new_state, dt, info = gillespie_step(
            state,
            t,
            rng,
            k_R=k_R,
            sigma0=sigma0,
            k_N_env=k_N_env,
            require_existing_template=require_existing_template,
        )

        if info is None:
            # No more events possible
            break

        t += dt
        state = new_state

        if info["event"]["kind"] == "R" and info["event"].get("subkind") == "birth":
            R_events += 1
            ep_R_total += info["ep_R"]

        # Record whenever we cross a sampling time
        while t >= next_sample_time and next_sample_time <= t_max:
            times.append(next_sample_time)
            states.append(state)
            n_templates_series.append(count_templates(state))
            cum_ep_R.append(ep_R_total)
            R_event_count.append(R_events)
            next_sample_time += sample_dt

    return {
        "times": times,
        "states": states,
        "n_templates": n_templates_series,
        "cum_ep_R": cum_ep_R,
        "R_events": R_event_count,
        "params": dict(
            N=N,
            L=L,
            k_R=k_R,
            sigma0=sigma0,
            k_N_env=k_N_env,
            t_max=t_max,
            sample_dt=sample_dt,
        ),
    }


# ============================================================
# Complexity calculations (Bar-Yam-style)
# ============================================================

def entropy_from_counts(counts):
    """
    Shannon entropy (nats) from a dict mapping outcome -> count.
    """
    total = sum(counts.values())
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log(p)
    return H


def complexity_profile_blocks(labels_samples):
    """
    Complexity profile C(k) for the 3 block labels (A,B,C).

    - Each label is 0 (BLANK), 1 (TEMPLATE), or 2 (other).
    - We treat blocks as the "atomic parts" and use Bar-Yam-style
      decomposition: compute entropies H(S) for all non-empty subsets S,
      then invert to information atoms I(S), then sum them to get C(k).
    """
    indices = [0, 1, 2]  # blocks A,B,C

    # 1) Entropies H(S) for all nonempty subsets S
    H = {}
    for r in range(1, len(indices) + 1):
        for subset in itertools.combinations(indices, r):
            S = tuple(subset)
            counts = {}
            for labels in labels_samples:
                key = tuple(labels[i] for i in S)
                counts[key] = counts.get(key, 0) + 1
            H[frozenset(S)] = entropy_from_counts(counts)

    # 2) Information atoms I(S) via Möbius inversion on the subset lattice
    I = {}
    for r in range(1, len(indices) + 1):
        for subset in itertools.combinations(indices, r):
            S = frozenset(subset)
            total = 0.0
            for k in range(1, r + 1):
                for Tt in itertools.combinations(subset, k):
                    T = frozenset(Tt)
                    coeff = (-1) ** (r - k)
                    total += coeff * H[T]
            I[S] = total

    # 3) Complexity profile C(k) = sum_{|S| >= k} I(S)
    C = {}
    for k in range(1, len(indices) + 1):
        Ck = 0.0
        for S, val in I.items():
            if len(S) >= k:
                Ck += val
        C[k] = Ck

    return C, H, I


def complexity_profile_block_sites(bit_samples):
    """
    Complexity profile for the 4 sites (bits) inside a single replication window.

    - bit_samples: list of 4-tuples (b0,b1,b2,b3) sampled over time.
    - This reveals multi-bit structure inside a template-block
      (e.g. how often we see 1101 vs 0000 vs mixed patterns, and the
       dependencies among those bits).
    """
    indices = [0, 1, 2, 3]  # the 4 sites in the window

    # 1) Entropies H(S) for nonempty subsets S
    H_bits = {}
    for r in range(1, len(indices) + 1):
        for subset in itertools.combinations(indices, r):
            S = tuple(subset)
            counts = {}
            for bits in bit_samples:
                key = tuple(bits[i] for i in S)
                counts[key] = counts.get(key, 0) + 1
            H_bits[frozenset(S)] = entropy_from_counts(counts)

    # 2) Information atoms I(S) via Möbius inversion
    I_bits = {}
    for r in range(1, len(indices) + 1):
        for subset in itertools.combinations(indices, r):
            S = frozenset(subset)
            total = 0.0
            for k in range(1, r + 1):
                for Tt in itertools.combinations(subset, k):
                    T = frozenset(Tt)
                    coeff = (-1) ** (r - k)
                    total += coeff * H_bits[T]
            I_bits[S] = total

    # 3) Complexity profile C_bits(k)
    C_bits = {}
    for k in range(1, len(indices) + 1):
        Ck = 0.0
        for S, val in I_bits.items():
            if len(S) >= k:
                Ck += val
        C_bits[k] = Ck

    return C_bits, H_bits, I_bits


# ============================================================
# Analysis + plotting
# ============================================================

def analyze_and_plot(result, sigma0):
    """
    Print summary statistics and generate basic plots:
      - n(t): template count over time
      - Σ_R(t): cumulative EP from replication births
      - EP per replication event (should ~ sigma0)
      - Complexity profiles at block and bit scales
    """
    times = np.array(result["times"])
    n_templates = np.array(result["n_templates"])
    cum_ep_R = np.array(result["cum_ep_R"])
    R_events = np.array(result["R_events"])

    with np.errstate(divide="ignore", invalid="ignore"):
        ep_per_event = np.where(R_events > 0, cum_ep_R / R_events, np.nan)

    # Coarse-grained labels for blocks A,B,C over time
    label_samples = [block_labels(s) for s in result["states"]]
    C_blocks, H_blocks, I_blocks = complexity_profile_blocks(label_samples)

    # Site-level samples for block A (window 0)
    start0, end0 = REPLICATION_WINDOWS[0]
    bit_samples = [tuple(state[start0:end0]) for state in result["states"]]
    C_bits, H_bits, I_bits = complexity_profile_block_sites(bit_samples)

    # ----- Print summary -----
    print("=== FINAL STATS ===")
    print(f"Simulated time:             {times[-1]:.3f}")
    print(f"Final template count:       {n_templates[-1]}")
    print(f"Average template count ⟨n⟩: {np.mean(n_templates):.3f}")
    print(f"Total replication births:   {R_events[-1]}")
    print(f"Total EP from births Σ_R:   {cum_ep_R[-1]:.4f}")
    if R_events[-1] > 0:
        print(f"EP per birth event:         {cum_ep_R[-1] / R_events[-1]:.4f}")
    print(f"Target sigma0:              {sigma0:.4f}")

    print("\nComplexity profile for blocks A,B,C (labels 0=β,1=τ):")
    for k in sorted(C_blocks.keys()):
        print(f"  C_blocks({k}) = {C_blocks[k]:.4f}")

    print("\nComplexity profile for 4 sites in block A:")
    for k in sorted(C_bits.keys()):
        print(f"  C_bits({k}) = {C_bits[k]:.4f}")

    # ----- Plots -----
    plt.figure(figsize=(10, 8))

    # 1) Template count over time
    plt.subplot(3, 1, 1)
    plt.plot(times, n_templates)
    plt.xlabel("Time")
    plt.ylabel("n(t)")
    plt.title("Template Count Over Time")

    # 2) Cumulative EP from replication births
    plt.subplot(3, 1, 2)
    plt.plot(times, cum_ep_R)
    plt.xlabel("Time")
    plt.ylabel("Σ_R(t)")
    plt.title("Cumulative EP from Replication Births")

    # 3) EP per replication event vs time
    plt.subplot(3, 1, 3)
    plt.plot(times, ep_per_event, marker=".", linestyle="-")
    plt.axhline(sigma0, color="black", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("EP per birth")
    plt.title("EP per Replication Event (dashed = sigma0)")

    plt.tight_layout()
    plt.show()


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    # Choose dynamical parameters for this experiment
    k_R = K_R_DEFAULT
    sigma0 = SIGMA0_DEFAULT
    k_N_env = K_N_ENV_DEFAULT

    # Total simulation time and sampling resolution
    t_max = 200.0        # run long enough to see many births/deaths
    sample_dt = 0.2

    # Initial state: one template in block A, blanks in B and C, all env bits = 0
    # (We only care about the replication windows for complexity.)
    initial_state = TEMPLATE + BLANK + BLANK

    result = simulate(
        initial_state=initial_state,
        t_max=t_max,
        k_R=k_R,
        sigma0=sigma0,
        k_N_env=k_N_env,
        require_existing_template=False,  # allow β->τ births even when n=0
        seed=42,
        sample_dt=sample_dt,
    )

    analyze_and_plot(result, sigma0)

