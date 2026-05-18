"""
seraphim_k0_sweep.py
====================
K₀ Prefactor Universality Test

GPT's challenge: "K₀ is just one normalization of c²/ℓ_P². Show it's actually
preferred by the data, not just one of many convenient values."

This test answers that directly by:

1. Treating K₀ = A · (c²/ℓ_P²) and sweeping A over 4 decades
2. Computing n from actual posterior samples (not medians) for each A
3. Finding which A minimizes spread around ANY center (not just 5.314)
4. Asking: does the minimum land at your predicted A?
5. Running a null model (shuffled masses) to check if the minimum is physical

Your predicted A = 1 / (128π² · γ_area) = 1 / (128 · π² · 0.2375)
                 ≈ 3.381 × 10⁻⁴

If that's where the data minimum lands → K₀ is physically selected.
If the minimum is flat or drifts → GPT is right.

USAGE:
    python seraphim_k0_sweep.py --dir /path/to/hdf5/files
    python seraphim_k0_sweep.py --dir .   (scans current dir recursively)

OUTPUTS:
    seraphim_k0_sweep_results.csv    per-A sweep data
    seraphim_k0_sweep_summary.csv    key statistics
    seraphim_k0_sweep.png            plot (if matplotlib available)
"""

import os
import sys
import re
import math
import csv
import argparse
import numpy as np
from collections import defaultdict

try:
    import h5py
except ImportError:
    sys.exit("ERROR: h5py required.  pip install h5py")

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — install for correlation analysis")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found — no plot will be generated")

# ── Constants (CODATA verified) ───────────────────────────────────────────────
G        = 6.67430e-11
C_LIGHT  = 2.99792458e8
HBAR     = 1.054571817e-34
M_SUN    = 1.98847e30
ALPHA    = 7.2973525693e-3
GAMMA_A  = 0.2375             # area-counting Immirzi (Meissner 2004)

# Planck length
L_P      = math.sqrt(HBAR * G / C_LIGHT**3)   # 1.6162e-35 m
NU_P_SQ  = C_LIGHT**2 / L_P**2                # = c²/ℓ_P²  (the Planck scale)
NU_P     = math.sqrt(NU_P_SQ)                 # Planck frequency

# Your predicted prefactor A* = 1/(128π²·γ_area)
A_PREDICTED = 1.0 / (128.0 * math.pi**2 * GAMMA_A)
K0_PREDICTED = A_PREDICTED * NU_P_SQ          # should match 1.1467e84 Hz²

print("=" * 68)
print("  SERAPHIM K₀ PREFACTOR SWEEP TEST")
print("=" * 68)
print()
print("  Constants:")
print("    ℓ_P        = %.4e m" % L_P)
print("    ν_P        = %.4e Hz" % NU_P)
print("    c²/ℓ_P²    = %.4e Hz²" % NU_P_SQ)
print("    γ_area     = %.4f" % GAMMA_A)
print("    A*         = 1/(128π²·γ) = %.6e" % A_PREDICTED)
print("    K₀_pred    = %.4e Hz²" % K0_PREDICTED)
print("    log₂(A*)   = %.4f" % math.log2(A_PREDICTED))
print()

# ── Waveform priority ─────────────────────────────────────────────────────────
PREFER_WF  = ["IMRPhenomXPHM","IMRPhenomXP","IMRPhenomPv2",
               "SEOBNRv4PHM","SEOBNRv4P","NRSur7dq4",
               "IMRPhenomD","SEOBNRv4"]
NON_BBH_WF = ["NRTidal","NRTidalv2","NSBH","BNS"]

def is_non_bbh(wf_key):
    ku = (wf_key or "").upper()
    return any(x.upper() in ku for x in NON_BBH_WF)

def wf_score(k):
    ku = k.upper()
    for i, p in enumerate(PREFER_WF):
        if p.upper() in ku:
            return i + (100 if is_non_bbh(k) else 0)
    return 999 + (100 if is_non_bbh(k) else 0)

# ── HDF5 navigation (same logic as gap_structure_v3) ─────────────────────────
def get_posteriors(hf):
    keys = list(hf.keys())
    if "posterior_samples" in keys:
        ps = hf["posterior_samples"]
        if hasattr(ps, "dtype") and ps.dtype is not None:
            return ps, "posterior_samples"
    candidates = [k for k in keys if any(x in k for x in ["IMR","SEOBNR","NR","EOB"])]
    candidates.sort(key=wf_score)
    for wf in candidates:
        grp = hf[wf]
        if hasattr(grp, "keys") and "posterior_samples" in grp:
            ps = grp["posterior_samples"]
            if hasattr(ps, "dtype") and ps.dtype is not None:
                return ps, wf
        if hasattr(grp, "dtype") and grp.dtype is not None:
            return grp, wf
        if hasattr(grp, "keys"):
            for sub in grp:
                item = grp[sub]
                if hasattr(item, "dtype") and item.dtype is not None:
                    return item, wf
    return None, None

def col_arr(ps, *names):
    available = ps.dtype.names or []
    for nm in names:
        if nm in available:
            try:
                a = np.array(ps[nm][:], dtype=float)
                a = a[np.isfinite(a)]
                if len(a) > 0:
                    return a
            except Exception:
                pass
    return None

def detect_catalog(fpath):
    p = fpath.replace("\\", "/")
    if "6513631"  in p or "gwtc2p1" in p.lower() or "gwtc-2" in p.lower(): return "GWTC-2.1"
    if "8177023"  in p or "gwtc3"   in p.lower() or "gwtc-3" in p.lower(): return "GWTC-3"
    if "16053484" in p or "gwtc4"   in p.lower() or "gwtc-4" in p.lower(): return "GWTC-4"
    b = os.path.basename(fpath).upper()
    if "GWTC2" in b or "GWTC2P1" in b: return "GWTC-2.1"
    if "GWTC3" in b:                    return "GWTC-3"
    if "GWTC4" in b:                    return "GWTC-4"
    return "UNKNOWN"

def event_name(fpath):
    m = re.search(r"(GW\d{6}(?:_\d{6})?)", os.path.basename(fpath))
    return m.group(1) if m else os.path.basename(fpath)[:24]

# ── N☉ computation from posterior samples ────────────────────────────────────
def compute_N_star_samples(ps):
    """
    N☉ = E_rad / (M_total * alpha)
    Returns array of N☉ values from posterior samples, or None.
    """
    # E_rad: radiated energy in solar masses
    E_rad = col_arr(ps, "radiated_energy_non_evolved", "radiated_energy",
                    "E_rad", "energy_rad_source")
    m1    = col_arr(ps, "mass_1_source", "mass1_source", "mass_1", "m1_source")
    m2    = col_arr(ps, "mass_2_source", "mass2_source", "mass_2", "m2_source")
    chirp = col_arr(ps, "chirp_mass_source", "chirp_mass")

    if E_rad is not None and m1 is not None and m2 is not None:
        n = min(len(E_rad), len(m1), len(m2))
        M_total = m1[:n] + m2[:n]
        E = E_rad[:n]
        # Filter: physical constraints
        valid = (M_total > 0) & (E > 0) & (E < M_total)
        if valid.sum() < 10:
            return None
        N_star = E[valid] / (M_total[valid] * ALPHA)
        N_star = N_star[N_star > 0]
        return N_star

    # Fallback: use chirp mass NR estimate (Healy 2014)
    if chirp is not None and m1 is not None and m2 is not None:
        nc = min(len(chirp), len(m1), len(m2))
        M_total = m1[:nc] + m2[:nc]
        E_est = 0.0842 * chirp[:nc]
        valid = (M_total > 0) & (E_est > 0) & (E_est < M_total)
        if valid.sum() < 10:
            return None
        N_star = E_est[valid] / (M_total[valid] * ALPHA)
        N_star = N_star[N_star > 0]
        return N_star

    return None

def compute_n_from_N_star(N_star_arr, A_val):
    """
    n = log2(ν_P / sqrt(K₀ / (sqrt_j · N☉)))
      = log2(ν_P) - 0.5*log2(K₀/(sqrt_j·N☉))

    With K₀ = A · ν_P²:
      n = log2(ν_P) - 0.5*log2(A·ν_P²/(sqrt_j·N☉))
        = log2(ν_P) - 0.5*(log2(A) + log2(ν_P²) - log2(sqrt_j·N☉))
        = log2(ν_P) - 0.5*log2(A) - log2(ν_P) + 0.5*log2(sqrt_j·N☉)
        = 0.5*log2(sqrt_j · N☉) - 0.5*log2(A)
        = 0.5*(log2(sqrt_j) + log2(N☉) - log2(A))

    sqrt_j = sqrt(j(j+1)) for j=1/2 = sqrt(3)/2
    """
    sqrt_j = math.sqrt(0.75)
    log2_sqrt_j = math.log2(sqrt_j)
    log2_A = math.log2(A_val)
    n_arr = 0.5 * (log2_sqrt_j + np.log2(N_star_arr) - log2_A)
    return n_arr

# ── Load all events ───────────────────────────────────────────────────────────
def load_events(root_dir):
    """
    Walk directory, load N☉ posterior samples from each BBH event.
    Returns list of dicts: {event, catalog, N_star_samples, n_median_from_K0_pred}
    """
    h5_files = []
    for dirpath, _, fnames in os.walk(root_dir):
        for fn in fnames:
            if fn.endswith((".h5", ".hdf5")):
                h5_files.append(os.path.join(dirpath, fn))
    h5_files.sort()

    if not h5_files:
        print("  ERROR: No .h5/.hdf5 files found in %s" % root_dir)
        sys.exit(1)

    print("  Found %d HDF5 files" % len(h5_files))
    print()

    events = []
    skipped = 0
    for fpath in h5_files:
        ename   = event_name(fpath)
        catalog = detect_catalog(fpath)
        try:
            with h5py.File(fpath, "r") as hf:
                ps, wf_key = get_posteriors(hf)
                if ps is None:
                    skipped += 1
                    continue
                if is_non_bbh(wf_key or ""):
                    skipped += 1
                    continue
                N_star = compute_N_star_samples(ps)
                if N_star is None or len(N_star) < 10:
                    skipped += 1
                    continue
                # Compute n at predicted K₀ for reference
                n_at_pred = compute_n_from_N_star(N_star, A_PREDICTED)
                median_n  = float(np.median(n_at_pred))
                events.append({
                    "event"    : ename,
                    "catalog"  : catalog,
                    "wf_key"   : wf_key or "",
                    "N_star"   : N_star,
                    "n_pred_K0": median_n,
                    "n_samples": len(N_star),
                })
        except Exception as ex:
            print("  SKIP %s : %s" % (ename, ex))
            skipped += 1

    print("  Loaded   : %d events" % len(events))
    print("  Skipped  : %d events" % skipped)
    print()
    return events

# ── Sweep ────────────────────────────────────────────────────────────────────
def run_sweep(events, n_points=600):
    """
    For each A in log-space sweep:
    1. Compute n for every posterior sample of every event
    2. Pool all samples
    3. Record: std (spread), mean (center), and |mean - 5.314|

    Returns arrays: A_vals, sigmas, means, delta_from_5314
    """
    # Sweep range: 4 decades centered on A_PREDICTED
    # A* ≈ 3.4e-4, so sweep 1e-6 to 1e-2
    log_A_min = math.log10(A_PREDICTED) - 2.0
    log_A_max = math.log10(A_PREDICTED) + 2.0
    A_vals    = np.logspace(log_A_min, log_A_max, n_points)

    # Pool N_star samples from all events
    all_N_star = np.concatenate([ev["N_star"] for ev in events])
    print("  Total posterior samples in pool: %d" % len(all_N_star))
    print("  Sweeping %d A values from %.2e to %.2e ..." % (
          n_points, A_vals[0], A_vals[-1]))
    print()

    sigmas       = np.zeros(n_points)
    means        = np.zeros(n_points)
    delta_5314   = np.zeros(n_points)

    for i, A in enumerate(A_vals):
        n_all = compute_n_from_N_star(all_N_star, A)
        sigmas[i]     = float(np.std(n_all))
        means[i]      = float(np.mean(n_all))
        delta_5314[i] = abs(float(np.mean(n_all)) - 5.314)

    return A_vals, sigmas, means, delta_5314

# ── Per-event sigma at each A (for subset stability check) ───────────────────
def sweep_per_event_sigma(events, A_vals):
    """
    For each A, compute the std of per-event MEDIAN n values.
    (Different from pooled std — this measures catalog-level clustering,
    not within-event spread.)
    """
    per_event_sigmas = np.zeros(len(A_vals))
    for i, A in enumerate(A_vals):
        medians = np.array([
            float(np.median(compute_n_from_N_star(ev["N_star"], A)))
            for ev in events
        ])
        per_event_sigmas[i] = float(np.std(medians))
    return per_event_sigmas

# ── Null model: shuffle N_star across events ─────────────────────────────────
def run_null_sweep(events, A_vals, n_trials=20):
    """
    Shuffle N_star samples randomly across events, re-run sweep.
    If null produces same minimum → signal is not physical.
    Returns mean and std of null sigmas across trials.
    """
    all_N_star = np.concatenate([ev["N_star"] for ev in events])
    null_sigmas = np.zeros((n_trials, len(A_vals)))

    print("  Running null model (%d trials, shuffling N☉ samples) ..." % n_trials)
    rng = np.random.default_rng(42)
    for trial in range(n_trials):
        shuffled = rng.permutation(all_N_star)
        for i, A in enumerate(A_vals):
            n_all = compute_n_from_N_star(shuffled, A)
            null_sigmas[trial, i] = float(np.std(n_all))

    null_mean = null_sigmas.mean(axis=0)
    null_std  = null_sigmas.std(axis=0)
    return null_mean, null_std

# ── Subset stability: does minimum shift by catalog or mass bin? ──────────────
def subset_stability(events, A_vals):
    """
    Find A_min separately for each catalog and mass bin.
    If A_min is stable → universal constant.
    If A_min drifts → selection effect.
    """
    results = {}
    catalogs = list(set(ev["catalog"] for ev in events))

    for cat in sorted(catalogs):
        sub = [ev for ev in events if ev["catalog"] == cat]
        if len(sub) < 5:
            continue
        all_N = np.concatenate([ev["N_star"] for ev in sub])
        sigs  = np.array([np.std(compute_n_from_N_star(all_N, A)) for A in A_vals])
        best_idx = int(np.argmin(sigs))
        results[cat] = {
            "A_min"  : float(A_vals[best_idx]),
            "sig_min": float(sigs[best_idx]),
            "n_events": len(sub),
        }

    # Mass bins: low / mid / high total mass
    n_pred_all = np.array([ev["n_pred_K0"] for ev in events])
    low_cut  = np.percentile(n_pred_all, 33)
    high_cut = np.percentile(n_pred_all, 67)
    bins = {
        "n_low (high mass)": [ev for ev in events if ev["n_pred_K0"] < low_cut],
        "n_mid"            : [ev for ev in events if low_cut <= ev["n_pred_K0"] <= high_cut],
        "n_high (low mass)": [ev for ev in events if ev["n_pred_K0"] > high_cut],
    }
    for bname, sub in bins.items():
        if len(sub) < 5:
            continue
        all_N = np.concatenate([ev["N_star"] for ev in sub])
        sigs  = np.array([np.std(compute_n_from_N_star(all_N, A)) for A in A_vals])
        best_idx = int(np.argmin(sigs))
        results[bname] = {
            "A_min"  : float(A_vals[best_idx]),
            "sig_min": float(sigs[best_idx]),
            "n_events": len(sub),
        }

    return results

# ── Plotting ──────────────────────────────────────────────────────────────────
def make_plot(A_vals, sigmas, per_event_sigmas, null_mean, null_std,
              best_A, outpath):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor("#0d0f14")
    for ax in axes:
        ax.set_facecolor("#0d0f14")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#cccccc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    log_A = np.log10(A_vals)
    log_A_pred = math.log10(A_PREDICTED)
    log_A_best = math.log10(best_A)

    # Panel 1: pooled sigma sweep
    ax = axes[0]
    ax.plot(log_A, sigmas, color="#00e5ff", lw=2, label="Pooled σ(n)")
    if null_mean is not None:
        ax.fill_between(log_A, null_mean - null_std, null_mean + null_std,
                        color="#7c4dff", alpha=0.3, label="Null model ±1σ")
        ax.plot(log_A, null_mean, color="#7c4dff", lw=1, ls="--",
                label="Null mean")
    ax.axvline(log_A_pred, color="#39ff14", lw=2, ls="--",
               label="A* predicted (K₀ = 1.1467×10⁸⁴)")
    ax.axvline(log_A_best, color="#ff4444", lw=1.5, ls=":",
               label="A* observed min")
    ax.set_xlabel("log₁₀(A)  [where K₀ = A · c²/ℓ_P²]")
    ax.set_ylabel("σ(n) octaves")
    ax.set_title("K₀ Prefactor Sweep — Pooled σ vs A")
    ax.legend(fontsize=8, facecolor="#1a1d24", labelcolor="#cccccc")

    # Panel 2: per-event sigma sweep
    ax = axes[1]
    ax.plot(log_A, per_event_sigmas, color="#ffd700", lw=2,
            label="Per-event median σ")
    ax.axvline(log_A_pred, color="#39ff14", lw=2, ls="--",
               label="A* predicted")
    ax.axvline(log_A_best, color="#ff4444", lw=1.5, ls=":",
               label="A* observed min")
    ax.set_xlabel("log₁₀(A)")
    ax.set_ylabel("σ of per-event medians (octaves)")
    ax.set_title("Per-Event Median Clustering vs A")
    ax.legend(fontsize=8, facecolor="#1a1d24", labelcolor="#cccccc")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("  Plot saved : %s" % outpath)

# ── Main ──────────────────────────────────────────────────────────────────────
def run(root_dir):
    # 1. Load events
    events = load_events(root_dir)
    if len(events) < 10:
        print("ERROR: Too few events loaded (%d). Check HDF5 directory." % len(events))
        sys.exit(1)

    # 2. Sweep
    A_vals, sigmas, means, delta_5314 = run_sweep(events, n_points=600)

    # 3. Per-event sigma
    print("  Computing per-event sigma sweep ...")
    per_event_sigmas = sweep_per_event_sigma(events, A_vals)

    # 4. Find best A from pooled sigma
    best_idx   = int(np.argmin(sigmas))
    best_A     = float(A_vals[best_idx])
    best_sigma = float(sigmas[best_idx])

    # Sigma at predicted A
    pred_idx      = int(np.argmin(np.abs(A_vals - A_PREDICTED)))
    sigma_at_pred = float(sigmas[pred_idx])
    mean_at_pred  = float(means[pred_idx])

    # How sharp is the minimum? Compare sigma at best_A vs sigma at 0.5 dex away
    half_dex_idx = int(np.argmin(np.abs(np.log10(A_vals) - (math.log10(best_A) + 0.5))))
    sigma_half_dex = float(sigmas[half_dex_idx]) if half_dex_idx < len(sigmas) else None

    # 5. Null model
    null_mean, null_std = run_null_sweep(events, A_vals, n_trials=20)

    # Is real min below null floor?
    null_floor = float(null_mean[pred_idx] - null_std[pred_idx])
    real_below_null = best_sigma < null_floor

    # 6. Subset stability
    print("  Checking subset stability ...")
    stability = subset_stability(events, A_vals)

    # 7. Decision
    # Offset between observed best A and predicted A in dex
    dex_offset = abs(math.log10(best_A) - math.log10(A_PREDICTED))

    # Sharpness: how much does sigma rise 0.5 dex from minimum?
    sharpness = ((sigma_half_dex - best_sigma) / best_sigma * 100.0
                 if sigma_half_dex is not None else None)

    # 8. Output results CSV
    sweep_rows = []
    for i, A in enumerate(A_vals):
        sweep_rows.append({
            "log10_A"        : round(math.log10(A), 6),
            "A"              : "%.6e" % A,
            "sigma_pooled"   : round(float(sigmas[i]), 6),
            "mean_n"         : round(float(means[i]), 6),
            "per_event_sigma": round(float(per_event_sigmas[i]), 6),
            "null_mean"      : round(float(null_mean[i]), 6),
            "null_std"       : round(float(null_std[i]), 6),
            "delta_from_5314": round(float(delta_5314[i]), 6),
        })

    out_sweep = "seraphim_k0_sweep_results.csv"
    with open(out_sweep, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    # Summary CSV
    stability_rows = [
        {
            "subset"  : k,
            "A_min"   : "%.4e" % v["A_min"],
            "log10_A_min": round(math.log10(v["A_min"]), 4),
            "sig_min" : round(v["sig_min"], 6),
            "n_events": v["n_events"],
            "dex_from_pred": round(abs(math.log10(v["A_min"]) - math.log10(A_PREDICTED)), 4),
        }
        for k, v in stability.items()
    ]
    out_sum = "seraphim_k0_sweep_summary.csv"
    with open(out_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stability_rows[0].keys()))
        w.writeheader()
        w.writerows(stability_rows)

    # 9. Plot
    if HAS_MPL:
        make_plot(A_vals, sigmas, per_event_sigmas, null_mean, null_std,
                  best_A, "seraphim_k0_sweep.png")

    # 10. Print verdict
    sep = "=" * 68
    print()
    print(sep)
    print("  RESULTS")
    print(sep)
    print()
    print("  Events loaded          : %d" % len(events))
    print("  Total posterior samples: %d" % sum(len(ev["N_star"]) for ev in events))
    print()
    print("  SWEEP MINIMUM:")
    print("    Best A observed      : %.4e  (log₁₀ = %.4f)" % (
          best_A, math.log10(best_A)))
    print("    Predicted A*         : %.4e  (log₁₀ = %.4f)" % (
          A_PREDICTED, math.log10(A_PREDICTED)))
    print("    Offset               : %.4f dex" % dex_offset)
    print("    σ at observed min    : %.6f octaves" % best_sigma)
    print("    σ at predicted A*    : %.6f octaves" % sigma_at_pred)
    print("    n̄ at predicted A*    : %.6f (predicted: 5.314)" % mean_at_pred)
    if sharpness is not None:
        print("    Sharpness (Δσ/σ at ±0.5 dex): +%.1f%%" % sharpness)
    print()
    print("  NULL MODEL COMPARISON:")
    print("    Null σ at predicted A : %.6f ± %.6f" % (
          float(null_mean[pred_idx]), float(null_std[pred_idx])))
    print("    Real σ at predicted A : %.6f" % sigma_at_pred)
    print("    Real below null floor : %s" % ("YES" if real_below_null else "NO"))
    print()
    print("  SUBSET STABILITY (A_min per subset):")
    for k, v in stability.items():
        print("    %-28s : %.4e  (%.3f dex from A*)" % (
              k, v["A_min"], abs(math.log10(v["A_min"]) - math.log10(A_PREDICTED))))
    print()

    # Decision logic
    print(sep)
    print("  VERDICT")
    print(sep)
    print()

    conditions = {
        "Min within 0.3 dex of A*"          : dex_offset < 0.3,
        "Min within 0.1 dex of A* (sharp)"  : dex_offset < 0.1,
        "σ(real) < null floor"               : real_below_null,
        "Sharpness > 5% per 0.5 dex"         : (sharpness is not None and sharpness > 5.0),
        "n̄ at A* within 0.1 oct of 5.314"   : abs(mean_at_pred - 5.314) < 0.1,
    }

    passed = sum(1 for v in conditions.values() if v)
    for label, result in conditions.items():
        print("    [%s] %s" % ("✓" if result else "✗", label))

    print()
    if passed >= 4:
        print("  *** STRONG EVIDENCE: K₀ prefactor is physically selected")
        print("      by the data. GPT's flat-minimum hypothesis is rejected.")
    elif passed >= 3:
        print("  **  MODERATE EVIDENCE: K₀ is preferred but minimum is")
        print("      not sharp enough to fully reject flat-minimum hypothesis.")
        print("      More events (O5) would resolve this.")
    elif passed >= 2:
        print("  *   WEAK EVIDENCE: Minimum near A* but not distinct")
        print("      from null. The framework's K₀ is not falsified,")
        print("      but also not independently confirmed by this test alone.")
    else:
        print("  ✗   NOT CONFIRMED: Minimum does not land near predicted A*,")
        print("      or null model reproduces the same structure.")
        print("      GPT's critique may have merit. Revisit K₀ derivation.")

    print()
    print("  Outputs:")
    print("    %s" % out_sweep)
    print("    %s" % out_sum)
    if HAS_MPL:
        print("    seraphim_k0_sweep.png")
    print()
    print("  Done.")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Seraphim K₀ prefactor sweep test — posterior samples")
    ap.add_argument(
        "--dir", "-d", default=".",
        help="Root directory containing HDF5 posterior files "
             "(default: current directory, searches recursively)")
    args = ap.parse_args()
    run(args.dir)
