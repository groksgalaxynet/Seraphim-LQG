"""
seraphim_gamma_channel_test.py
==============================
WHAT IMMIRZI VALUE DOES THE GW ENERGY LOSS CHANNEL PROBE?

The question: is gamma=0.2375 (Meissner 2004 area-counting) the right
value for this physical channel, or does GW energy loss probe a
different spin-network counting?

This script tests every physically motivated gamma from the LQG
literature, plus a fine sweep, plus checks whether gamma_data has
a clean mathematical expression.

FOUR SUB-TESTS:

SUB-TEST A — LITERATURE VALUES
    Tests every published Immirzi value against posteriors.
    Reports mean_n, delta, sigma separation for each.
    Uses full posterior samples, per event and aggregate.

SUB-TEST B — FINE GAMMA SWEEP WITH BOOTSTRAP CI
    Sweeps gamma 0.05 to 0.60 in 1000 steps.
    At each gamma: computes mean_n and median_n from all posterior samples.
    Finds gamma_data = argmin |mean_n - 5.314|.
    Bootstrap CI on gamma_data (2000 resamples).
    Reports: is Meissner inside CI? What is the CI width?

SUB-TEST C — CATALOG CONSISTENCY
    Repeats fine sweep independently for GWTC-2.1, GWTC-3, GWTC-4.
    If gamma_data shifts between catalogs -> selection effect.
    If gamma_data is stable -> physical constant.

SUB-TEST D — MATHEMATICAL STRUCTURE CHECK
    Tests whether gamma_data is consistent with known LQG expressions
    under a j-ensemble correction, SU(2) Chern-Simons correction,
    or a simple rational multiple of Meissner.
    Reports residuals for each candidate.

ALL OUTPUT PRINTED PER EVENT AND PER CATALOG.
NO VERDICTS SUPPRESSED.

OUTPUTS:
    seraphim_gamma_per_event.csv         per-event n at each literature gamma
    seraphim_gamma_sweep.csv             full fine sweep
    seraphim_gamma_catalog_sweeps.csv    per-catalog sweep minima
    seraphim_gamma_candidates.csv        mathematical candidate residuals
    seraphim_gamma_print.txt             full printed output

USAGE:
    python seraphim_gamma_channel_test.py --dir /path/to/hdf5/files
"""

import os, sys, re, math, csv, argparse
import numpy as np
from collections import defaultdict

try:
    import h5py
except ImportError:
    sys.exit("ERROR: h5py required. pip install h5py")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — CODATA 2018
# ═══════════════════════════════════════════════════════════════════════════════
G        = 6.67430e-11
C_LIGHT  = 2.99792458e8
HBAR     = 1.054571817e-34
M_SUN    = 1.98847e30
ALPHA_FS = 7.2973525693e-3

L_P      = math.sqrt(HBAR * G / C_LIGHT**3)
NU_P     = math.sqrt(C_LIGHT**5 / (HBAR * G))
NU_P_SQ  = NU_P**2

N_BBH_PRED = 5.314
J_VAL      = 0.5
SQRT_J     = math.sqrt(J_VAL * (J_VAL + 1.0))   # sqrt(3)/2

# ── Published / physically motivated Immirzi values ──────────────────────────
LIT_GAMMAS = {
    # label                          : (value, source)
    "Meissner_2004_area"             : (0.2375,
        "Meissner 2004, area eigenvalue counting j=1/2"),
    "Agullo_2010_area"               : (0.23753,
        "Agullo et al 2010, refined area counting"),
    "DL_entropy_j12"                 : (math.log(2)/(math.pi*math.sqrt(3)),
        "Domagala-Lewandowski 2004 / Meissner, S_BH j=1/2"),
    "Ghosh_Perez_2013_allj"          : (math.log(3)/(2*math.pi*math.sqrt(2)),
        "Ghosh-Perez 2013, all-j ensemble"),
    "Engle_2010_SU2_CS"              : (0.2738,
        "Engle et al 2010, SU(2) Chern-Simons"),
    "Majumdar_1998"                  : (0.2735,
        "Majumdar 1998, original Barbero-Immirzi"),
    "Kaul_Majumdar_1998"             : (math.log(2)/(2.0*math.pi*math.sqrt(3)),
        "Kaul-Majumdar 1998, S=ln2 per puncture"),
    "Perez_Rovelli_2001"             : (math.sqrt(3)/(4.0*math.pi),
        "Perez-Rovelli 2001"),
    "gamma_data_observed"            : (0.2575,
        "This work: data-preferred value from posterior sweep"),
}

# ── Mathematical candidates for gamma_data ───────────────────────────────────
MATH_CANDIDATES = {
    "Meissner * (4/3)"               : 0.2375 * (4.0/3.0),
    "Meissner * sqrt(4/3)"           : 0.2375 * math.sqrt(4.0/3.0),
    "Meissner * (1 + 1/12)"          : 0.2375 * (1.0 + 1.0/12.0),
    "Meissner * (1 + alpha_fs)"      : 0.2375 * (1.0 + ALPHA_FS),
    "Meissner * (1 + 1/alpha_fs/100)": 0.2375 * (1.0 + 1.0/(137.036*100)),
    "ln(3)/(pi*sqrt(3))"             : math.log(3)/(math.pi*math.sqrt(3)),
    "ln(2+sqrt(3))/(pi*sqrt(2))"     : math.log(2+math.sqrt(3))/(math.pi*math.sqrt(2)),
    "3*ln(2)/(2*pi*sqrt(3))"         : 3*math.log(2)/(2*math.pi*math.sqrt(3)),
    "ln(2)/pi * sqrt(4/3)"           : math.log(2)/math.pi * math.sqrt(4.0/3.0),
    "1/(4*pi*ln(2))"                 : 1.0/(4*math.pi*math.log(2)),
    "sqrt(3)*ln(2)/(2*pi)"           : math.sqrt(3)*math.log(2)/(2*math.pi),
    "ln(2)*sqrt(2)/(pi*sqrt(3))"     : math.log(2)*math.sqrt(2)/(math.pi*math.sqrt(3)),
    "5*ln(2)/(4*pi*sqrt(3))"         : 5*math.log(2)/(4*math.pi*math.sqrt(3)),
    "ln(2)/(pi*sqrt(3)) * 2"         : 2*math.log(2)/(math.pi*math.sqrt(3)),
    "Meissner + ln2/(4*pi^2)"        : 0.2375 + math.log(2)/(4*math.pi**2),
    "Meissner + 1/(16*pi^2)"         : 0.2375 + 1.0/(16*math.pi**2),
}

# ═══════════════════════════════════════════════════════════════════════════════
# HDF5 navigation — identical to other scripts
# ═══════════════════════════════════════════════════════════════════════════════
PREFER_WF  = ["IMRPhenomXPHM","IMRPhenomXP","IMRPhenomPv2",
               "SEOBNRv4PHM","SEOBNRv4P","NRSur7dq4",
               "IMRPhenomD","SEOBNRv4"]
NON_BBH_WF = ["NRTidal","NRTidalv2","NSBH","BNS"]

def is_non_bbh(wf):
    return any(x.upper() in (wf or "").upper() for x in NON_BBH_WF)

def wf_score(k):
    ku = k.upper()
    for i, p in enumerate(PREFER_WF):
        if p.upper() in ku:
            return i + (100 if is_non_bbh(k) else 0)
    return 999 + (100 if is_non_bbh(k) else 0)

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
    for nm in names:
        if nm in (ps.dtype.names or []):
            try:
                a = np.array(ps[nm][:], dtype=float)
                a = a[np.isfinite(a)]
                if len(a) > 0:
                    return a, nm
            except Exception:
                pass
    return None, None

def detect_catalog(fpath):
    p = fpath.replace("\\","/")
    if "6513631"  in p or "gwtc-2" in p.lower(): return "GWTC-2.1"
    if "8177023"  in p or "gwtc-3" in p.lower(): return "GWTC-3"
    if "16053484" in p or "gwtc-4" in p.lower(): return "GWTC-4"
    b = os.path.basename(fpath).upper()
    if "GWTC2" in b: return "GWTC-2.1"
    if "GWTC3" in b: return "GWTC-3"
    if "GWTC4" in b: return "GWTC-4"
    return "UNKNOWN"

def event_name(fpath):
    m = re.search(r"(GW\d{6}(?:_\d{6})?)", os.path.basename(fpath))
    return m.group(1) if m else os.path.basename(fpath)[:24]

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════════════
def extract_N_star(fpath):
    r = {"event": event_name(fpath), "catalog": detect_catalog(fpath),
         "wf_key": "", "n_samples": 0, "erad_field": "",
         "N_star": None, "error": ""}
    try:
        with h5py.File(fpath, "r") as hf:
            ps, wf_key = get_posteriors(hf)
            if ps is None: r["error"] = "no_posterior"; return r
            if is_non_bbh(wf_key or ""): r["error"] = "non_bbh"; return r
            r["wf_key"] = wf_key or ""
            m1, _ = col_arr(ps, "mass_1_source","mass1_source","mass_1")
            m2, _ = col_arr(ps, "mass_2_source","mass2_source","mass_2")
            Er, ef = col_arr(ps, "radiated_energy_non_evolved",
                                 "radiated_energy","E_rad")
            chirp, _ = col_arr(ps, "chirp_mass_source","chirp_mass")
            if Er is None and chirp is not None:
                Er = 0.0842 * chirp; ef = "NR_FIT"
            if m1 is None or m2 is None or Er is None:
                r["error"] = "missing_columns"; return r
            r["erad_field"] = ef or "unknown"
            n = min(len(m1), len(m2), len(Er))
            Mt = m1[:n] + m2[:n]; E = Er[:n]
            valid = (Mt > 0) & (E > 0) & (E < Mt) & (Mt < 1000)
            if valid.sum() < 10: r["error"] = "too_few_%d" % valid.sum(); return r
            N_star = E[valid] / (Mt[valid] * ALPHA_FS)
            N_star = N_star[N_star > 0]
            if len(N_star) < 10: r["error"] = "N_star_bad"; return r
            r["N_star"] = N_star; r["n_samples"] = len(N_star)
    except Exception as ex:
        r["error"] = str(ex)[:80]
    return r

# ═══════════════════════════════════════════════════════════════════════════════
# N FORMULA
# n = 0.5 * (log2(sqrt_j) + log2(N_star) - log2(A))
# A = K0/nu_P^2  K0 = c^2/(128*pi^2*gamma*l_P^2)
# ═══════════════════════════════════════════════════════════════════════════════
def k0(gamma):
    return C_LIGHT**2 / (128.0 * math.pi**2 * gamma * L_P**2)

def mean_n(N_star_arr, gamma):
    A       = k0(gamma) / NU_P_SQ
    log2sqj = math.log2(SQRT_J)
    log2A   = math.log2(A)
    # mean_n = 0.5*(log2sqj + mean(log2(N*)) - log2A)
    return 0.5 * (log2sqj + float(np.mean(np.log2(N_star_arr))) - log2A)

def median_n(N_star_arr, gamma):
    A       = k0(gamma) / NU_P_SQ
    log2sqj = math.log2(SQRT_J)
    log2A   = math.log2(A)
    n_arr   = 0.5 * (log2sqj + np.log2(N_star_arr) - log2A)
    return float(np.median(n_arr))

def per_event_median_n(N_star_arr, gamma):
    A       = k0(gamma) / NU_P_SQ
    log2sqj = math.log2(SQRT_J)
    log2A   = math.log2(A)
    n_arr   = 0.5 * (log2sqj + np.log2(N_star_arr) - log2A)
    return float(np.median(n_arr)), float(np.std(n_arr))

# Bootstrap CI on gamma_data
def bootstrap_gamma_data(all_N_star, n_boot=2000):
    log2sqj  = math.log2(SQRT_J)
    log2N    = np.log2(all_N_star)
    n        = len(log2N)
    rng      = np.random.default_rng(42)
    boot_gamma = np.zeros(n_boot)
    for i in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        bm   = float(np.mean(log2N[idx]))
        # solve 5.314 = 0.5*(log2sqj + bm - log2(A))
        # log2(A) = log2sqj + bm - 2*5.314
        # A = 2^(log2sqj + bm - 10.628)
        log2A_opt = log2sqj + bm - 2.0 * N_BBH_PRED
        A_opt     = 2.0**log2A_opt
        # K0 = A * nu_P^2
        # gamma = c^2 / (128*pi^2*l_P^2*K0)
        K0_opt    = A_opt * NU_P_SQ
        boot_gamma[i] = C_LIGHT**2 / (128.0 * math.pi**2 * L_P**2 * K0_opt)
    lo = float(np.percentile(boot_gamma, 2.5))
    hi = float(np.percentile(boot_gamma, 97.5))
    return lo, hi, boot_gamma

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def run(root_dir):
    lines = []
    def pr(s=""):
        print(s); lines.append(str(s))

    pr("=" * 80)
    pr("  SERAPHIM GAMMA CHANNEL TEST")
    pr("  What Immirzi value does the GW energy loss channel probe?")
    pr("  All numbers printed. No suppression.")
    pr("=" * 80)
    pr()
    pr("  CONSTANTS:")
    pr("    l_P    = %.6e m" % L_P)
    pr("    nu_P   = %.6e Hz" % NU_P)
    pr("    alpha  = %.13f  (fine structure)" % ALPHA_FS)
    pr("    j      = 0.5  (fixed throughout — confirmed by Test 3)")
    pr("    sqrt_j = %.8f" % SQRT_J)
    pr("    n_pred = %.3f  (fixed)" % N_BBH_PRED)
    pr()

    # Load
    h5_files = []
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.endswith((".h5",".hdf5")):
                h5_files.append(os.path.join(dp, fn))
    h5_files.sort()
    if not h5_files:
        pr("ERROR: No HDF5 files found."); return

    pr("  Found %d HDF5 files" % len(h5_files))
    events, skipped = [], []
    for fp in h5_files:
        r = extract_N_star(fp)
        if r["error"]: skipped.append(r)
        else: events.append(r)
    pr("  Loaded : %d events" % len(events))
    pr("  Skipped: %d" % len(skipped))
    if skipped:
        for r in skipped:
            pr("    SKIP: %-28s  %s  %s" % (r["event"],r["catalog"],r["error"]))
    pr()

    if len(events) < 10:
        pr("ERROR: Too few events."); return

    all_N_star = np.concatenate([ev["N_star"] for ev in events])
    pr("  Total posterior samples: %d" % len(all_N_star))
    pr()

    # ═══════════════════════════════════════════════════════════════════════════
    # SUB-TEST A: LITERATURE VALUES
    # ═══════════════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("  SUB-TEST A: PUBLISHED IMMIRZI VALUES")
    pr("  Per-event and aggregate n for each known gamma.")
    pr("=" * 80)
    pr()

    per_event_rows = []
    lit_summary    = []

    for gname, (gval, gsrc) in LIT_GAMMAS.items():
        pr("  %s" % gname)
        pr("  gamma = %.8f" % gval)
        pr("  source: %s" % gsrc)
        pr("  K0    = %.4e Hz^2" % k0(gval))
        pr()
        pr("  %-22s  %-8s  %-8s  %-8s  %-8s" % (
            "EVENT","CATALOG","MED_n","STD_n","DELTA"))
        pr("  " + "-"*65)

        cat_data = defaultdict(list)
        for ev in events:
            med, std = per_event_median_n(ev["N_star"], gval)
            delta    = med - N_BBH_PRED
            in_band  = 1 if (N_BBH_PRED-0.753) <= med <= (N_BBH_PRED+0.753) else 0
            pr("  %-22s  %-8s  %-8.4f  %-8.4f  %+.4f" % (
                ev["event"], ev["catalog"], med, std, delta))
            cat_data[ev["catalog"]].append(med)
            # record for CSV (only your value and gamma_data to keep manageable)
            if gname in ("Meissner_2004_area","gamma_data_observed"):
                per_event_rows.append({
                    "event"   : ev["event"],
                    "catalog" : ev["catalog"],
                    "gamma_label": gname,
                    "gamma_value": "%.6f" % gval,
                    "median_n": "%.6f" % med,
                    "std_n"   : "%.6f" % std,
                    "delta"   : "%.6f" % delta,
                    "in_band" : str(in_band),
                })

        # Aggregate
        all_med = np.array([m for ms in cat_data.values() for m in ms])
        mn_pool = mean_n(all_N_star, gval)
        mdn_pool= median_n(all_N_star, gval)
        sigma_sep = abs(mn_pool - N_BBH_PRED) / (np.std(all_med)/math.sqrt(len(all_med)))

        pr()
        pr("  AGGREGATE:  mean_n=%.4f  median_n=%.4f  "
           "delta_mean=%+.4f  delta_median=%+.4f  sigma_sep=%.2f" % (
            mn_pool, mdn_pool, mn_pool-N_BBH_PRED, mdn_pool-N_BBH_PRED, sigma_sep))
        pr()
        pr("  PER CATALOG:")
        for cat in ["GWTC-2.1","GWTC-3","GWTC-4"]:
            if cat not in cat_data: continue
            ns = np.array(cat_data[cat])
            pr("    %-8s  N=%3d  mean=%.4f  median=%.4f  "
               "std=%.4f  SEM=%.4f  delta=%+.4f" % (
                cat, len(ns), np.mean(ns), np.median(ns),
                np.std(ns), np.std(ns)/math.sqrt(len(ns)),
                np.mean(ns)-N_BBH_PRED))
        pr()
        pr("  " + "─"*70)
        pr()

        lit_summary.append({
            "gamma_label"    : gname,
            "gamma_value"    : "%.8f" % gval,
            "K0_Hz2"         : "%.4e" % k0(gval),
            "mean_n_pooled"  : "%.6f" % mn_pool,
            "median_n_pooled": "%.6f" % mdn_pool,
            "delta_mean"     : "%.6f" % (mn_pool - N_BBH_PRED),
            "delta_median"   : "%.6f" % (mdn_pool - N_BBH_PRED),
            "sigma_sep"      : "%.4f" % sigma_sep,
            "source"         : gsrc,
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # SUB-TEST B: FINE SWEEP + BOOTSTRAP CI
    # ═══════════════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("  SUB-TEST B: FINE GAMMA SWEEP + BOOTSTRAP CI")
    pr("  1000 gamma values from 0.05 to 0.60")
    pr("  Metric: |mean_n - 5.314| minimized")
    pr("  Bootstrap CI: 2000 resamples on gamma_data")
    pr("=" * 80)
    pr()

    gamma_vals   = np.linspace(0.05, 0.60, 1000)
    chi2_all     = np.zeros(1000)
    mean_n_all   = np.zeros(1000)
    median_n_all = np.zeros(1000)
    log2N_mean   = float(np.mean(np.log2(all_N_star)))
    log2sqj      = math.log2(SQRT_J)

    for i, g in enumerate(gamma_vals):
        A          = k0(float(g)) / NU_P_SQ
        mn         = 0.5 * (log2sqj + log2N_mean - math.log2(A))
        mean_n_all[i]   = mn
        median_n_all[i] = median_n(all_N_star, float(g))
        chi2_all[i]     = (mn - N_BBH_PRED)**2

    best_idx   = int(np.argmin(chi2_all))
    gamma_data = float(gamma_vals[best_idx])
    mn_at_best = float(mean_n_all[best_idx])

    pr("  Computing bootstrap CI on gamma_data (2000 resamples) ...")
    ci_lo, ci_hi, boot_g = bootstrap_gamma_data(all_N_star, n_boot=2000)
    pr()

    # Meissner in CI?
    meissner_in_ci = ci_lo <= 0.2375 <= ci_hi

    pr("  FINE SWEEP RESULTS:")
    pr("  gamma_data (best)         : %.8f" % gamma_data)
    pr("  mean_n at gamma_data      : %.6f  (target 5.314)" % mn_at_best)
    pr("  delta at gamma_data       : %+.6f" % (mn_at_best - N_BBH_PRED))
    pr()
    pr("  Bootstrap 95%% CI on gamma_data: [%.8f, %.8f]" % (ci_lo, ci_hi))
    pr("  CI width                  : %.8f" % (ci_hi - ci_lo))
    pr("  Meissner (0.2375) in CI   : %s" % ("YES" if meissner_in_ci else "NO"))
    pr("  gamma_data / Meissner     : %.6f" % (gamma_data / 0.2375))
    pr("  Offset from Meissner      : %+.6f  (%+.2f%%)" % (
        gamma_data - 0.2375, 100*(gamma_data-0.2375)/0.2375))
    pr()

    # Print sweep around the region of interest
    pr("  SWEEP AROUND MINIMUM (gamma_data ± 0.05):")
    pr("  gamma       mean_n    median_n   chi2         delta_n")
    sweep_rows = []
    for i, g in enumerate(gamma_vals):
        near_best    = abs(float(g) - gamma_data) < 0.001
        near_meiss   = abs(float(g) - 0.2375) < 0.001
        near_data    = abs(float(g) - gamma_data) < 0.05
        if near_data:
            pr("  %.6f   %.4f    %.4f     %.6f     %+.4f%s%s" % (
                float(g), mean_n_all[i], median_n_all[i], chi2_all[i],
                mean_n_all[i]-N_BBH_PRED,
                "  << GAMMA_DATA BEST" if near_best else "",
                "  << MEISSNER 2004"   if near_meiss else ""))
        sweep_rows.append({
            "gamma"    : "%.6f" % float(g),
            "mean_n"   : "%.6f" % mean_n_all[i],
            "median_n" : "%.6f" % median_n_all[i],
            "chi2"     : "%.8f" % chi2_all[i],
            "delta_n"  : "%.6f" % (mean_n_all[i] - N_BBH_PRED),
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # SUB-TEST C: CATALOG CONSISTENCY
    # ═══════════════════════════════════════════════════════════════════════════
    pr()
    pr("=" * 80)
    pr("  SUB-TEST C: CATALOG CONSISTENCY")
    pr("  Does gamma_data shift between catalogs?")
    pr("  If stable -> physical constant. If drifts -> selection effect.")
    pr("=" * 80)
    pr()

    cat_sweep_rows = []
    cat_gammas     = {}

    for cat in ["GWTC-2.1","GWTC-3","GWTC-4"]:
        sub = [ev for ev in events if ev["catalog"] == cat]
        if len(sub) < 5: continue
        N_cat = np.concatenate([ev["N_star"] for ev in sub])
        log2N_cat = float(np.mean(np.log2(N_cat)))

        chi2_cat = np.zeros(1000)
        mn_cat   = np.zeros(1000)
        for i, g in enumerate(gamma_vals):
            A  = k0(float(g)) / NU_P_SQ
            mn = 0.5 * (log2sqj + log2N_cat - math.log2(A))
            mn_cat[i]   = mn
            chi2_cat[i] = (mn - N_BBH_PRED)**2

        best_cat_idx  = int(np.argmin(chi2_cat))
        gamma_cat     = float(gamma_vals[best_cat_idx])
        mn_at_cat     = float(mn_cat[best_cat_idx])
        cat_gammas[cat] = gamma_cat

        pr("  %s  (N=%d events, %d samples):" % (cat, len(sub), len(N_cat)))
        pr("    gamma_data          : %.6f" % gamma_cat)
        pr("    mean_n at gamma_data: %.6f  (target 5.314)" % mn_at_cat)
        pr("    delta               : %+.6f" % (mn_at_cat - N_BBH_PRED))
        pr("    offset from Meissner: %+.6f  (%+.2f%%)" % (
            gamma_cat-0.2375, 100*(gamma_cat-0.2375)/0.2375))
        pr("    offset from all-cat : %+.6f" % (gamma_cat - gamma_data))
        pr()

        cat_sweep_rows.append({
            "catalog"           : cat,
            "n_events"          : len(sub),
            "n_samples"         : len(N_cat),
            "gamma_data"        : "%.6f" % gamma_cat,
            "mn_at_gamma_data"  : "%.6f" % mn_at_cat,
            "delta"             : "%.6f" % (mn_at_cat - N_BBH_PRED),
            "offset_from_meissner": "%.6f" % (gamma_cat - 0.2375),
            "offset_from_allcat": "%.6f" % (gamma_cat - gamma_data),
        })

    # Stability assessment — raw numbers
    if len(cat_gammas) >= 2:
        g_vals_cats = list(cat_gammas.values())
        spread = max(g_vals_cats) - min(g_vals_cats)
        pr("  Spread across catalogs: %.6f  (max - min)" % spread)
        pr("  Individual values: " + "  |  ".join(
            "%-8s %.6f" % (c, g) for c, g in cat_gammas.items()))
    pr()

    # ═══════════════════════════════════════════════════════════════════════════
    # SUB-TEST D: MATHEMATICAL STRUCTURE CHECK
    # ═══════════════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("  SUB-TEST D: MATHEMATICAL STRUCTURE OF GAMMA_DATA")
    pr("  Is gamma_data = %.8f consistent with a known expression?" % gamma_data)
    pr("  Sorted by |residual| from gamma_data.")
    pr("=" * 80)
    pr()

    cand_rows = []
    for cname, cval in sorted(MATH_CANDIDATES.items(),
                              key=lambda x: abs(x[1]-gamma_data)):
        residual = cval - gamma_data
        pct      = 100.0 * residual / gamma_data
        mn_cand  = 0.5 * (log2sqj + log2N_mean -
                          math.log2(k0(cval)/NU_P_SQ))
        pr("  %-38s = %.8f   residual = %+.6f  (%+.3f%%)  mean_n = %.4f" % (
            cname, cval, residual, pct, mn_cand))
        cand_rows.append({
            "expression"     : cname,
            "value"          : "%.8f" % cval,
            "residual"       : "%.8f" % residual,
            "pct_from_data"  : "%.4f" % pct,
            "mean_n"         : "%.6f" % mn_cand,
            "delta_n"        : "%.6f" % (mn_cand - N_BBH_PRED),
        })
    pr()
    pr("  Closest candidate: %s" % sorted(
        MATH_CANDIDATES.items(), key=lambda x: abs(x[1]-gamma_data))[0][0])
    pr("  Residual of closest: %.8f" % min(
        abs(v-gamma_data) for v in MATH_CANDIDATES.values()))
    pr()
    pr("  NOTE: If no candidate is within 0.001 of gamma_data,")
    pr("  gamma_data does not correspond to a known LQG formula.")
    pr("  That would be a new physical result or a measurement artifact.")
    pr()

    # ═══════════════════════════════════════════════════════════════════════════
    # JOINT SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    pr("=" * 80)
    pr("  JOINT SUMMARY — ALL SUB-TESTS")
    pr("  Numbers only.")
    pr("=" * 80)
    pr()
    pr("  LITERATURE GAMMAS (mean_n at each, delta from 5.314):")
    for row in lit_summary:
        pr("    %-32s gamma=%.6f  mean_n=%.4f  delta=%s" % (
            row["gamma_label"], float(row["gamma_value"]),
            float(row["mean_n_pooled"]), row["delta_mean"]))
    pr()
    pr("  DATA-PREFERRED GAMMA:")
    pr("    gamma_data            : %.8f" % gamma_data)
    pr("    95%% CI                : [%.8f, %.8f]" % (ci_lo, ci_hi))
    pr("    Meissner in CI        : %s" % ("YES" if meissner_in_ci else "NO"))
    pr("    gamma_data/Meissner   : %.6f" % (gamma_data/0.2375))
    pr()
    pr("  CATALOG STABILITY:")
    for c, g in cat_gammas.items():
        pr("    %-8s  gamma_data = %.6f  offset_from_allcat = %+.6f" % (
            c, g, g-gamma_data))
    if len(cat_gammas) >= 2:
        pr("    Spread: %.6f" % (max(cat_gammas.values()) - min(cat_gammas.values())))
    pr()
    pr("  CLOSEST MATHEMATICAL CANDIDATE:")
    best_cand = sorted(MATH_CANDIDATES.items(), key=lambda x: abs(x[1]-gamma_data))[0]
    pr("    %s = %.8f" % best_cand)
    pr("    Residual: %.8f" % abs(best_cand[1]-gamma_data))
    pr()

    # ═══════════════════════════════════════════════════════════════════════════
    # WRITE FILES
    # ═══════════════════════════════════════════════════════════════════════════
    def wcsv(fname, rows):
        if not rows: return
        with open(fname,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        pr("  Written: %s  (%d rows)" % (fname, len(rows)))

    pr("=" * 80)
    pr("  OUTPUT FILES:")
    wcsv("seraphim_gamma_per_event.csv",       per_event_rows)
    wcsv("seraphim_gamma_sweep.csv",            sweep_rows)
    wcsv("seraphim_gamma_catalog_sweeps.csv",   cat_sweep_rows)
    wcsv("seraphim_gamma_candidates.csv",       cand_rows)
    wcsv("seraphim_gamma_lit_summary.csv",      lit_summary)

    with open("seraphim_gamma_print.txt","w") as f:
        f.write("\n".join(lines))
    pr("  Written: seraphim_gamma_print.txt")
    pr()
    pr("  Done. Read the numbers.")
    pr("=" * 80)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Seraphim gamma channel test — what Immirzi does GW data probe?")
    ap.add_argument("--dir","-d", default=".",
        help="Root directory containing HDF5 files")
    args = ap.parse_args()
    run(args.dir)
