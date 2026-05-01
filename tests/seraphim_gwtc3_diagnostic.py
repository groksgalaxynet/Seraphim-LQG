"""
seraphim_gwtc3_diagnostic.py
============================
GWTC-3 SCATTER DIAGNOSTIC

Question: Is the GWTC-3 gamma_data = 0.276 (vs 0.246-0.248 for other catalogs)
driven by a handful of high-scatter events, or is it catalog-wide?

METHOD:
- Apply progressive quality cuts to GWTC-3 events
- At each cut level, recompute gamma_data independently
- If gamma_data tightens toward GWTC-2.1/GWTC-4 values as cuts tighten
  -> scatter events are the problem, not the catalog
- If gamma_data stays at 0.276 regardless of cuts
  -> systematic is catalog-wide (waveform model, calibration, selection)

CUT LEVELS:
  Level 0: All GWTC-3 events (baseline, 72 events)
  Level 1: Remove std_n > 0.40 (worst 6 events)
  Level 2: Remove std_n > 0.30 (9 events)
  Level 3: Remove std_n > 0.20 (17 events)
  Level 4: Remove std_n > 0.15 (tighter)
  Level 5: Remove NSBH outliers (n < 4.56) AND std_n > 0.20
  Level 6: Remove NR_FIT events (only real E_rad posteriors)

Also runs same cuts on GWTC-2.1 and GWTC-4 as controls.
If cuts on GWTC-2.1/GWTC-4 DON'T change their gamma_data much
but cuts on GWTC-3 DO -> confirms GWTC-3 scatter is the driver.

PRINTS EVERYTHING PER EVENT AT EACH CUT LEVEL.
NO SUPPRESSION.

OUTPUTS:
    seraphim_gwtc3_diag_per_cut.csv      gamma_data per cut level per catalog
    seraphim_gwtc3_diag_removed.csv      which events removed at each level
    seraphim_gwtc3_diag_print.txt        full printed output

USAGE:
    python seraphim_gwtc3_diagnostic.py --dir /path/to/hdf5/files
"""

import os, sys, re, math, csv, argparse
import numpy as np
from collections import defaultdict

try:
    import h5py
except ImportError:
    sys.exit("ERROR: h5py required. pip install h5py")

# ── Constants ─────────────────────────────────────────────────────────────────
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
SQRT_J     = math.sqrt(J_VAL * (J_VAL + 1.0))
GAMMA_MEISSNER = 0.2375

# Known NSBH / outlier events in GWTC-3 (n < 4.56 at Meissner K0)
NSBH_EVENTS = {
    "GW191219_163120",  # n ~ 3.27
    "GW200105_162426",  # n ~ 4.69
    "GW200115_042309",  # n ~ 4.74
    "GW200210_092254",  # n ~ 4.22
    "GW191113_071753",  # n ~ 4.62
}

# ── HDF5 navigation ───────────────────────────────────────────────────────────
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

# ── Load ──────────────────────────────────────────────────────────────────────
def extract_event(fpath):
    r = {"event": event_name(fpath), "catalog": detect_catalog(fpath),
         "wf_key": "", "n_samples": 0, "erad_field": "",
         "N_star": None, "median_n": None, "std_n": None,
         "is_nsbh": False, "error": ""}
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
            if valid.sum() < 10: r["error"] = "too_few"; return r
            N_star = E[valid] / (Mt[valid] * ALPHA_FS)
            N_star = N_star[N_star > 0]
            if len(N_star) < 10: r["error"] = "N_star_bad"; return r
            # compute n at Meissner K0
            k0 = C_LIGHT**2 / (128.0*math.pi**2*GAMMA_MEISSNER*L_P**2)
            A  = k0 / NU_P_SQ
            n_arr = 0.5*(math.log2(SQRT_J) + np.log2(N_star) - math.log2(A))
            r["N_star"]    = N_star
            r["n_samples"] = len(N_star)
            r["median_n"]  = float(np.median(n_arr))
            r["std_n"]     = float(np.std(n_arr))
            r["is_nsbh"]   = r["event"] in NSBH_EVENTS
    except Exception as ex:
        r["error"] = str(ex)[:80]
    return r

# ── Gamma recovery ────────────────────────────────────────────────────────────
def gamma_data_from_events(events):
    """Find gamma that makes mean_n = 5.314 for this set of events."""
    if not events: return None, None
    all_N = np.concatenate([ev["N_star"] for ev in events])
    log2sqj   = math.log2(SQRT_J)
    log2N_mean = float(np.mean(np.log2(all_N)))
    # solve: 5.314 = 0.5*(log2sqj + log2N_mean - log2(A))
    # log2(A) = log2sqj + log2N_mean - 2*5.314
    log2A_opt = log2sqj + log2N_mean - 2.0*N_BBH_PRED
    A_opt     = 2.0**log2A_opt
    K0_opt    = A_opt * NU_P_SQ
    gamma_opt = C_LIGHT**2 / (128.0*math.pi**2*L_P**2*K0_opt)
    # mean_n at this gamma
    mn = 0.5*(log2sqj + log2N_mean - log2A_opt)
    return float(gamma_opt), float(mn)

def mean_n_at_meissner(events):
    if not events: return None
    all_N = np.concatenate([ev["N_star"] for ev in events])
    k0  = C_LIGHT**2 / (128.0*math.pi**2*GAMMA_MEISSNER*L_P**2)
    A   = k0 / NU_P_SQ
    log2sqj = math.log2(SQRT_J)
    return float(0.5*(log2sqj + np.mean(np.log2(all_N)) - math.log2(A)))

def bootstrap_gamma(events, n_boot=2000):
    if not events: return None, None
    all_N  = np.concatenate([ev["N_star"] for ev in events])
    log2N  = np.log2(all_N)
    log2sqj = math.log2(SQRT_J)
    n      = len(log2N)
    rng    = np.random.default_rng(42)
    boot_g = np.zeros(n_boot)
    for i in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        bm   = float(np.mean(log2N[idx]))
        log2A_opt = log2sqj + bm - 2.0*N_BBH_PRED
        A_opt     = 2.0**log2A_opt
        K0_opt    = A_opt * NU_P_SQ
        boot_g[i] = C_LIGHT**2/(128.0*math.pi**2*L_P**2*K0_opt)
    lo = float(np.percentile(boot_g, 2.5))
    hi = float(np.percentile(boot_g, 97.5))
    return lo, hi

# ── Cut definitions ───────────────────────────────────────────────────────────
def apply_cut(events, level):
    """
    Returns (kept, removed, description) for a given cut level.
    Level 0 = no cuts (baseline).
    """
    if level == 0:
        return events, [], "No cuts (baseline)"
    elif level == 1:
        kept    = [ev for ev in events if ev["std_n"] <= 0.40]
        removed = [ev for ev in events if ev["std_n"] >  0.40]
        return kept, removed, "Remove std_n > 0.40"
    elif level == 2:
        kept    = [ev for ev in events if ev["std_n"] <= 0.30]
        removed = [ev for ev in events if ev["std_n"] >  0.30]
        return kept, removed, "Remove std_n > 0.30"
    elif level == 3:
        kept    = [ev for ev in events if ev["std_n"] <= 0.20]
        removed = [ev for ev in events if ev["std_n"] >  0.20]
        return kept, removed, "Remove std_n > 0.20"
    elif level == 4:
        kept    = [ev for ev in events if ev["std_n"] <= 0.15]
        removed = [ev for ev in events if ev["std_n"] >  0.15]
        return kept, removed, "Remove std_n > 0.15"
    elif level == 5:
        kept    = [ev for ev in events
                   if ev["std_n"] <= 0.20 and not ev["is_nsbh"]]
        removed = [ev for ev in events
                   if ev["std_n"] > 0.20 or ev["is_nsbh"]]
        return kept, removed, "Remove std_n > 0.20 AND NSBH outliers"
    elif level == 6:
        kept    = [ev for ev in events if "NR_FIT" not in ev["erad_field"]]
        removed = [ev for ev in events if "NR_FIT" in ev["erad_field"]]
        return kept, removed, "Remove NR_FIT E_rad (keep real posteriors only)"
    elif level == 7:
        kept    = [ev for ev in events
                   if "NR_FIT" not in ev["erad_field"]
                   and ev["std_n"] <= 0.20
                   and not ev["is_nsbh"]]
        removed = [ev for ev in events
                   if "NR_FIT" in ev["erad_field"]
                   or ev["std_n"] > 0.20
                   or ev["is_nsbh"]]
        return kept, removed, "Remove NR_FIT + std_n > 0.20 + NSBH (strictest)"
    return events, [], "Unknown cut"

# ── Main ──────────────────────────────────────────────────────────────────────
def run(root_dir):
    lines = []
    def pr(s=""):
        print(s); lines.append(str(s))

    pr("="*72)
    pr("  SERAPHIM GWTC-3 SCATTER DIAGNOSTIC")
    pr("  Does gamma_data tighten toward 0.246-0.248 as scatter is removed?")
    pr("  All numbers printed. No suppression.")
    pr("="*72)
    pr()
    pr("  BASELINE GAMMA VALUES FROM FULL CATALOG TEST:")
    pr("    GWTC-2.1  gamma_data = 0.247648")
    pr("    GWTC-3    gamma_data = 0.276276  <- outlier under investigation")
    pr("    GWTC-4    gamma_data = 0.245445")
    pr("    Meissner  gamma*     = 0.237500")
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
    all_events, skipped = [], []
    for fp in h5_files:
        r = extract_event(fp)
        if r["error"]: skipped.append(r)
        else: all_events.append(r)
    pr("  Loaded : %d  Skipped: %d" % (len(all_events), len(skipped)))
    pr()

    # Split by catalog
    by_cat = defaultdict(list)
    for ev in all_events:
        by_cat[ev["catalog"]].append(ev)

    pr("  Events per catalog:")
    for cat in ["GWTC-2.1","GWTC-3","GWTC-4"]:
        evs = by_cat[cat]
        pr("    %-8s  %d events  %d samples" % (
            cat, len(evs), sum(ev["n_samples"] for ev in evs)))
    pr()

    # ── Per-event table for GWTC-3 ────────────────────────────────────────────
    pr("="*72)
    pr("  GWTC-3 PER-EVENT DETAIL (sorted by std_n descending)")
    pr("  Shows which events are driving the scatter")
    pr("="*72)
    pr()
    pr("  %-24s  %-7s  %-7s  %-5s  %-5s  %s" % (
        "EVENT","MEDIAN_N","STD_N","NSBH","NR_FIT","ERAD_FIELD"))
    pr("  " + "-"*80)

    g3_sorted = sorted(by_cat["GWTC-3"], key=lambda ev: -ev["std_n"])
    for ev in g3_sorted:
        nsbh_flag = "YES" if ev["is_nsbh"] else "no"
        nr_flag   = "YES" if "NR_FIT" in ev["erad_field"] else "no"
        pr("  %-24s  %-7.4f  %-7.4f  %-5s  %-5s  %s" % (
            ev["event"], ev["median_n"], ev["std_n"],
            nsbh_flag, nr_flag, ev["erad_field"][:30]))
    pr()

    # ── Progressive cut analysis ───────────────────────────────────────────────
    pr("="*72)
    pr("  PROGRESSIVE CUT ANALYSIS")
    pr("  Applying cuts to GWTC-3, GWTC-2.1, GWTC-4")
    pr("  Key metric: gamma_data at each cut level")
    pr("  Control question: do GWTC-2.1 and GWTC-4 stay stable?")
    pr("="*72)

    cut_rows   = []
    removed_rows = []
    N_CUTS = 8

    for level in range(N_CUTS):
        pr()
        pr("  ── CUT LEVEL %d ──────────────────────────────────────────────" % level)

        for cat in ["GWTC-3","GWTC-2.1","GWTC-4"]:
            cat_events = by_cat[cat]
            if not cat_events: continue

            kept, removed, desc = apply_cut(cat_events, level)
            if level == 0:
                pr("  %s — %s" % (cat, desc))
            else:
                pr()
                pr("  %s — %s" % (cat, desc))
                if removed:
                    pr("    REMOVED (%d events):" % len(removed))
                    for ev in sorted(removed, key=lambda e: -e["std_n"]):
                        pr("      %-24s  std=%.4f  n=%.4f  nsbh=%-3s  erad=%s" % (
                            ev["event"], ev["std_n"], ev["median_n"],
                            "yes" if ev["is_nsbh"] else "no",
                            ev["erad_field"][:25]))

            if len(kept) < 3:
                pr("    TOO FEW EVENTS AFTER CUT (%d) — skipping" % len(kept))
                continue

            gamma_d, mn_at_gd = gamma_data_from_events(kept)
            mn_meiss           = mean_n_at_meissner(kept)
            ci_lo, ci_hi       = bootstrap_gamma(kept, n_boot=1000)
            meiss_in_ci        = ci_lo <= GAMMA_MEISSNER <= ci_hi if ci_lo else False

            # Per-event n values at recovered gamma
            k0_gd = C_LIGHT**2/(128.0*math.pi**2*gamma_d*L_P**2)
            A_gd  = k0_gd / NU_P_SQ
            log2sqj = math.log2(SQRT_J)
            medians = []
            for ev in kept:
                n_arr = 0.5*(log2sqj + np.log2(ev["N_star"]) - math.log2(A_gd))
                medians.append(float(np.median(n_arr)))
            medians = np.array(medians)

            pr()
            pr("    KEPT: %d events  |  %d samples" % (
                len(kept), sum(ev["n_samples"] for ev in kept)))
            pr("    gamma_data           : %.6f  (%.2f%% from Meissner)" % (
                gamma_d, 100*(gamma_d-GAMMA_MEISSNER)/GAMMA_MEISSNER))
            pr("    mean_n at gamma_data : %.6f  (target 5.314)" % mn_at_gd)
            pr("    mean_n at Meissner   : %.6f  (delta %+.6f)" % (
                mn_meiss, mn_meiss - N_BBH_PRED))
            pr("    95%% CI on gamma_data : [%.6f, %.6f]" % (ci_lo, ci_hi))
            pr("    Meissner in CI       : %s" % ("YES" if meiss_in_ci else "NO"))
            pr("    Per-event n (at gamma_data): "
               "mean=%.4f  median=%.4f  std=%.4f" % (
                np.mean(medians), np.median(medians), np.std(medians)))

            cut_rows.append({
                "catalog"         : cat,
                "cut_level"       : level,
                "cut_desc"        : desc,
                "n_kept"          : len(kept),
                "n_removed"       : len(removed),
                "n_samples"       : sum(ev["n_samples"] for ev in kept),
                "gamma_data"      : "%.6f" % gamma_d,
                "pct_from_meissner": "%.4f" % (100*(gamma_d-GAMMA_MEISSNER)/GAMMA_MEISSNER),
                "mn_at_gamma_data": "%.6f" % mn_at_gd,
                "mn_at_meissner"  : "%.6f" % mn_meiss,
                "delta_meissner"  : "%.6f" % (mn_meiss - N_BBH_PRED),
                "ci_lo"           : "%.6f" % ci_lo if ci_lo else "",
                "ci_hi"           : "%.6f" % ci_hi if ci_hi else "",
                "meissner_in_ci"  : "YES" if meiss_in_ci else "NO",
            })

            if level > 0 and removed:
                for ev in removed:
                    removed_rows.append({
                        "catalog"   : cat,
                        "cut_level" : level,
                        "cut_desc"  : desc,
                        "event"     : ev["event"],
                        "median_n"  : "%.4f" % ev["median_n"],
                        "std_n"     : "%.4f" % ev["std_n"],
                        "is_nsbh"   : "yes" if ev["is_nsbh"] else "no",
                        "erad_field": ev["erad_field"],
                    })

    # ── Summary table ─────────────────────────────────────────────────────────
    pr()
    pr("="*72)
    pr("  SUMMARY: GAMMA_DATA ACROSS CUT LEVELS")
    pr("  Does GWTC-3 gamma converge toward GWTC-2.1/GWTC-4 as cuts tighten?")
    pr("="*72)
    pr()
    pr("  %-8s  %-7s  %-10s  %-10s  %-7s  %s" % (
        "CATALOG","CUT","GAMMA_DATA","MEISSNER%","N_KEPT","CI_MEISS_IN"))
    pr("  " + "-"*65)

    for row in cut_rows:
        pr("  %-8s  %-7s  %-10s  %-10s  %-7s  %s" % (
            row["catalog"],
            row["cut_level"],
            row["gamma_data"],
            row["pct_from_meissner"] + "%",
            row["n_kept"],
            row["meissner_in_ci"]))

    # ── Convergence check ──────────────────────────────────────────────────────
    pr()
    pr("="*72)
    pr("  CONVERGENCE CHECK")
    pr("  At strictest cut (level 7), where does each catalog land?")
    pr("="*72)
    pr()

    for cat in ["GWTC-3","GWTC-2.1","GWTC-4"]:
        strict = [r for r in cut_rows if r["catalog"]==cat and int(r["cut_level"])==7]
        if strict:
            r = strict[0]
            pr("  %-8s  gamma_data=%-10s  (%s from Meissner)  N=%s  Meissner_in_CI=%s" % (
                cat, r["gamma_data"], r["pct_from_meissner"]+"%",
                r["n_kept"], r["meissner_in_ci"]))

    pr()
    pr("  INTERPRETATION GUIDE (numbers, not conclusions):")
    pr("  If GWTC-3 gamma_data moves toward 0.246-0.248 as cuts tighten:")
    pr("    -> High-scatter events are driving the 0.276 value")
    pr("    -> Catalog systematic is localized, not catalog-wide")
    pr("  If GWTC-3 gamma_data stays near 0.276 regardless of cuts:")
    pr("    -> Something catalog-wide: waveform model, calibration, or selection")
    pr("    -> Deeper investigation needed (waveform comparison, SNR stratification)")
    pr("  If GWTC-2.1 and GWTC-4 gamma_data stay stable across cuts:")
    pr("    -> Their values are robust, GWTC-3 is the outlier")
    pr("  If all three converge at strictest cut:")
    pr("    -> A common gamma exists, scatter was the problem")
    pr()

    # ── Write outputs ──────────────────────────────────────────────────────────
    def wcsv(fname, rows):
        if not rows: return
        with open(fname,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        pr("  Written: %s  (%d rows)" % (fname, len(rows)))

    pr("="*72)
    pr("  OUTPUT FILES:")
    wcsv("seraphim_gwtc3_diag_per_cut.csv",  cut_rows)
    wcsv("seraphim_gwtc3_diag_removed.csv",  removed_rows)

    with open("seraphim_gwtc3_diag_print.txt","w") as f:
        f.write("\n".join(lines))
    pr("  Written: seraphim_gwtc3_diag_print.txt")
    pr()
    pr("  Done. Read the numbers.")
    pr("="*72)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="GWTC-3 scatter diagnostic — progressive cuts on gamma_data")
    ap.add_argument("--dir","-d", default=".",
        help="Root directory containing HDF5 files")
    args = ap.parse_args()
    run(args.dir)
