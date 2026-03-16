"""
seraphim_alpha_mass_corrected.py
=================================
Follow-up to seraphim_alpha_test.py.

The first test showed:
  - gap = n_carrier - n_BBH = 132.24 (median, 264 BBH events)
  - 1/alpha = 137.04
  - Offset = -4.80 octaves (3.5%)

The offset matches log2(M_observed / M_reference) exactly.
This test asks: after removing the mass scaling, does the
corrected gap collapse to 1/alpha?

CORRECTED GAP = gap - log2(M_total / M_ref)
             = (n_carrier - n_BBH) - log2(M_total / M_ref)

If the relationship is:  gap = 1/alpha + log2(M / M_ref)
then corrected gap should cluster at exactly 1/alpha = 137.036

We test three reference masses:
  M_ref_1 = 1692 Msun  (exact 1/alpha intercept from previous test)
  M_ref_2 =   30 Msun  (conventional reference)
  M_ref_3 = fitted      (least-squares best fit M_ref)

Also tests:
  - Does corrected gap std drop vs raw gap std?
  - Does mass correlation vanish after correction?
  - Is the slope of log2(M) vs gap exactly 1.0?

USAGE:
    python seraphim_alpha_mass_corrected.py --dir /path/to/hdf5/files

Outputs:
    seraphim_alpha_corrected_results.csv
    seraphim_alpha_corrected_summary.csv
"""

import os, sys, glob, math, re, argparse, csv
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("ERROR: h5py not installed. Run: pip install h5py")

try:
    from scipy.stats import spearmanr, pearsonr, ttest_1samp
    from scipy.optimize import minimize_scalar
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Some tests will be skipped.")

# ── Constants ─────────────────────────────────────────────────────────────────
G        = 6.67430e-11
C        = 2.99792458e8
M_SUN    = 1.98847e30
ALPHA    = 0.007297
ALPHA_INV= 1.0 / ALPHA       # 137.036
K0       = 1.1467e84
NU_P     = 1.8549e43
SQRT_J   = math.sqrt(0.75)   # sqrt(j(j+1)), j=0.5
N_BBH    = math.log2(NU_P / math.sqrt(K0 / (SQRT_J * 6.09)))

M_REF_EXACT = 1692.0   # Msun — from previous test (exact 1/alpha intercept)
M_REF_CONV  =   30.0   # Msun — conventional reference

print("Constants:")
print("  n_BBH   = %.6f" % N_BBH)
print("  1/alpha = %.6f" % ALPHA_INV)
print("  M_ref (exact intercept) = %.1f Msun" % M_REF_EXACT)
print()

# ── HDF5 navigation (identical to seraphim_alpha_test.py) ────────────────────
PREFER = ["IMRPhenomXPHM", "IMRPhenomPv2", "SEOBNRv4PHM",
          "IMRPhenomD",    "SEOBNRv4P",    "NRSur7dq4"]

def get_posteriors(hf):
    keys = list(hf.keys())
    if "posterior_samples" in keys:
        return hf["posterior_samples"], "posterior_samples"
    wf = None
    for p in PREFER:
        for k in keys:
            if p in k:
                wf = k; break
        if wf: break
    if wf is None:
        for k in keys:
            if any(x in k for x in ["IMR","SEOBNR","NR"]):
                wf = k; break
    if wf is None and keys:
        wf = keys[0]
    if wf is None:
        return None, None
    grp = hf[wf]
    if "posterior_samples" in grp:
        return grp["posterior_samples"], wf
    if hasattr(grp, "dtype") and grp.dtype is not None:
        return grp, wf
    for name in grp:
        item = grp[name]
        if hasattr(item, "dtype") and item.dtype is not None:
            return item, wf
    return None, wf

def col(ps, name):
    try:
        if name in (ps.dtype.names or []):
            return np.array(ps[name][:], dtype=float)
    except Exception:
        pass
    return None

def med(ps, *names):
    for n in names:
        a = col(ps, n)
        if a is not None:
            a = a[np.isfinite(a)]
            if len(a): return float(np.median(a))
    return None

def classify(wf):
    k = (wf or "").upper()
    if "NSBH" in k: return "NSBH"
    if "NRTIDAL" in k: return "BNS_NSBH"
    return "BBH"

def catalog_of(fname):
    b = os.path.basename(fname).upper()
    if "GWTC2P1" in b: return "GWTC-2.1"
    if "GWTC3"   in b: return "GWTC-3"
    if "GWTC4"   in b: return "GWTC-4"
    return "UNKNOWN"

def event_of(fname):
    m = re.search(r"(GW\d{6}(?:_\d{6})?)", os.path.basename(fname))
    return m.group(1) if m else os.path.basename(fname)[:20]

def f_isco(M_msun):
    return C**3 / (6.0 * math.sqrt(6.0) * math.pi * G * M_msun * M_SUN)

def n_from_f(f_hz):
    return math.log2(NU_P / f_hz) if (f_hz and f_hz > 0) else None

def process(fpath):
    try:
        with h5py.File(fpath, "r") as hf:
            ps, wf = get_posteriors(hf)
            if ps is None:
                return None, "no dataset found"
            M_total = med(ps, "total_mass_source", "total_mass")
            M_chirp = med(ps, "chirp_mass_source", "chirp_mass")
            chi_eff = med(ps, "chi_eff")
            z       = med(ps, "redshift")
            return dict(wf=wf, etype=classify(wf),
                        M_total=M_total, M_chirp=M_chirp,
                        chi_eff=chi_eff, z=z), None
    except Exception as e:
        return None, str(e)

# ── Main ──────────────────────────────────────────────────────────────────────
def run(search_dir):
    files = sorted(set(
        glob.glob(os.path.join(search_dir, "**", "*.h5"),   recursive=True) +
        glob.glob(os.path.join(search_dir, "**", "*.hdf5"), recursive=True) +
        glob.glob(os.path.join(search_dir, "*.h5")) +
        glob.glob(os.path.join(search_dir, "*.hdf5"))
    ))
    if not files:
        sys.exit("ERROR: no HDF5 files found under " + search_dir)

    print("Found %d HDF5 files" % len(files))
    print()

    rows, skipped = [], 0
    for fpath in files:
        res, err = process(fpath)
        ename = event_of(fpath)
        cat   = catalog_of(fpath)
        if res is None or not res["M_total"] or res["M_total"] <= 0:
            skipped += 1; continue
        if res["etype"] != "BBH":
            skipped += 1; continue

        M  = res["M_total"]
        fc = f_isco(M)
        nc = n_from_f(fc)
        if nc is None:
            skipped += 1; continue

        gap = nc - N_BBH

        # Three corrections
        corr_exact = gap - math.log2(M / M_REF_EXACT)
        corr_conv  = gap - math.log2(M / M_REF_CONV)

        def f4(v): return round(v, 4) if v is not None else ""
        def f6(v): return round(v, 6) if v is not None else ""

        rows.append({
            "event":           ename,
            "catalog":         cat,
            "M_total_Msun":    round(M, 3),
            "M_chirp_Msun":    round(res["M_chirp"], 3) if res["M_chirp"] else "",
            "chi_eff":         f4(res["chi_eff"]),
            "redshift":        f4(res["z"]),
            "f_isco_hz":       f4(fc),
            "n_carrier":       f6(nc),
            "n_BBH":           round(N_BBH, 6),
            "gap_raw":         f6(gap),
            "log2_M_over_1692":f6(math.log2(M / M_REF_EXACT)),
            "log2_M_over_30":  f6(math.log2(M / M_REF_CONV)),
            "gap_corr_1692":   f6(corr_exact),
            "gap_corr_30":     f6(corr_conv),
            "alpha_inv":       round(ALPHA_INV, 6),
            "delta_raw":       f6(gap - ALPHA_INV),
            "delta_corr_1692": f6(corr_exact - ALPHA_INV),
            "delta_corr_30":   f6(corr_conv  - ALPHA_INV),
        })

    print("BBH events: %d  |  Skipped: %d" % (len(rows), skipped))
    print()

    if not rows:
        sys.exit("No BBH events processed.")

    out = "seraphim_alpha_corrected_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Per-event CSV: " + out)
    print()

    # ── Arrays ────────────────────────────────────────────────────────────
    gaps     = np.array([float(r["gap_raw"])       for r in rows])
    corr1692 = np.array([float(r["gap_corr_1692"]) for r in rows])
    corr30   = np.array([float(r["gap_corr_30"])   for r in rows])
    masses   = np.array([float(r["M_total_Msun"])  for r in rows])
    ncs      = np.array([float(r["n_carrier"])      for r in rows])

    # ── Find best-fit M_ref ───────────────────────────────────────────────
    # Want corrected gap = 1/alpha for all events
    # gap - log2(M/M_ref) = 1/alpha
    # log2(M_ref) = gap - 1/alpha + log2(M)
    # Least-squares: minimize variance of corrected gaps
    log2M = np.log2(masses)

    def variance_of_corrected(log2_Mref):
        corrected = gaps - (log2M - log2_Mref)
        return np.var(corrected)

    if HAS_SCIPY:
        result = minimize_scalar(variance_of_corrected,
                                 bounds=(0, 20), method='bounded')
        log2_Mref_best = result.x
        M_ref_best = 2**log2_Mref_best
        corr_best = gaps - (log2M - log2_Mref_best)
    else:
        # Manual grid search
        best_var = 1e99
        log2_Mref_best = 10.0
        for lm in np.linspace(0, 20, 10000):
            v = variance_of_corrected(lm)
            if v < best_var:
                best_var = v
                log2_Mref_best = lm
        M_ref_best = 2**log2_Mref_best
        corr_best = gaps - (log2M - log2_Mref_best)

    # ── Measure slope of gap vs log2(M) ──────────────────────────────────
    # If gap = 1/alpha + log2(M/M_ref) then slope should = 1.0 exactly
    if HAS_SCIPY:
        slope, intercept = np.polyfit(log2M, gaps, 1)
        r_slope, p_slope = pearsonr(log2M, gaps)
    else:
        slope = np.cov(log2M, gaps)[0,1] / np.var(log2M)
        intercept = np.mean(gaps) - slope * np.mean(log2M)
        r_slope, p_slope = None, None

    # ── Print results ─────────────────────────────────────────────────────
    print("=" * 65)
    print("  MASS-CORRECTED ALPHA GAP TEST")
    print("=" * 65)
    print()
    print("  Raw gap stats:")
    print("    Median %.4f   Std %.4f   (1/alpha = %.4f)" % (
          np.median(gaps), np.std(gaps), ALPHA_INV))
    print("    Delta from 1/alpha: %+.4f (%.3f%%)" % (
          np.median(gaps)-ALPHA_INV,
          abs(np.median(gaps)-ALPHA_INV)/ALPHA_INV*100))
    print()

    print("  SLOPE TEST (gap vs log2 M):")
    print("    Fitted slope = %.6f" % slope)
    print("    Expected     = 1.000000  (pure mass scaling)")
    print("    Deviation    = %+.6f" % (slope - 1.0))
    if r_slope is not None:
        print("    Pearson r    = %.4f   p = %.2e" % (r_slope, p_slope))
    print()

    print("  BEST-FIT M_reference:")
    print("    M_ref        = %.2f Msun" % M_ref_best)
    print("    log2(M_ref)  = %.4f" % log2_Mref_best)
    print("    (M_ref_exact from prev test: %.1f Msun)" % M_REF_EXACT)
    print()

    # Corrected gap statistics for each M_ref
    for label, corr, mref in [
        ("M_ref = 1692 Msun (exact intercept)", corr1692, M_REF_EXACT),
        ("M_ref = 30 Msun  (conventional)",     corr30,   M_REF_CONV),
        ("M_ref = %.1f Msun (best fit)" % M_ref_best, corr_best, M_ref_best),
    ]:
        med_c = np.median(corr)
        std_c = np.std(corr)
        dlt_c = med_c - ALPHA_INV
        pct_c = abs(dlt_c) / ALPHA_INV * 100
        std_reduction = (1.0 - std_c / np.std(gaps)) * 100

        print("  Corrected gap (%s):" % label)
        print("    Median %.4f   Std %.4f   (raw std was %.4f)" % (
              med_c, std_c, np.std(gaps)))
        print("    Delta from 1/alpha: %+.4f (%.4f%%)" % (dlt_c, pct_c))
        print("    Std reduction vs raw: %.1f%%" % std_reduction)

        if HAS_SCIPY:
            t, p_t = ttest_1samp(corr, ALPHA_INV)
            print("    t-test vs 1/alpha: t=%.4f  p=%.4e" % (t, p_t))
            r_mc, p_mc = pearsonr(np.log10(masses), corr)
            print("    Residual mass corr: r=%.4f  p=%.4e" % (r_mc, p_mc))
            print("    (want r~0 after correction)")
        print()

    # Per-catalog corrected stats
    print("  Per-catalog corrected gap (M_ref = %.1f Msun):" % M_ref_best)
    for cat in ["GWTC-2.1", "GWTC-3", "GWTC-4"]:
        idx = [i for i,r in enumerate(rows) if r["catalog"] == cat]
        if idx:
            cg = corr_best[idx]
            print("    %s: N=%3d  median=%.4f  std=%.4f  delta=%+.4f" % (
                  cat, len(idx), np.median(cg), np.std(cg),
                  np.median(cg)-ALPHA_INV))
    print()

    # The key question: after mass correction, what's left?
    best_med   = np.median(corr_best)
    best_delta = best_med - ALPHA_INV
    best_pct   = abs(best_delta) / ALPHA_INV * 100

    print("=" * 65)
    print("  VERDICT")
    print("=" * 65)
    print()
    print("  Raw gap:           %.4f  (%.3f%% from 1/alpha)" % (
          np.median(gaps),
          abs(np.median(gaps)-ALPHA_INV)/ALPHA_INV*100))
    print("  Best corrected:    %.4f  (%.4f%% from 1/alpha)" % (
          best_med, best_pct))
    print()
    print("  Slope of gap vs log2(M): %.4f  (ideal = 1.0000)" % slope)
    print("  Best-fit M_ref:          %.2f Msun" % M_ref_best)
    print()

    if best_pct < 0.1:
        print("  *** EXACT COLLAPSE: corrected gap = 1/alpha to <0.1% ***")
        print("  The relationship is:  gap = 1/alpha + log2(M / %.1f Msun)" % M_ref_best)
    elif best_pct < 1.0:
        print("  ** STRONG COLLAPSE: corrected gap within 1% of 1/alpha **")
        print("  Residual may be spin, redshift, or waveform systematics")
    elif best_pct < 5.0:
        print("  * PARTIAL COLLAPSE: mass explains most of offset *")
        print("  Residual structure remains — investigate chi_eff, redshift")
    else:
        print("  Mass correction insufficient — other physics in the residual")
    print()

    # Summary CSV
    sum_rows = []
    for cat in ["GWTC-2.1", "GWTC-3", "GWTC-4", "ALL"]:
        if cat == "ALL":
            idx = list(range(len(rows)))
        else:
            idx = [i for i,r in enumerate(rows) if r["catalog"] == cat]
        if not idx: continue
        g  = gaps[idx]; c1 = corr1692[idx]; cb = corr_best[idx]
        sum_rows.append({
            "catalog":               cat,
            "N":                     len(idx),
            "median_gap_raw":        round(np.median(g),  4),
            "std_gap_raw":           round(np.std(g),     4),
            "delta_raw":             round(np.median(g)-ALPHA_INV, 4),
            "pct_raw":               round(abs(np.median(g)-ALPHA_INV)/ALPHA_INV*100, 4),
            "slope_gap_vs_log2M":    round(slope, 6),
            "M_ref_best_fit_Msun":   round(M_ref_best, 2),
            "median_gap_corr_best":  round(np.median(cb), 4),
            "std_gap_corr_best":     round(np.std(cb),    4),
            "delta_corr_best":       round(np.median(cb)-ALPHA_INV, 4),
            "pct_corr_best":         round(abs(np.median(cb)-ALPHA_INV)/ALPHA_INV*100, 4),
            "alpha_inv":             round(ALPHA_INV, 6),
        })

    sc = "seraphim_alpha_corrected_summary.csv"
    with open(sc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
        w.writeheader(); w.writerows(sum_rows)
    print("Summary CSV: " + sc)
    print()
    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", "-d", default=".",
                    help="Directory with GWTC HDF5 files")
    run(ap.parse_args().dir)
