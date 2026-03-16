"""
seraphim_gap_structure_v3.py
==============================
Straight-from-source gap test. No CSV intermediaries.

Walks the directory you run it from (and ALL subdirectories)
and processes every .h5 / .hdf5 file it finds.

Two independent octave depth measurements per event:

  n_seraphim  = log2(nu_P / sqrt(K0 / (sqrt_j * n_star)))
                where n_star = E_rad / (M_total * alpha)
                energy loss from posterior samples directly

  n_carrier   = log2(nu_P / f_isco)
                where f_isco = c^3 / (6*sqrt(6)*pi*G*M_total)

Gap = n_carrier - n_seraphim

Then correlates gap scatter against:
  - redshift       (cosmic expansion hypothesis)
  - chi_eff        (spin)
  - luminosity distance
  - mass ratio
  - total mass

CATALOG DETECTION:
  Inferred from file path — whichever of 6513631, 8177023,
  16053484 appears in the path, or GWTC2p1/GWTC3/GWTC4 strings.

FINAL MASS STRATEGY (per catalog):
  Priority 1: final_mass_source_non_evolved   (GWTC-3, GWTC-4)
  Priority 2: final_mass_source               (some GWTC-3)
  Priority 3: final_mass_non_evolved          (fallback)
  Priority 4: final_mass                      (fallback)
  Priority 5: M_total - radiated_energy_non_evolved  (reconstruct)
  Priority 6: M_total - radiated_energy              (reconstruct)
  Priority 7: mass_1_source + mass_2_source - E_NR_fit  (GWTC-2.1 fallback)
              where E_NR_fit = 0.0842 * chirp_mass_source (Healy+2014)

Each event row records which strategy was used so you can audit.

USAGE:
    cd /path/to/your/hdf5/folder
    python seraphim_gap_structure_v3.py

    Or point explicitly:
    python seraphim_gap_structure_v3.py --dir /path/to/folder

Outputs (written to wherever you run it from):
    seraphim_gap_v3_results.csv     <- one row per event
    seraphim_gap_v3_summary.csv     <- per-catalog statistics
"""

import os, sys, glob, math, re, csv, argparse
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("ERROR: h5py not installed.  pip install h5py")

try:
    from scipy.stats import spearmanr, pearsonr, ttest_1samp, kruskal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — install it for full correlation analysis")
    print("         pip install scipy")
    print()

# ── Seraphim constants (CODATA, verified) ─────────────────────────────────────
G         = 6.67430e-11          # m^3 kg^-1 s^-2
C_LIGHT   = 2.99792458e8         # m/s
M_SUN     = 1.98847e30           # kg
ALPHA     = 0.007297352569       # fine structure constant
ALPHA_INV = 1.0 / ALPHA          # 137.0360
K0        = 1.1467e84            # Hz^2  (LQG area spectrum + Robertson)
NU_P      = 1.8549e43            # Hz    (Planck frequency)
SQRT_J    = math.sqrt(0.75)      # sqrt(j(j+1)) for j=1/2
N_BBH     = math.log2(NU_P / math.sqrt(K0 / (SQRT_J * 6.09)))

# NR energy-loss fit coefficient (Healy et al. 2014) used as GWTC-2.1 fallback
# E_rad ≈ E_NR_COEFF * chirp_mass_source  (solar masses)
E_NR_COEFF = 0.0842

print("=" * 68)
print("  SERAPHIM GAP STRUCTURE TEST  v3  —  Straight from source")
print("=" * 68)
print()
print("  Constants:")
print("    n_BBH   = %.6f" % N_BBH)
print("    1/alpha = %.6f" % ALPHA_INV)
print()

# ── Waveform preference order (BBH-first) ────────────────────────────────────
PREFER_WF = [
    "IMRPhenomXPHM", "IMRPhenomXP", "IMRPhenomPv2",
    "SEOBNRv4PHM",   "SEOBNRv4P",   "NRSur7dq4",
    "IMRPhenomD",    "SEOBNRv4",
]

# Waveforms that indicate non-BBH (skip these)
NON_BBH_WF = ["NRTidal", "NRTidalv2", "NSBH", "BNS"]

def is_non_bbh(wf_key):
    ku = (wf_key or "").upper()
    return any(x.upper() in ku for x in NON_BBH_WF)

# ── HDF5 navigation ───────────────────────────────────────────────────────────
def get_posteriors(hf):
    """
    Navigate HDF5 structure to find the best posterior_samples dataset.
    Returns (dataset, waveform_key_string).
    Handles three layouts seen across GWTC catalogs:
      Layout A: hf["posterior_samples"]          (root level)
      Layout B: hf[wf_key]["posterior_samples"]  (standard GWTC)
      Layout C: hf[wf_key] is dataset directly
    Prefers BBH waveforms over NRTidal/NSBH waveforms.
    """
    keys = list(hf.keys())

    # Layout A
    if "posterior_samples" in keys:
        ps = hf["posterior_samples"]
        if hasattr(ps, "dtype") and ps.dtype is not None:
            return ps, "posterior_samples"

    # Find best waveform key
    wf_candidates = []
    for k in keys:
        if any(x in k for x in ["IMR","SEOBNR","NR","EOB"]):
            wf_candidates.append(k)

    # Sort: prefer preferred waveforms, deprioritise non-BBH
    def wf_score(k):
        ku = k.upper()
        for i, p in enumerate(PREFER_WF):
            if p.upper() in ku:
                penalty = 100 if is_non_bbh(k) else 0
                return i + penalty
        return 999 + (100 if is_non_bbh(k) else 0)

    wf_candidates.sort(key=wf_score)

    for wf in wf_candidates:
        grp = hf[wf]
        # Layout B
        if hasattr(grp, "keys") and "posterior_samples" in grp:
            ps = grp["posterior_samples"]
            if hasattr(ps, "dtype") and ps.dtype is not None:
                return ps, wf
        # Layout C
        if hasattr(grp, "dtype") and grp.dtype is not None:
            return grp, wf
        # Layout B variant: subgroup contains dataset
        if hasattr(grp, "keys"):
            for sub in grp:
                item = grp[sub]
                if hasattr(item, "dtype") and item.dtype is not None:
                    return item, wf

    return None, None

def col_arr(ps, *names):
    """Return first available column as float array, or None."""
    available = ps.dtype.names or []
    for n in names:
        if n in available:
            try:
                a = np.array(ps[n][:], dtype=float)
                a = a[np.isfinite(a)]
                if len(a) > 0:
                    return a
            except Exception:
                pass
    return None

def med_col(ps, *names):
    """Median of first available column, or None."""
    a = col_arr(ps, *names)
    return float(np.median(a)) if a is not None else None

# ── Catalog detection ─────────────────────────────────────────────────────────
def detect_catalog(fpath):
    p = fpath.replace("\\", "/")
    if "6513631"  in p or "gwtc2p1" in p.lower() or "gwtc-2" in p.lower():
        return "GWTC-2.1"
    if "8177023"  in p or "gwtc3"   in p.lower() or "gwtc-3" in p.lower():
        return "GWTC-3"
    if "16053484" in p or "gwtc4"   in p.lower() or "gwtc-4" in p.lower():
        return "GWTC-4"
    # Fallback: check filename patterns
    b = os.path.basename(fpath).upper()
    if "GWTC2" in b or "GWTC2P1" in b: return "GWTC-2.1"
    if "GWTC3" in b:                    return "GWTC-3"
    if "GWTC4" in b:                    return "GWTC-4"
    return "UNKNOWN"

def event_name(fpath):
    m = re.search(r"(GW\d{6}(?:_\d{6})?)", os.path.basename(fpath))
    return m.group(1) if m else os.path.basename(fpath)[:24]

# ── Physics calculations ──────────────────────────────────────────────────────
def f_isco_hz(M_total_msun):
    """ISCO frequency from total source-frame mass in solar masses."""
    return C_LIGHT**3 / (6.0 * math.sqrt(6.0) * math.pi * G
                         * M_total_msun * M_SUN)

def n_from_freq(f_hz):
    """Octave depth from frequency."""
    if f_hz and f_hz > 0:
        return math.log2(NU_P / f_hz)
    return None

def seraphim_n_from_erad(E_rad_arr, M_total_arr):
    """
    Compute per-sample Seraphim n from energy loss and total mass.
    n_star = E_rad / (M_total * alpha)
    nu_s   = sqrt(K0 / (sqrt_j * n_star))
    n_s    = log2(nu_P / nu_s)
    Returns (median_n, std_n, n_samples_used).
    """
    # Both arrays same length — compute per-sample
    E   = E_rad_arr
    M   = M_total_arr

    # Align lengths
    L = min(len(E), len(M))
    E, M = E[:L], M[:L]

    valid = (E > 0) & (M > 0) & (E < M)
    if valid.sum() < 10:
        return None, None, 0

    n_star = E[valid] / (M[valid] * ALPHA)
    nu_s   = np.sqrt(K0 / (SQRT_J * n_star))
    n_s    = np.log2(NU_P / nu_s)

    # Restrict to physically reasonable window
    ok = np.isfinite(n_s) & (n_s > 2.0) & (n_s < 10.0)
    if ok.sum() < 10:
        return None, None, 0

    return float(np.median(n_s[ok])), float(np.std(n_s[ok])), int(ok.sum())

# ── Per-file processing ───────────────────────────────────────────────────────
def process_file(fpath):
    """
    Returns dict of results or (None, reason_string).
    """
    try:
        with h5py.File(fpath, "r") as hf:
            ps, wf = get_posteriors(hf)
            if ps is None:
                return None, "no posterior dataset found"

            # Skip non-BBH waveforms
            if is_non_bbh(wf):
                return None, "non-BBH waveform: %s" % wf

            avail = set(ps.dtype.names or [])

            # ── Total mass (source frame) ─────────────────────────────────
            M_total_arr = col_arr(ps, "total_mass_source", "total_mass")
            if M_total_arr is None or len(M_total_arr) < 10:
                return None, "no total_mass column"
            M_total_med = float(np.median(M_total_arr))

            # ── Component masses (for fallback) ───────────────────────────
            m1_arr = col_arr(ps, "mass_1_source", "mass_1")
            m2_arr = col_arr(ps, "mass_2_source", "mass_2")
            mc_arr = col_arr(ps, "chirp_mass_source", "chirp_mass")

            # ── Radiated energy — try every known column name ─────────────
            E_rad_arr = None
            erad_strategy = ""

            # Strategy 1: final_mass_source_non_evolved -> E = M_total - M_final
            for fm_col in ["final_mass_source_non_evolved",
                            "final_mass_source",
                            "final_mass_non_evolved",
                            "final_mass"]:
                if fm_col in avail:
                    fm_arr = col_arr(ps, fm_col)
                    if fm_arr is not None:
                        L = min(len(M_total_arr), len(fm_arr))
                        e_try = M_total_arr[:L] - fm_arr[:L]
                        good  = (e_try > 0) & (e_try < M_total_arr[:L])
                        if good.sum() >= 10:
                            E_rad_arr     = e_try[good]
                            M_total_aligned = M_total_arr[:L][good]
                            erad_strategy = "M_total - %s" % fm_col
                            break

            # Strategy 2: direct radiated_energy column
            if E_rad_arr is None:
                for re_col in ["radiated_energy_non_evolved", "radiated_energy"]:
                    if re_col in avail:
                        re_arr = col_arr(ps, re_col)
                        if re_arr is not None:
                            L = min(len(M_total_arr), len(re_arr))
                            good = (re_arr[:L] > 0) & (re_arr[:L] < M_total_arr[:L])
                            if good.sum() >= 10:
                                E_rad_arr       = re_arr[:L][good]
                                M_total_aligned = M_total_arr[:L][good]
                                erad_strategy   = re_col
                                break

            # Strategy 3: GWTC-2.1 fallback — NR fit from chirp mass
            if E_rad_arr is None and mc_arr is not None:
                mc_med  = float(np.median(mc_arr))
                e_fit   = E_NR_COEFF * mc_arr
                L = min(len(M_total_arr), len(e_fit))
                good = (e_fit[:L] > 0) & (e_fit[:L] < M_total_arr[:L])
                if good.sum() >= 10:
                    E_rad_arr       = e_fit[:L][good]
                    M_total_aligned = M_total_arr[:L][good]
                    erad_strategy   = "NR_fit(0.0842*Mc)"

            # Strategy 4: component mass fallback
            if E_rad_arr is None and m1_arr is not None and m2_arr is not None:
                L   = min(len(m1_arr), len(m2_arr), len(M_total_arr))
                eta = (m1_arr[:L] * m2_arr[:L]) / M_total_arr[:L]**2
                # Phenomenological: E_rad ~ 0.1 * eta * M_total
                e_ph  = 0.1 * eta * M_total_arr[:L]
                good  = (e_ph > 0) & (e_ph < M_total_arr[:L])
                if good.sum() >= 10:
                    E_rad_arr       = e_ph[good]
                    M_total_aligned = M_total_arr[:L][good]
                    erad_strategy   = "phenom_eta(0.1*eta*M)"

            if E_rad_arr is None:
                return None, "no energy loss column and all fallbacks failed"

            # ── Compute Seraphim n ────────────────────────────────────────
            n_sph, n_sph_std, n_sph_count = seraphim_n_from_erad(
                E_rad_arr, M_total_aligned)

            if n_sph is None:
                return None, "seraphim_n computation returned no valid samples"

            # ── Compute carrier n from median total mass ──────────────────
            fc   = f_isco_hz(M_total_med)
            n_car = n_from_freq(fc)
            if n_car is None:
                return None, "invalid ISCO frequency"

            # ── Ancillary columns ─────────────────────────────────────────
            chi_eff  = med_col(ps, "chi_eff")
            redshift = med_col(ps, "redshift")
            lum_dist = med_col(ps, "luminosity_distance")
            mass_rat = med_col(ps, "mass_ratio", "inverted_mass_ratio")
            M_chirp  = med_col(ps, "chirp_mass_source", "chirp_mass")
            E_rad_med = float(np.median(E_rad_arr))

            return dict(
                wf            = wf,
                erad_strategy = erad_strategy,
                n_seraphim    = n_sph,
                n_sph_std     = n_sph_std,
                n_sph_count   = n_sph_count,
                n_carrier     = n_car,
                M_total       = M_total_med,
                M_chirp       = M_chirp,
                mass_ratio    = mass_rat,
                E_rad_med     = E_rad_med,
                chi_eff       = chi_eff,
                redshift      = redshift,
                lum_dist      = lum_dist,
            ), None

    except Exception as ex:
        return None, str(ex)

# ── File discovery — walks CWD and all subdirectories ────────────────────────
def find_hdf5_files(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".h5") or fn.lower().endswith(".hdf5"):
                found.append(os.path.join(dirpath, fn))
    return sorted(set(found))

# ── Main ──────────────────────────────────────────────────────────────────────
def run(root):
    print("  Scanning: %s" % os.path.abspath(root))
    files = find_hdf5_files(root)
    print("  Found %d HDF5 files" % len(files))
    if not files:
        sys.exit("No .h5 or .hdf5 files found under %s" % root)
    print()

    rows    = []
    skipped = []

    for fpath in files:
        ename = event_name(fpath)
        cat   = detect_catalog(fpath)
        res, reason = process_file(fpath)

        if res is None:
            skipped.append((ename, cat, reason))
            continue

        gap         = res["n_carrier"] - res["n_seraphim"]
        d_alpha     = gap - ALPHA_INV
        log2M       = math.log2(res["M_total"])

        def f4(v): return round(float(v), 4) if v is not None else ""
        def f6(v): return round(float(v), 6) if v is not None else ""

        rows.append({
            "event"         : ename,
            "catalog"       : cat,
            "waveform"      : (res["wf"] or "")[:50],
            "erad_strategy" : res["erad_strategy"],
            "M_total_Msun"  : f4(res["M_total"]),
            "M_chirp_Msun"  : f4(res["M_chirp"]),
            "mass_ratio"    : f4(res["mass_ratio"]),
            "E_rad_Msun"    : f4(res["E_rad_med"]),
            "chi_eff"       : f4(res["chi_eff"]),
            "redshift"      : f4(res["redshift"]),
            "lum_dist_Mpc"  : f4(res["lum_dist"]),
            "n_seraphim"    : f6(res["n_seraphim"]),
            "n_sph_std"     : f6(res["n_sph_std"]),
            "n_sph_samples" : res["n_sph_count"],
            "n_carrier"     : f6(res["n_carrier"]),
            "n_BBH_theory"  : round(N_BBH, 6),
            "log2_M"        : f6(log2M),
            "gap"           : f6(gap),
            "alpha_inv"     : round(ALPHA_INV, 6),
            "delta_alpha"   : f6(d_alpha),
        })

    print("  Processed: %d events" % len(rows))
    print("  Skipped:   %d files" % len(skipped))
    if skipped:
        # Show skip reason breakdown
        from collections import Counter
        reason_counts = Counter(r for _, _, r in skipped)
        for reason, cnt in reason_counts.most_common(8):
            print("    [%3d] %s" % (cnt, reason[:80]))
    print()

    if not rows:
        print("No events processed. Common causes:")
        print("  - All files are BNS/NSBH (NRTidal waveforms)")
        print("  - HDF5 structure not recognised")
        print("  - No mass columns found")
        sys.exit(1)

    # ── Write per-event CSV ───────────────────────────────────────────────
    out_results = "seraphim_gap_v3_results.csv"
    with open(out_results, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  Per-event CSV  : %s  (%d rows)" % (out_results, len(rows)))

    # ── Strategy audit ────────────────────────────────────────────────────
    from collections import Counter
    strat_counts = Counter(r["erad_strategy"] for r in rows)
    print()
    print("  Energy-loss strategy used:")
    for s, c in strat_counts.most_common():
        print("    [%3d] %s" % (c, s))
    print()

    # ── Arrays ────────────────────────────────────────────────────────────
    def arr(col):
        return np.array([float(r[col]) for r in rows
                         if r.get(col, "") not in ("", None)])

    def aligned2(c1, c2):
        pairs = [(float(r[c1]), float(r[c2])) for r in rows
                 if r.get(c1,"") not in ("",None)
                 and r.get(c2,"") not in ("",None)]
        if len(pairs) < 5: return np.array([]), np.array([])
        a, b = zip(*pairs)
        return np.array(a), np.array(b)

    def aligned3(c1, c2, c3):
        trips = [(float(r[c1]), float(r[c2]), float(r[c3])) for r in rows
                 if r.get(c1,"") not in ("",None)
                 and r.get(c2,"") not in ("",None)
                 and r.get(c3,"") not in ("",None)]
        if len(trips) < 5: return np.array([]), np.array([]), np.array([])
        a, b, c = zip(*trips)
        return np.array(a), np.array(b), np.array(c)

    gaps    = arr("gap")
    d_alpha = arr("delta_alpha")
    n_sphs  = arr("n_seraphim")
    n_cars  = arr("n_carrier")
    masses  = arr("M_total_Msun")
    log2Ms  = arr("log2_M")

    # ── Results output ────────────────────────────────────────────────────
    sep = "=" * 68
    print(sep)
    print("  GAP ANALYSIS: n_carrier  -  n_seraphim")
    print(sep)
    print()
    print("  n_seraphim  (LQG energy-loss octave depth):")
    print("    Median  %.4f   Std %.4f   Theory: %.4f" % (
          np.median(n_sphs), np.std(n_sphs), N_BBH))
    print()
    print("  n_carrier   (ISCO frequency octave depth):")
    print("    Median  %.4f   Std %.4f" % (
          np.median(n_cars), np.std(n_cars)))
    print()
    print("  GAP  =  n_carrier - n_seraphim:")
    print("    N       %d" % len(gaps))
    print("    Median  %.4f" % np.median(gaps))
    print("    Mean    %.4f" % np.mean(gaps))
    print("    Std     %.4f" % np.std(gaps))
    print("    Min     %.4f    Max  %.4f" % (np.min(gaps), np.max(gaps)))
    print()
    print("  1/alpha     =  %.4f" % ALPHA_INV)
    print("  Gap - 1/alpha:")
    print("    Median  %+.4f" % np.median(d_alpha))
    print("    Pct     %.3f%% from 1/alpha" % (
          abs(np.median(d_alpha)) / ALPHA_INV * 100))
    print()

    if HAS_SCIPY:
        t, p_t = ttest_1samp(gaps, ALPHA_INV)
        print("  t-test (gap == 1/alpha):  t = %.4f   p = %.2e" % (t, p_t))
        print()

    # ── Correlation analysis ──────────────────────────────────────────────
    print("-" * 68)
    print("  WHAT DRIVES THE SCATTER?")
    print("-" * 68)
    print()

    if HAS_SCIPY:

        def partial_r(x, y, z):
            """Pearson r(x,y) after removing linear z from both."""
            cx = np.polyfit(z, x, 1)
            cy = np.polyfit(z, y, 1)
            return pearsonr(x - np.polyval(cx, z),
                            y - np.polyval(cy, z))

        # MASS
        g_m, m_m = aligned2("gap", "M_total_Msun")
        if len(g_m) >= 5:
            rho_m, p_m = spearmanr(m_m, g_m)
            rp_m,  pp_m = pearsonr(np.log2(m_m), g_m)
            print("  Mass vs gap:")
            print("    Spearman r(M, gap)       = %+.4f   p = %.2e" % (rho_m, p_m))
            print("    Pearson  r(log2M, gap)   = %+.4f   p = %.2e" % (rp_m,  pp_m))
            print()

        # REDSHIFT — the main hypothesis
        g_z, z_z = aligned2("gap", "redshift")
        if len(g_z) >= 5:
            rho_z, p_z = spearmanr(z_z, g_z)
            rp_z,  pp_z = pearsonr(z_z,  g_z)
            sig_z = p_z < 0.05
            print("  REDSHIFT vs gap  (cosmic expansion hypothesis):")
            print("    N                        = %d" % len(g_z))
            print("    Spearman r(z, gap)       = %+.4f   p = %.2e  %s" % (
                  rho_z, p_z, "** YES **" if sig_z else "no"))
            print("    Pearson  r(z, gap)       = %+.4f   p = %.2e" % (rp_z, pp_z))
            if sig_z:
                if rho_z > 0:
                    print("    Direction: higher redshift → LARGER gap (consistent with expansion)")
                else:
                    print("    Direction: higher redshift → SMALLER gap")
            print()

            # Partial: z controlling for log2(M)
            g3, z3, m3 = aligned3("gap", "redshift", "log2_M")
            if len(g3) >= 5:
                rp_part, pp_part = partial_r(z3, g3, m3)
                print("    Partial r(z, gap | log2 M) = %+.4f   p = %.2e  %s" % (
                      rp_part, pp_part,
                      "** INDEPENDENT of mass **" if pp_part < 0.05 else "absorbed by mass"))
                print()

        # LUMINOSITY DISTANCE
        g_d, d_d = aligned2("gap", "lum_dist_Mpc")
        if len(g_d) >= 5:
            rho_d, p_d = spearmanr(d_d, g_d)
            print("  Luminosity distance vs gap:")
            print("    Spearman r(dL, gap)      = %+.4f   p = %.2e  %s" % (
                  rho_d, p_d, "** YES **" if p_d < 0.05 else "no"))
            print()

        # CHI_EFF
        g_c, c_c = aligned2("gap", "chi_eff")
        if len(g_c) >= 5:
            rho_c, p_c = spearmanr(c_c, g_c)
            print("  Chi_eff (spin) vs gap:")
            print("    Spearman r(chi, gap)     = %+.4f   p = %.2e  %s" % (
                  rho_c, p_c, "** YES **" if p_c < 0.05 else "no"))

            # Partial: chi_eff controlling for log2(M)
            g4, c4, m4 = aligned3("gap", "chi_eff", "log2_M")
            if len(g4) >= 5:
                rp_cs, pp_cs = partial_r(c4, g4, m4)
                print("    Partial r(chi, gap | log2M)= %+.4f   p = %.2e  %s" % (
                      rp_cs, pp_cs,
                      "** INDEPENDENT **" if pp_cs < 0.05 else "absorbed by mass"))
            print()

        # MASS RATIO
        g_q, q_q = aligned2("gap", "mass_ratio")
        if len(g_q) >= 5:
            rho_q, p_q = spearmanr(q_q, g_q)
            print("  Mass ratio vs gap:")
            print("    Spearman r(q, gap)       = %+.4f   p = %.2e  %s" % (
                  rho_q, p_q, "** YES **" if p_q < 0.05 else "no"))
            print()

        # REDSHIFT QUARTILE BINS
        g_z2, z_z2 = aligned2("gap", "redshift")
        if len(g_z2) >= 20:
            pcts = np.percentile(z_z2, [0, 25, 50, 75, 100])
            print("  Gap by redshift quartile:")
            groups = []
            qlabels = ["Q1 low-z ", "Q2       ", "Q3       ", "Q4 high-z"]
            for i in range(4):
                lo, hi = pcts[i], pcts[i+1]
                mask = (z_z2 >= lo) & (z_z2 <= hi) if i == 3 else \
                       (z_z2 >= lo) & (z_z2 < hi)
                gq = g_z2[mask]
                groups.append(gq)
                print("    %s z=[%.3f, %.3f]  N=%3d  "
                      "median_gap=%.4f  delta=%+.4f" % (
                      qlabels[i], lo, hi, len(gq),
                      np.median(gq) if len(gq) else 0,
                      np.median(gq) - ALPHA_INV if len(gq) else 0))
            print()
            groups_valid = [g for g in groups if len(g) >= 3]
            if len(groups_valid) >= 2:
                stat, p_kw = kruskal(*groups_valid)
                print("    Kruskal-Wallis across quartiles: p = %.4e  %s" % (
                      p_kw,
                      "** gap differs across z bins **" if p_kw < 0.05
                      else "consistent across z bins"))
                print()

    else:
        print("  (scipy required for correlations — pip install scipy)")
        print()

    # ── Per-catalog breakdown ─────────────────────────────────────────────
    print("-" * 68)
    print("  PER-CATALOG SUMMARY")
    print("-" * 68)
    sum_rows = []
    for cat in ["GWTC-2.1", "GWTC-3", "GWTC-4", "UNKNOWN"]:
        sub = [r for r in rows
               if r["catalog"] == cat and r["gap"] not in ("", None)]
        if not sub:
            continue
        gs  = np.array([float(r["gap"])      for r in sub])
        nss = np.array([float(r["n_seraphim"]) for r in sub])
        ncs = np.array([float(r["n_carrier"])  for r in sub])
        zs  = [float(r["redshift"]) for r in sub
               if r["redshift"] not in ("", None)]
        strats = Counter(r["erad_strategy"] for r in sub)
        top_strat = strats.most_common(1)[0][0]

        print("  %s:  N=%d  median_gap=%.4f  std=%.4f  "
              "delta=%+.4f  median_z=%.3f" % (
              cat, len(sub), np.median(gs), np.std(gs),
              np.median(gs) - ALPHA_INV,
              np.median(zs) if zs else -1))
        print("    n_sph: median=%.4f  n_car: median=%.4f  "
              "strategy: %s" % (np.median(nss), np.median(ncs), top_strat))
        print()

        sum_rows.append({
            "catalog"          : cat,
            "N"                : len(sub),
            "median_n_sph"     : round(np.median(nss), 4),
            "std_n_sph"        : round(np.std(nss),    4),
            "median_n_carrier" : round(np.median(ncs), 4),
            "std_n_carrier"    : round(np.std(ncs),    4),
            "median_gap"       : round(np.median(gs),  4),
            "std_gap"          : round(np.std(gs),     4),
            "alpha_inv"        : round(ALPHA_INV,      6),
            "delta_from_alpha" : round(float(np.median(gs)) - ALPHA_INV, 4),
            "pct_from_alpha"   : round(abs(float(np.median(gs)) - ALPHA_INV)
                                       / ALPHA_INV * 100, 4),
            "median_redshift"  : round(np.median(zs), 4) if zs else "",
            "top_erad_strategy": top_strat,
        })

    # ALL combined
    gs_all = gaps
    zs_all = [float(r["redshift"]) for r in rows
              if r["redshift"] not in ("","None",None)]
    sum_rows.append({
        "catalog"          : "ALL",
        "N"                : len(rows),
        "median_n_sph"     : round(float(np.median(n_sphs)), 4),
        "std_n_sph"        : round(float(np.std(n_sphs)),    4),
        "median_n_carrier" : round(float(np.median(n_cars)), 4),
        "std_n_carrier"    : round(float(np.std(n_cars)),    4),
        "median_gap"       : round(float(np.median(gs_all)), 4),
        "std_gap"          : round(float(np.std(gs_all)),    4),
        "alpha_inv"        : round(ALPHA_INV, 6),
        "delta_from_alpha" : round(float(np.median(gs_all)) - ALPHA_INV, 4),
        "pct_from_alpha"   : round(abs(float(np.median(gs_all)) - ALPHA_INV)
                                   / ALPHA_INV * 100, 4),
        "median_redshift"  : round(np.median(zs_all), 4) if zs_all else "",
        "top_erad_strategy": "mixed",
    })

    out_sum = "seraphim_gap_v3_summary.csv"
    with open(out_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
        w.writeheader()
        w.writerows(sum_rows)
    print("  Summary CSV    : %s" % out_sum)
    print()

    # ── Final verdict ─────────────────────────────────────────────────────
    pct = abs(float(np.median(d_alpha))) / ALPHA_INV * 100
    print(sep)
    print("  VERDICT")
    print(sep)
    print()
    print("  Median gap       = %.4f" % np.median(gaps))
    print("  1 / alpha        = %.4f" % ALPHA_INV)
    print("  |delta|          = %.4f  (%.3f%%)" % (
          abs(np.median(d_alpha)), pct))
    print("  Std of gap       = %.4f  octaves (real physical scatter)" % np.std(gaps))
    print()
    if pct < 1.0:
        print("  *** STRONG MATCH: gap within 1% of 1/alpha ***")
    elif pct < 5.0:
        print("  **  CANDIDATE:   gap within 5% of 1/alpha  **")
    elif pct < 10.0:
        print("  *   WEAK:        gap within 10% of 1/alpha  *")
    else:
        print("  x   NO MATCH:    gap >10% from 1/alpha")
    print()
    print("  Done.")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Seraphim gap structure test — reads HDF5 files directly")
    ap.add_argument(
        "--dir", "-d", default=".",
        help="Root directory to scan (default: current directory). "
             "All subdirectories are searched automatically.")
    args = ap.parse_args()
    run(args.dir)
