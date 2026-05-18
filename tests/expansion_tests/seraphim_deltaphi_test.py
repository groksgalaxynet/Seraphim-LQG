#!/usr/bin/env python3
"""
seraphim_deltaphi_test.py  (multi-catalog edition v2)
======================================================
Seraphim LQG Framework — Delta_phi Phase Burst Test

Runs across all three GWTC catalogs in one pass:
    6513631   GWTC-2.1  (C01:XPHM waveform keys -> posterior_samples subgroup;
                          no final_mass column — uses NR fit fallback)
    8177023   GWTC-3    (C01:IMRPhenomXPHM -> posterior_samples subgroup;
                          has final_mass_source_non_evolved)
    16053484  GWTC-4    (hash-prefixed or C01+XPHM keys; Layout B or C;
                          no Tidal/NSBH waveform keys)

WHAT IS TESTED:
    The Phase Mismatch Theorem (Spin Foam Addendum v5) predicts that at the
    moment orbital compactness crosses C_threshold = 0.2563, a single phase
    burst of Delta_phi = 0.59918 rad fires (settlement between gamma_area
    and gamma_entropy Immirzi channels).

    If this burst is absorbed by GR template fitting, M_chirp shifts by:
        delta_Mc/Mc = -(3/5) * Delta_phi / Phi_total(f_start -> f_threshold)

    Per-event tests:
      1. n_seraphim prediction check (pred = 5.314 for BBH)
      2. f_threshold distribution across mass range
      3. Mc_skew vs bias_frac correlation (key Delta_phi signal test)
      4. QNM spacing prediction (gamma_area vs gamma_entropy channels)
      5. Threshold-mass table for LIGO sensitive band
      6. Events in 100-200 Hz band (most detectable Delta_phi window)
      7. Out-of-BBH-band events

USAGE:
    # Point at parent folder containing all three catalog subfolders:
    python seraphim_deltaphi_test.py --dir /path/to/Downloads

    # Or list catalogs explicitly:
    python seraphim_deltaphi_test.py \
        --dirs /path/6513631 /path/8177023 /path/16053484

    # Or run from inside a folder (walks subdirs automatically):
    cd /path/to/Downloads && python seraphim_deltaphi_test.py

OUTPUTS:
    seraphim_deltaphi_results.csv
    seraphim_deltaphi_summary.txt

REQUIREMENTS:
    pip install h5py numpy scipy
"""

import os, sys, math, csv, argparse, re
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("ERROR: h5py not installed.  pip install h5py")

try:
    from scipy.stats import pearsonr, spearmanr, ttest_1samp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — install for full stats.  pip install scipy")

# ── Seraphim constants ────────────────────────────────────────────────────────
G             = 6.67430e-11
C_LIGHT       = 2.99792458e8
M_SUN         = 1.98847e30
ALPHA         = 0.007297352569
K0            = 1.1467e84        # Hz^2  CRITICAL: exponent is 84
NU_PLANCK     = 1.8549e43        # Hz
SQRT_J        = math.sqrt(0.75)  # sqrt(j(j+1)) for j=1/2
GAMMA_AREA    = 0.2375
GAMMA_ENTROPY = math.log(2) / (math.pi * math.sqrt(3))   # 0.12738
DELTA_PHI     = math.log(2) * (GAMMA_AREA / GAMMA_ENTROPY - 1)  # 0.59918 rad
C_THRESHOLD   = 0.2563
N_BBH         = 5.314
E_NR_COEFF    = 0.0842           # Healy+2014: E_rad ~ 0.0842 * Mc

print("Seraphim Delta_phi Phase Burst Test  (multi-catalog edition v2)")
print(f"  Delta_phi     = {DELTA_PHI:.5f} rad")
print(f"  C_threshold   = {C_THRESHOLD}")
print(f"  gamma_area    = {GAMMA_AREA}")
print(f"  gamma_entropy = {GAMMA_ENTROPY:.5f}")
print(f"  K0            = 1.1467e84 Hz^2")
print(f"  N_BBH pred    = {N_BBH}")
print()

# ── Physics ───────────────────────────────────────────────────────────────────

def f_threshold(M_total_solar):
    """GW frequency when orbital compactness = C_threshold."""
    M     = M_total_solar * M_SUN
    r_g   = G * M / C_LIGHT**2
    r_thr = r_g / C_THRESHOLD
    f_orb = (1.0 / (2.0 * math.pi)) * math.sqrt(G * M / r_thr**3)
    return 2.0 * f_orb

def predicted_Mc_bias(M_total_solar, M_chirp_solar, f_start=20.0):
    """
    Fractional M_chirp shift from absorbing Delta_phi burst.
    Returns (bias_frac, f_thr, Phi_total_rad).
    """
    f_thr = f_threshold(M_total_solar)
    if f_thr <= f_start:
        return 0.0, f_thr, 0.0
    Mc    = M_chirp_solar * M_SUN
    coeff = (5.0 / (32.0 * math.pi)) * (C_LIGHT**3 / (G * Mc))**(5.0/3.0) * math.pi**(-5.0/3.0)
    Phi   = coeff * (3.0/5.0) * (f_start**(-5.0/3.0) - f_thr**(-5.0/3.0)) * 2.0 * math.pi
    bias  = -(3.0/5.0) * DELTA_PHI / Phi if Phi > 0 else 0.0
    return bias, f_thr, Phi

def qnm_spacing(M_rem_solar):
    """QNM overtone spacing under each Immirzi channel."""
    M          = M_rem_solar * M_SUN
    df_area    = C_LIGHT**3 / (8.0 * math.pi**2 * G * M) * math.log(2) / GAMMA_AREA
    df_entropy = C_LIGHT**3 / (8.0 * math.pi**2 * G * M) * math.log(2) / GAMMA_ENTROPY
    return df_area, df_entropy

# ── Catalog / event helpers ───────────────────────────────────────────────────

def detect_catalog(fpath):
    p  = fpath.replace("\\", "/")
    pl = p.lower()
    if "6513631"  in p  or "gwtc2p1" in pl or "gwtc-2" in pl or "gwtc2.1" in pl:
        return "GWTC-2.1"
    if "8177023"  in p  or "gwtc3"   in pl or "gwtc-3" in pl:
        return "GWTC-3"
    if "16053484" in p  or "gwtc4"   in pl or "gwtc-4" in pl:
        return "GWTC-4"
    b = os.path.basename(fpath).upper()
    if "GWTC2" in b or "GWTC2P1" in b: return "GWTC-2.1"
    if "GWTC3" in b:                    return "GWTC-3"
    if "GWTC4" in b:                    return "GWTC-4"
    return "UNKNOWN"

def clean_event_name(fpath):
    base = os.path.basename(fpath)
    for pre in ["IGWN-GWTC2p1-v2-", "IGWN-GWTC3p0-v2-", "IGWN-GWTC4p0-"]:
        if base.startswith(pre):
            base = base[len(pre):]
    base = re.sub(r'^[0-9a-f]{8,}_\d+-', '', base, flags=re.IGNORECASE)
    for suf in ["_PEDataRelease_mixed_cosmo.h5", "_PEDataRelease_mixed_nocosmo.h5",
                "_PEDataRelease_mixed.h5", "_combined_PEDataRelease.hdf5",
                "_PEDataRelease.h5", "_PEDataRelease.hdf5", ".h5", ".hdf5"]:
        if base.endswith(suf):
            base = base[:-len(suf)]
    for suf in ["-combined_PEDataRelease", "-PEDataRelease"]:
        if suf in base:
            base = base[:base.index(suf)]
    base = base.replace("_cosmo", "").replace("_nocosmo", "")
    m = re.search(r"(GW\d{6}(?:_\d{6})?)", base)
    if m:
        return m.group(1)
    return base

def dedup_key(name):
    return name.replace("_cosmo", "").replace("_nocosmo", "").strip()

# ── Column helpers ────────────────────────────────────────────────────────────

MASS1_COLS    = ["mass_1_source", "mass_1", "m1_source", "m1"]
MASS2_COLS    = ["mass_2_source", "mass_2", "m2_source", "m2"]
MFINAL_COLS   = ["final_mass_source_non_evolved", "final_mass_source",
                 "final_mass_non_evolved", "final_mass",
                 "remnant_mass_source", "remnant_mass"]
MCHIRP_COLS   = ["chirp_mass_source", "chirp_mass", "Mc_source", "Mc"]
ERAD_COLS     = ["radiated_energy_non_evolved", "radiated_energy"]
LOGLIKE_COLS  = ["log_likelihood", "logL", "log_like"]
CHIEFF_COLS   = ["chi_eff", "chi_effective"]
REDSHIFT_COLS = ["redshift", "z"]
LUMDIST_COLS  = ["luminosity_distance", "lum_dist"]
TOTMASS_COLS  = ["total_mass_source", "total_mass"]

def col_arr(ps, *names):
    """Return first available column as clean finite float array, or None."""
    try:
        avail = ps.dtype.names or []
    except Exception:
        return None
    for n in names:
        if n in avail:
            try:
                a = np.array(ps[n][:], dtype=float)
                a = a[np.isfinite(a)]
                if len(a) > 0:
                    return a
            except Exception:
                pass
    return None

def med_col(ps, *names):
    a = col_arr(ps, *names)
    return float(np.median(a)) if a is not None else None

# ── HDF5 posterior extraction ─────────────────────────────────────────────────
#
# GWTC-2.1 (6513631):
#   hf["C01:IMRPhenomXPHM"]["posterior_samples"]
#   No final_mass column; E_rad via NR fit on Mc.
#
# GWTC-3 (8177023):
#   hf["C01:IMRPhenomXPHM"]["posterior_samples"]
#   Has final_mass_source_non_evolved.
#
# GWTC-4 (16053484):
#   Keys are hash-prefixed like "a3f8b291_1-GW200311..." OR bare "C01+XPHM"
#   or just "IMRPhenomXPHM". Layout B (-> posterior_samples) or C (direct dataset).
#   The hash prefix does NOT contain "IMR"/"SEOBNR" — must scan ALL non-meta keys.

PREFER_WF  = ["IMRPhenomXPHM", "IMRPhenomXP",  "SEOBNRv4PHM",
              "SEOBNRv4P",     "NRSur7dq4",    "IMRPhenomPv2",
              "IMRPhenomD",    "SEOBNRv4",      "XPHM"]
NON_BBH_WF = ["NRTidal", "NRTidalv2", "NSBH", "BNS"]
META_KEYS  = {"history", "version", "meta", "labels", "config",
              "strain", "psd", "asd", "noise"}

def is_non_bbh(key):
    ku = (key or "").upper()
    return any(x.upper() in ku for x in NON_BBH_WF)

def wf_score(k):
    ku      = k.upper()
    penalty = 1000 if is_non_bbh(k) else 0
    for i, p in enumerate(PREFER_WF):
        if p.upper() in ku:
            return i + penalty
    return 500 + penalty   # unknown waveform key: try it but deprioritise

def _event_type(ps):
    try:
        cols = ps.dtype.names or []
    except Exception:
        cols = []
    if any("lambda" in c.lower() or "tidal" in c.lower() for c in cols):
        return "BNS/NSBH"
    return "BBH"

def get_posteriors(f):
    """
    Returns (posterior_dataset, waveform_key_string, event_type_string).

    Strategy:
      1. Root-level posterior_samples (Layout A)
      2. All non-metadata top-level keys sorted by wf_score (covers GWTC-4
         hash keys that don't contain model name strings)
         a. key -> posterior_samples subgroup  (Layout B)
         b. key is dataset directly            (Layout C)
         c. key -> any immediate child dataset (Layout B-variant)
    """
    keys = list(f.keys())

    # Layout A: root-level posterior_samples
    if "posterior_samples" in keys:
        ps = f["posterior_samples"]
        if hasattr(ps, "dtype") and ps.dtype is not None:
            return ps, "posterior_samples", _event_type(ps)

    # Gather all non-metadata candidates — MUST include hash-prefixed GWTC-4 keys
    candidates = [k for k in keys if k.lower() not in META_KEYS]
    candidates.sort(key=wf_score)

    for wf in candidates:
        try:
            node = f[wf]
        except Exception:
            continue

        # Layout B: wf_key -> posterior_samples subgroup
        try:
            if hasattr(node, "keys") and "posterior_samples" in node:
                ps = node["posterior_samples"]
                if hasattr(ps, "dtype") and ps.dtype is not None:
                    return ps, wf, _event_type(ps)
        except Exception:
            pass

        # Layout C: wf_key IS the dataset directly
        if hasattr(node, "dtype") and node.dtype is not None:
            return node, wf, _event_type(node)

        # Layout B-variant: wf_key -> any immediate child that is a dataset
        try:
            if hasattr(node, "keys"):
                for sub in list(node.keys()):
                    try:
                        item = node[sub]
                        if hasattr(item, "dtype") and item.dtype is not None:
                            return item, f"{wf}/{sub}", _event_type(item)
                    except Exception:
                        pass
        except Exception:
            pass

    return None, None, "UNKNOWN"

# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(fpath):
    try:
        with h5py.File(fpath, "r") as f:
            ps, wf_key, etype = get_posteriors(f)
            if ps is None:
                return None, "no posteriors found"

            avail = set(ps.dtype.names or [])

            # ── Component masses ──────────────────────────────────────────
            m1_arr = col_arr(ps, *MASS1_COLS)
            m2_arr = col_arr(ps, *MASS2_COLS)
            if m1_arr is None or m2_arr is None:
                return None, "no component mass columns"

            L = min(len(m1_arr), len(m2_arr))
            m1_arr = m1_arr[:L]
            m2_arr = m2_arr[:L]

            # ── Total mass ────────────────────────────────────────────────
            mt_arr = col_arr(ps, *TOTMASS_COLS)
            if mt_arr is not None:
                L      = min(L, len(mt_arr))
                mt_arr = mt_arr[:L]
                m1_arr = m1_arr[:L]
                m2_arr = m2_arr[:L]
            else:
                mt_arr = m1_arr + m2_arr

            # ── Chirp mass ────────────────────────────────────────────────
            mc_col = col_arr(ps, *MCHIRP_COLS)
            if mc_col is not None:
                Lm     = min(len(mt_arr), len(mc_col))
                mc_use = mc_col[:Lm]
                mt_use = mt_arr[:Lm]
            else:
                Lm     = len(mt_arr)
                mc_use = (m1_arr[:Lm] * m2_arr[:Lm])**0.6 / mt_arr[:Lm]**0.2
                mt_use = mt_arr[:Lm]

            # ── Radiated energy (four strategies) ─────────────────────────
            E_rad_arr       = None
            M_total_aligned = None
            mc_aligned      = None
            erad_strategy   = ""

            # Strategy 1: final_mass column -> E = M_total - M_final
            for fm_col in MFINAL_COLS:
                if fm_col in avail:
                    fm_arr = col_arr(ps, fm_col)
                    if fm_arr is not None:
                        Lf    = min(len(mt_use), len(fm_arr))
                        e_try = mt_use[:Lf] - fm_arr[:Lf]
                        good  = (e_try > 0) & (e_try < mt_use[:Lf])
                        if good.sum() >= 50:
                            E_rad_arr       = e_try[good]
                            M_total_aligned = mt_use[:Lf][good]
                            _mc             = mc_use[:min(Lf, len(mc_use))]
                            mc_aligned      = _mc[good[:len(_mc)]]
                            erad_strategy   = f"M_total-{fm_col}"
                            break

            # Strategy 2: explicit radiated_energy column
            if E_rad_arr is None:
                for re_col in ERAD_COLS:
                    if re_col in avail:
                        re_arr = col_arr(ps, re_col)
                        if re_arr is not None:
                            Lr   = min(len(mt_use), len(re_arr))
                            good = (re_arr[:Lr] > 0) & (re_arr[:Lr] < mt_use[:Lr])
                            if good.sum() >= 50:
                                E_rad_arr       = re_arr[:Lr][good]
                                M_total_aligned = mt_use[:Lr][good]
                                _mc             = mc_use[:min(Lr, len(mc_use))]
                                mc_aligned      = _mc[good[:len(_mc)]]
                                erad_strategy   = re_col
                                break

            # Strategy 3: NR fit E_rad ~ 0.0842 * Mc  (primary GWTC-2.1 path)
            if E_rad_arr is None:
                Ln    = len(mt_use)
                e_fit = E_NR_COEFF * mc_use[:Ln]
                good  = (e_fit > 0) & (e_fit < mt_use[:Ln])
                if good.sum() >= 50:
                    E_rad_arr       = e_fit[good]
                    M_total_aligned = mt_use[:Ln][good]
                    mc_aligned      = mc_use[:Ln][good]
                    erad_strategy   = "NR_fit(0.0842*Mc)"

            # Strategy 4: phenomenological eta fit (last resort)
            if E_rad_arr is None:
                Lp   = min(len(m1_arr), len(m2_arr), len(mt_arr))
                eta  = (m1_arr[:Lp] * m2_arr[:Lp]) / mt_arr[:Lp]**2
                e_ph = 0.1 * eta * mt_arr[:Lp]
                good = (e_ph > 0) & (e_ph < mt_arr[:Lp])
                if good.sum() >= 50:
                    E_rad_arr       = e_ph[good]
                    M_total_aligned = mt_arr[:Lp][good]
                    _mc             = mc_use[:min(Lp, len(mc_use))]
                    mc_aligned      = _mc[good[:len(_mc)]]
                    erad_strategy   = "phenom_eta"

            if E_rad_arr is None:
                return None, "all E_rad strategies failed (< 50 valid samples)"

            # ── Seraphim n ────────────────────────────────────────────────
            N_star = E_rad_arr / (M_total_aligned * ALPHA)
            nu2    = K0 / (SQRT_J * N_star)
            n_arr  = np.log2(NU_PLANCK / np.sqrt(nu2))
            ok     = np.isfinite(n_arr) & (n_arr > 2.0) & (n_arr < 10.0)
            if ok.sum() < 50:
                return None, f"only {ok.sum()} valid n samples"

            n_arr  = n_arr[ok]
            med_n  = float(np.median(n_arr))
            std_n  = float(np.std(n_arr))
            in_bbh = int(abs(med_n - N_BBH) < 0.5)

            # ── Posterior mass statistics ─────────────────────────────────
            mt_ok    = M_total_aligned[ok]
            _mc_full = mc_aligned if len(mc_aligned) >= ok.sum() else mc_aligned
            mc_ok    = _mc_full[ok[:len(_mc_full)]] if len(_mc_full) == len(ok) else _mc_full[:ok.sum()]
            med_Mtot = float(np.median(mt_ok))
            med_Mc   = float(np.median(mc_ok))
            std_Mc   = float(np.std(mc_ok))
            mean_Mc  = float(np.mean(mc_ok))
            Mc_skew  = mean_Mc - med_Mc

            # ── Final mass (for QNM spacing) ──────────────────────────────
            fm_med_arr = col_arr(ps, *MFINAL_COLS)
            med_Mfin   = float(np.median(fm_med_arr)) if fm_med_arr is not None \
                         else med_Mtot - float(np.median(E_rad_arr))
            med_Erad   = float(np.median(E_rad_arr))

            # ── Threshold frequency and predicted bias ────────────────────
            f_thr                 = f_threshold(med_Mtot)
            bias_frac, _, Phi_tot = predicted_Mc_bias(med_Mtot, med_Mc)
            bias_Msun             = bias_frac * med_Mc
            in_ligo               = int(20.0 <= f_thr <= 300.0)

            # ── QNM spacing ───────────────────────────────────────────────
            df_area, df_entropy = qnm_spacing(med_Mfin)

            # ── Log-likelihood stats ──────────────────────────────────────
            ll_col = col_arr(ps, *LOGLIKE_COLS)
            if ll_col is not None and len(ll_col) > 50:
                med_ll   = float(np.median(ll_col))
                std_ll   = float(np.std(ll_col))
                ll_range = float(np.max(ll_col) - np.min(ll_col))
            else:
                med_ll = std_ll = ll_range = float('nan')

            # ── Ancillary medians ─────────────────────────────────────────
            chi_eff  = med_col(ps, *CHIEFF_COLS)
            redshift = med_col(ps, *REDSHIFT_COLS)
            lum_dist = med_col(ps, *LUMDIST_COLS)

            def _r(v, d=4):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return ""
                return round(float(v), d)

            return {
                "med_Mtot":   _r(med_Mtot),
                "med_Mc":     _r(med_Mc),
                "mean_Mc":    _r(mean_Mc),
                "std_Mc":     _r(std_Mc),
                "Mc_skew":    _r(Mc_skew, 6),
                "med_Mfin":   _r(med_Mfin),
                "med_Erad":   _r(med_Erad),
                "med_n":      _r(med_n, 5),
                "std_n":      _r(std_n, 5),
                "in_bbh":     in_bbh,
                "f_thr":      _r(f_thr, 2),
                "in_ligo":    in_ligo,
                "Phi_tot":    _r(Phi_tot, 4),
                "bias_frac":  _r(bias_frac, 6),
                "bias_Msun":  _r(bias_Msun, 4),
                "med_ll":     _r(med_ll, 3),
                "std_ll":     _r(std_ll, 3),
                "ll_range":   _r(ll_range, 3),
                "df_area_hz": _r(df_area, 2),
                "df_entr_hz": _r(df_entropy, 2),
                "chi_eff":    _r(chi_eff, 5),
                "redshift":   _r(redshift, 5),
                "lum_dist":   _r(lum_dist, 2),
                "n_samples":  int(ok.sum()),
                "waveform":   (wf_key or "")[:60],
                "erad_strat": erad_strategy,
                "event_type": etype,
            }, None

    except Exception as ex:
        import traceback
        tb_last = traceback.format_exc().splitlines()[-1]
        return None, f"error: {ex}  [{tb_last}]"

# ── File discovery ────────────────────────────────────────────────────────────

SKIP_PATTERNS = ["strain", "gwosc_4k", "timeseries", "summary",
                 "pesummarytable", "noise", "psd", "asd"]

def is_pe_file(fpath):
    bn = os.path.basename(fpath).lower()
    return not any(p in bn for p in SKIP_PATTERNS)

def find_hdf5(root):
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".h5", ".hdf5")):
                full = os.path.join(dirpath, fn)
                if is_pe_file(full):
                    found.append(full)
    return sorted(set(found))

# ── Array helpers  (NOTE: use 'is None' not truthiness check on pool) ─────────
# BUG FIX: "pool or rows" is wrong — an empty list is falsy and falls back to
# all rows. Use "rows if pool is None else pool" instead.

def _flt(r, col):
    try:    return float(r.get(col, ""))
    except: return None

def _arr(col, pool, default_pool):
    # pool=None means use default_pool; pool=[] means genuinely empty
    src = default_pool if pool is None else pool
    vals = [_flt(r, col) for r in src]
    return np.array([v for v in vals if v is not None and not math.isnan(v)])

def _aligned(c1, c2, pool):
    pairs = [(_flt(r,c1), _flt(r,c2)) for r in pool
             if _flt(r,c1) is not None and _flt(r,c2) is not None
             and not math.isnan(_flt(r,c1)) and not math.isnan(_flt(r,c2))]
    if len(pairs) < 3:
        return np.array([]), np.array([])
    a, b = zip(*pairs)
    return np.array(a), np.array(b)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seraphim Delta_phi test — all three GWTC catalogs")
    parser.add_argument("--dir",  default=None,
        help="Single root directory to walk recursively")
    parser.add_argument("--dirs", nargs="+", default=None,
        help="Explicit dirs: --dirs ./6513631 ./8177023 ./16053484")
    parser.add_argument("--out",  default="seraphim_deltaphi_results.csv")
    parser.add_argument("--debug-hdf5", action="store_true",
        help="Print HDF5 key structure for first file in each catalog (diagnosis)")
    args = parser.parse_args()

    if args.dirs:
        roots = [os.path.abspath(d) for d in args.dirs]
    elif args.dir:
        roots = [os.path.abspath(args.dir)]
    else:
        roots = [os.path.abspath(".")]

    all_files = []
    for root in roots:
        files = find_hdf5(root)
        print(f"  {root}  ->  {len(files)} HDF5 PE files")
        all_files.extend(files)
    all_files = sorted(set(all_files))
    print(f"\nTotal HDF5 files to process: {len(all_files)}\n")

    if not all_files:
        sys.exit("No HDF5 PE files found. Check --dir / --dirs paths.")

    # Optional: dump key structure of first file per catalog for diagnosis
    if args.debug_hdf5:
        seen_cats = set()
        for fp in all_files:
            cat = detect_catalog(fp)
            if cat not in seen_cats:
                seen_cats.add(cat)
                print(f"  DEBUG {cat}: {fp}")
                try:
                    with h5py.File(fp, "r") as f:
                        def _walk(name, obj, depth=0):
                            indent = "    " * depth
                            dtype_str = str(obj.dtype) if hasattr(obj,"dtype") else "group"
                            print(f"  {indent}{name}  [{dtype_str}]")
                        f.visititems(lambda n,o: _walk(n, o,
                            depth=n.count("/")))
                except Exception as ex:
                    print(f"  Could not inspect: {ex}")
                print()
        print()

    rows        = []
    skipped     = []
    seen_events = {}

    for fpath in all_files:
        ename = clean_event_name(fpath)
        cat   = detect_catalog(fpath)
        ekey  = dedup_key(ename)

        if ekey in seen_events:
            skipped.append((ename, cat, f"duplicate of {seen_events[ekey]}"))
            continue
        seen_events[ekey] = ename

        result, reason = process_file(fpath)
        if result is None:
            skipped.append((ename, cat, reason))
            print(f"  [SKIP] {ename:<42} {reason}")
            continue

        row = {"event": ename, "catalog": cat, **result}
        rows.append(row)

        flag      = "[BBH]" if result["in_bbh"] else "[OUT]"
        ligo_flag = "[INBAND]" if result["in_ligo"] else "        "
        strat     = result.get("erad_strat", "")[:20]
        print(f"  {flag} {ligo_flag} {ename:<36} "
              f"n={result['med_n']:.4f}  M={result['med_Mtot']:.1f}  "
              f"f_thr={result['f_thr']:.0f}Hz  "
              f"bias={float(result['bias_frac'] or 0)*100:+.3f}%  "
              f"[{cat}|{strat}]")

    print(f"\nProcessed: {len(rows)} events   Skipped: {len(skipped)}\n")

    if not rows:
        sys.exit("No events processed. Try running with --debug-hdf5 to inspect file structure.")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {args.out}\n")

    # ── Subsets ────────────────────────────────────────────────────────────────
    sep       = "=" * 70
    bbh_rows  = [r for r in rows if r["in_bbh"] == 1]
    ligo_rows = [r for r in rows if r["in_bbh"] == 1 and r["in_ligo"] == 1]
    out_rows  = [r for r in rows if r["in_bbh"] == 0]

    all_n = _arr("med_n", None, rows)
    bbh_n = _arr("med_n", bbh_rows, rows)

    # ── PART 1: N_Seraphim prediction check ───────────────────────────────────
    print(sep)
    print("PART 1: N_SERAPHIM PREDICTION CHECK  (pred = 5.314)")
    print(sep)
    print()
    if len(all_n):
        sem = np.std(all_n) / len(all_n)**0.5
        print(f"All events ({len(rows)}):  "
              f"mean={np.mean(all_n):.5f}  std={np.std(all_n):.5f}  "
              f"sigma_from_5.314={(np.mean(all_n)-N_BBH)/sem:+.2f}σ")
    if len(bbh_n):
        sem = np.std(bbh_n) / len(bbh_n)**0.5
        print(f"BBH only   ({len(bbh_rows)}):  "
              f"mean={np.mean(bbh_n):.5f}  std={np.std(bbh_n):.5f}  "
              f"sigma_from_5.314={(np.mean(bbh_n)-N_BBH)/sem:+.2f}σ")
    for cat in ["GWTC-2.1", "GWTC-3", "GWTC-4"]:
        sub = [r for r in bbh_rows if r["catalog"] == cat]
        # NOTE: pass sub explicitly; do NOT use "or" fallback on empty list
        ns = _arr("med_n", sub, rows)
        if len(ns) == 0:
            print(f"  {cat}: no BBH events processed")
            continue
        sem = np.std(ns) / len(ns)**0.5
        print(f"  {cat} BBH ({len(sub)}):  "
              f"mean={np.mean(ns):.5f}  std={np.std(ns):.5f}  "
              f"sigma={(np.mean(ns)-N_BBH)/sem:+.2f}σ")
    print()

    # ── PART 2: Threshold frequency distribution ───────────────────────────────
    print(sep)
    print("PART 2: THRESHOLD FREQUENCY DISTRIBUTION  (BBH only)")
    print(sep)
    print()
    bands = [("<20",None,20),("20-50",20,50),("50-100",50,100),
             ("100-150",100,150),("150-200",150,200),("200-300",200,300),(">300",300,None)]
    print(f"  {'Band(Hz)':>10}  {'N':>4}  {'mean_n':>8}  {'std':>6}  "
          f"{'sigma_from_5.314':>18}  {'mean_Mc_skew':>14}")
    print("  " + "-"*72)
    for label, lo, hi in bands:
        sub = [r for r in bbh_rows
               if (lo is None or (_flt(r,"f_thr") or 0) >= lo)
               and (hi is None or (_flt(r,"f_thr") or 9999) < hi)]
        ns  = _arr("med_n",   sub, rows)
        sks = _arr("Mc_skew", sub, rows)
        if len(ns) == 0: continue
        sem = np.std(ns)/len(ns)**0.5 if len(ns) > 1 else 1.0
        print(f"  {label:>10}  {len(sub):>4}  {np.mean(ns):>8.5f}  "
              f"{np.std(ns):>6.4f}  {(np.mean(ns)-N_BBH)/sem:>+18.3f}σ  "
              f"{(np.mean(sks) if len(sks) else float('nan')):>+14.6f}")
    print()

    # ── PART 3: Delta_phi bias correlation tests ───────────────────────────────
    print(sep)
    print("PART 3: DELTA_PHI BIAS CORRELATION TESTS")
    print(sep)
    print()
    print(f"  Delta_phi = {DELTA_PHI:.5f} rad  (C_threshold = {C_THRESHOLD})")
    print(f"  Prediction: negative bias_frac; Mc_skew correlated positively with |bias_frac|")
    print()

    bf_arr = _arr("bias_frac", bbh_rows, rows)
    sk_arr = _arr("Mc_skew",   bbh_rows, rows)
    bm_arr = _arr("bias_Msun", bbh_rows, rows)
    if len(bf_arr):
        print(f"  Mean bias_frac:  {np.mean(bf_arr)*100:+.5f}%  "
              f"(range {np.min(bf_arr)*100:+.4f}% to {np.max(bf_arr)*100:+.4f}%)")
    if len(bm_arr):
        print(f"  Mean bias_Msun:  {np.mean(bm_arr):+.5f} Msun")
    if len(sk_arr):
        print(f"  Mean Mc_skew:    {np.mean(sk_arr):+.6f} Msun  "
              f"(frac positive: {sum(s>0 for s in sk_arr)/len(sk_arr):.3f})")
    print()

    if HAS_SCIPY:
        sk2, bf2 = _aligned("Mc_skew", "bias_frac", bbh_rows)
        if len(sk2) > 5:
            r_sb, p_sb   = pearsonr(sk2, bf2)
            sr_sb, sp_sb = spearmanr(sk2, bf2)
            print(f"  r(Mc_skew, bias_frac):   {r_sb:+.4f}  p={p_sb:.3e}  "
                  f"[Spearman: {sr_sb:+.4f}  p={sp_sb:.3e}]")
            interp = ("POSITIVE → bias absorbed as predicted" if r_sb > 0.1 else
                      "NEGATIVE → opposite direction" if r_sb < -0.1 else
                      "NEAR ZERO → no correlation")
            print(f"    Interpretation: {interp}")

        n2, ft2 = _aligned("med_n", "f_thr", bbh_rows)
        if len(n2) > 5:
            r_nf, p_nf = pearsonr(n2, ft2)
            print(f"  r(n, f_threshold):       {r_nf:+.4f}  p={p_nf:.3e}")

        n2, mt2 = _aligned("med_n", "med_Mtot", bbh_rows)
        if len(n2) > 5:
            r_nM, p_nM = pearsonr(n2, mt2)
            print(f"  r(n, M_total):           {r_nM:+.4f}  p={p_nM:.3e}")

        sk2, ft2 = _aligned("Mc_skew", "f_thr", bbh_rows)
        if len(sk2) > 5:
            r_sf, p_sf = pearsonr(sk2, ft2)
            print(f"  r(Mc_skew, f_threshold): {r_sf:+.4f}  p={p_sf:.3e}")

        if len(sk_arr) > 5:
            t_stat, t_p = ttest_1samp(sk_arr, 0)
            print(f"  t-test Mc_skew != 0:     t={t_stat:+.3f}  p={t_p:.3e}  "
                  f"mean={np.mean(sk_arr):+.6f}")
    else:
        def _pearson(a, b):
            am,bm = np.mean(a),np.mean(b)
            return np.sum((a-am)*(b-bm)) / (np.std(a)*np.std(b)*len(a))
        sk2, bf2 = _aligned("Mc_skew", "bias_frac", bbh_rows)
        n2,  ft2 = _aligned("med_n",   "f_thr",     bbh_rows)
        n2b, mt2 = _aligned("med_n",   "med_Mtot",  bbh_rows)
        if len(sk2)>3: print(f"  r(Mc_skew, bias_frac): {_pearson(sk2,bf2):+.4f}")
        if len(n2) >3: print(f"  r(n, f_threshold):     {_pearson(n2,ft2):+.4f}")
        if len(n2b)>3: print(f"  r(n, M_total):         {_pearson(n2b,mt2):+.4f}")
    print()

    # ── PART 4: QNM ringdown spacing ──────────────────────────────────────────
    print(sep)
    print("PART 4: QNM RINGDOWN SPACING PREDICTION  (lightest 12 BBH remnants)")
    print(sep)
    print()
    print(f"  gamma_entropy/gamma_area = {GAMMA_ENTROPY/GAMMA_AREA:.6f}")
    print(f"  {'Event':<30}  {'M_rem':>7}  {'df_area(Hz)':>12}  "
          f"{'df_entr(Hz)':>12}  {'diff(Hz)':>10}")
    print("  " + "-"*80)
    for r in sorted(bbh_rows, key=lambda r: _flt(r,"med_Mfin") or 9999)[:12]:
        da = _flt(r,"df_area_hz"); de = _flt(r,"df_entr_hz"); mf = _flt(r,"med_Mfin")
        if da is None: continue
        print(f"  {r['event']:<30}  {mf:>7.1f}  {da:>12.2f}  "
              f"{de:>12.2f}  {de-da:>+10.2f}")
    print()

    # ── PART 5: Threshold mass table ───────────────────────────────────────────
    print(sep)
    print("PART 5: THRESHOLD MASS FOR LIGO SENSITIVE BAND")
    print(sep)
    print()
    for f_t in [100, 115, 130, 150]:
        M_thr = (C_THRESHOLD**1.5 * C_LIGHT**3) / (math.pi * G * f_t * M_SUN)
        print(f"  f_threshold = {f_t} Hz  =>  M_total = {M_thr:.1f} M_sun")
    print()
    print(f"  GW150914: M=65.3  =>  f_thr = {f_threshold(65.3):.0f} Hz")
    print(f"  GW151226: M=21.7  =>  f_thr = {f_threshold(21.7):.0f} Hz")
    print()

    # ── PART 6: Events in 100-200 Hz band ─────────────────────────────────────
    print(sep)
    print("PART 6: EVENTS WITH f_threshold IN 100-200 Hz  (Delta_phi most detectable)")
    print(sep)
    print()
    target = sorted([r for r in bbh_rows
                     if 100 <= (_flt(r,"f_thr") or 0) <= 200],
                    key=lambda r: abs((_flt(r,"f_thr") or 0) - 130))
    print(f"  {'Event':<32}  {'M_tot':>7}  {'f_thr':>8}  {'bias%':>10}  "
          f"{'Mc_skew':>10}  {'cat':>8}")
    print("  " + "-"*82)
    for r in target[:25]:
        print(f"  {r['event']:<32}  {_flt(r,'med_Mtot'):>7.1f}  "
              f"{_flt(r,'f_thr'):>8.1f}  "
              f"{(_flt(r,'bias_frac') or 0)*100:>+9.3f}%  "
              f"{_flt(r,'Mc_skew') or 0:>+10.6f}  "
              f"{r['catalog']:>8}")
    print()

    # ── PART 7: Out-of-band events ─────────────────────────────────────────────
    print(sep)
    print("PART 7: OUT-OF-BBH-BAND EVENTS  (n not near 5.314)")
    print(sep)
    print()
    for r in sorted(out_rows, key=lambda r: _flt(r,"med_n") or 0):
        print(f"  {r['event']:<35}  n={_flt(r,'med_n'):.4f}  "
              f"M={_flt(r,'med_Mtot'):.1f}  "
              f"type={r['event_type']}  cat={r['catalog']}")
    print()

    # ── Write summary ──────────────────────────────────────────────────────────
    with open("seraphim_deltaphi_summary.txt", "w") as f:
        f.write("SERAPHIM DELTA_PHI PHASE BURST TEST SUMMARY\n")
        f.write("=" * 52 + "\n")
        f.write(f"Events processed:    {len(rows)}\n")
        f.write(f"BBH events:          {len(bbh_rows)}\n")
        f.write(f"In LIGO band (BBH):  {len(ligo_rows)}\n")
        f.write(f"Non-BBH band:        {len(out_rows)}\n")
        f.write(f"Delta_phi:           {DELTA_PHI:.5f} rad\n")
        f.write(f"C_threshold:         {C_THRESHOLD}\n")
        f.write(f"K0:                  1.1467e84 Hz^2\n")
        if len(bbh_n):
            sem = np.std(bbh_n) / len(bbh_n)**0.5
            f.write(f"BBH mean n:          {np.mean(bbh_n):.5f}\n")
            f.write(f"BBH sigma from pred: {(np.mean(bbh_n)-N_BBH)/sem:+.3f}sigma\n")
        for cat in ["GWTC-2.1", "GWTC-3", "GWTC-4"]:
            sub = [r for r in bbh_rows if r["catalog"] == cat]
            # Explicit sub passed — no falsy-pool fallback bug
            ns = _arr("med_n", sub, rows)
            if len(ns) == 0:
                f.write(f"{cat} BBH mean n:  NO EVENTS PROCESSED\n")
            else:
                f.write(f"{cat} BBH mean n:  {np.mean(ns):.5f}  (N={len(sub)})\n")

    print(f"Summary written: seraphim_deltaphi_summary.txt")
    print()
    print("DONE.")

if __name__ == "__main__":
    main()
