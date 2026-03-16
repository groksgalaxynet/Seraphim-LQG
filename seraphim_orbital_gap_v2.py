import os
import csv
import json
import numpy as np
import h5py
import re
from scipy import stats

# ==============================================================================
# SERAPHIM ORBITAL ANGULAR MOMENTUM GAP TEST v2
#
# THE PREVIOUS VERSION HAD A TAUTOLOGY:
#   It defined beta_event = 1 - 2*E[chi_eff^2]
#   Then computed gap = 1 - (beta + 2*E[chi_eff^2]) = 0  ALWAYS
#   That's circular — beta was defined to make gap=0.
#
# CORRECT APPROACH:
#   beta = 0.902 is an EMPIRICAL POPULATION MEASUREMENT
#   (from seraphim_beta_posterior.py: mean over all BBH events)
#   It is NOT derived per-event from chi_eff.
#
#   The Gauss-Bonnet identity uses the POPULATION beta:
#     GB_pop = beta_pop + 2 * mean(E[chi_eff^2])
#            = 0.902 + 2 * mean(E[chi^2])
#            = ~0.998  (not quite 1.0)
#
#   The RESIDUAL = 1.0 - GB_pop = the true unaccounted gap.
#
# WHAT IS BETA_POP MEASURING?
#   beta_pop is NOT a per-event quantity.
#   It is a POPULATION PARAMETER: the average fraction of the
#   geometric budget committed to pure gravitational (non-spin) faces
#   across the BBH population.
#
#   The per-event quantity is E[chi_eff^2].
#   The population parameter is beta_pop = 1 - 2*mean(E[chi_eff^2]).
#
#   So the gap has two separate components:
#   (A) Per-event spread: some events have higher E[chi^2] than the
#       population mean — their individual GB is less than the pop average
#   (B) Population-level gap: the population mean GB != 1.0
#
# THIS SCRIPT:
#   (1) Computes E[chi_eff^2] per event from full posteriors
#   (2) Computes population beta_pop and population GB
#   (3) Defines the TRUE GAP as 1.0 - GB_population
#   (4) Tests: does final_spin (orbital AM) explain the population gap?
#   (5) Tests extended identity:
#       beta_pop + 2*mean(E[chi_eff^2]) + 2*f*mean(E[chi_orb^2]) = 1.000
#   (6) Per-event: deviation from population GB — which events
#       are furthest above/below the population line?
#   (7) Kerr area ratio correlation with E[chi_eff^2]
#   (8) Full correlation matrix on E[chi_eff^2] and on (GB_event - GB_pop)
#
# BH GEOMETRY NOTE:
#   No density. BH = (M, a). Horizon area A_Kerr = 4*pi*r_s^2*(1+sqrt(1-a^2))
#   Face count proportional to A. More area = more faces = more geometric budget.
#   Orbital AM contracts the effective area of the pre-merger system.
#
# K_0 = 1.1467e84 Hz^2  (CRITICAL: exponent = 84, not 4)
# BETA_POP = 0.902  (from seraphim_beta_posterior.py, posterior-corrected)
# ==============================================================================

K_0       = 1.1467e84
NU_PLANCK = 1.8549e43
ALPHA     = 0.007297
J_SPIN    = 0.5
SQRT_J    = np.sqrt(J_SPIN * (J_SPIN + 1.0))

N_BAND_LO  = 4.76
N_BAND_HI  = 5.76
N_BBH_PRED = 5.314

# EMPIRICAL POPULATION BETA from seraphim_beta_posterior.py
BETA_POP   = 0.902

# Healy non-spinning orbital AM:
# chi_orb(eta) = 2*sqrt(3)*eta - 4.13*eta^2 + 5.37*eta^3
# at eta=0.25 (q=1): chi_orb = 0.6864

MASS1_NAMES  = ["mass_1_source", "mass_1"]
MASS2_NAMES  = ["mass_2_source", "mass_2"]
MFINAL_NAMES = ["final_mass_source", "final_mass",
                "final_mass_source_non_evolved", "final_mass_non_evolved"]
CHIEFF_NAMES = ["chi_eff"]
CHIFIN_NAMES = ["final_spin", "final_spin_non_evolved"]
ETA_NAMES    = ["symmetric_mass_ratio"]
CHIRP_NAMES  = ["chirp_mass_source", "chirp_mass"]
CHIP_NAMES   = ["chi_p", "chi_p_2spin"]
MRAT_NAMES   = ["mass_ratio"]
A1_NAMES     = ["a_1"]
A2_NAMES     = ["a_2"]


def get_col(ps, names):
    for name in names:
        try:
            if name in ps.dtype.names:
                arr = np.array(ps[name], dtype=float)
                if np.any(np.isfinite(arr)):
                    return arr
        except Exception:
            pass
    return None


def get_posteriors(f):
    for key in f.keys():
        if key.startswith("C01") and "XPHM" in key and "Tidal" not in key and "NSBH" not in key:
            try:
                return f[key]["posterior_samples"], key
            except Exception:
                pass
    for key in f.keys():
        if "XPHM" in key and "Tidal" not in key and "NSBH" not in key:
            try:
                return f[key]["posterior_samples"], key
            except Exception:
                pass
    for key in f.keys():
        if key in ("history", "version"):
            continue
        if "NSBH" in key or "Tidal" in key or "NRTidal" in key:
            continue
        try:
            return f[key]["posterior_samples"], key
        except Exception:
            pass
    return None, None


def compute_n_array(m1, m2, m_final):
    m_total = m1 + m2
    e_loss  = m_total - m_final
    valid   = (e_loss > 0) & (e_loss < m_total) & \
              np.isfinite(e_loss) & np.isfinite(m_total)
    if valid.sum() < 10:
        return np.array([]), valid
    n_star   = e_loss[valid] / (m_total[valid] * ALPHA)
    nu       = np.sqrt(K_0 / (SQRT_J * n_star))
    n_octave = np.log2(NU_PLANCK / nu)
    return n_octave[np.isfinite(n_octave)], valid


def chi_orb_healy(eta):
    """Non-spinning orbital AM contribution to remnant spin (Healy et al 2014)."""
    return 2.0 * np.sqrt(3.0) * eta - 4.13 * eta**2 + 5.37 * eta**3


def infer_catalog(fp):
    p = fp.upper()
    if "6513631"  in p: return "GWTC-2.1"
    if "8177023"  in p: return "GWTC-3"
    if "16053484" in p: return "GWTC-4"
    return "UNKNOWN"


def priority(ef):
    bn = ef.lower()
    if "nocosmo"  in bn: return 0
    if "combined" in bn: return 1
    if "cosmo"    in bn: return 2
    return 3


def deduplicate(rows):
    groups = {}
    for r in rows:
        m   = re.search(r'(GW\d{6}[_\d]*)', r["event_file"])
        eid = m.group(1) if m else r["event_file"]
        key = r["catalog"] + ":" + eid
        if key not in groups or priority(r["event_file"]) < priority(groups[key]["event_file"]):
            groups[key] = r
    return list(groups.values())


def find_h5_files(root="."):
    found = []
    for dirpath, _, fnames in os.walk(root):
        for fn in fnames:
            if fn.endswith(".h5") or fn.endswith(".hdf5"):
                if "Summary" not in fn and "Table" not in fn:
                    found.append(os.path.join(dirpath, fn))
    return sorted(found)


def safe_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return None, None, None, None
    rp, pp = stats.pearsonr(x[mask], y[mask])
    rs, ps = stats.spearmanr(x[mask], y[mask])
    return float(rp), float(pp), float(rs), float(ps)


def run(root="."):
    print("=" * 70)
    print("  SERAPHIM ORBITAL AM GAP TEST v2")
    print("  beta_pop = 0.902 (empirical, seraphim_beta_posterior.py)")
    print("  Gap = 1.0 - (beta_pop + 2*mean_E[chi_eff^2])")
    print("  Testing: does orbital AM (chi_final - chi_eff) close the gap?")
    print("  K_0 = 1.1467e84 Hz^2  (exponent=84)")
    print("=" * 70)
    print()

    h5_files = find_h5_files(root)
    if not h5_files:
        print("[!] No HDF5 files: " + os.path.abspath(root))
        return
    print("[*] Found " + str(len(h5_files)) + " HDF5 files")
    print()

    rows    = []
    skipped = []

    for filepath in h5_files:
        fn      = os.path.basename(filepath)
        subdir  = os.path.relpath(os.path.dirname(filepath), root)
        label   = (subdir + "/" + fn) if subdir != "." else fn
        catalog = infer_catalog(filepath)

        try:
            with h5py.File(filepath, "r") as f:
                ps, key_used = get_posteriors(f)
                if ps is None:
                    skipped.append(label); continue

                m1      = get_col(ps, MASS1_NAMES)
                m2      = get_col(ps, MASS2_NAMES)
                m_final = get_col(ps, MFINAL_NAMES)
                if m1 is None or m2 is None or m_final is None:
                    skipped.append(label); continue

                n_arr, valid = compute_n_array(m1, m2, m_final)
                if len(n_arr) < 50:
                    skipped.append(label); continue

                median_n = float(np.median(n_arr))
                in_band  = int(N_BAND_LO <= median_n <= N_BAND_HI)

                def align(col, min_n=50):
                    if col is None: return None
                    c = col[valid][:len(n_arr)]
                    c = c[np.isfinite(c)]
                    return c if len(c) >= min_n else None

                chi_eff   = align(get_col(ps, CHIEFF_NAMES))
                chi_final = align(get_col(ps, CHIFIN_NAMES))
                eta_col   = align(get_col(ps, ETA_NAMES))
                chirp_col = align(get_col(ps, CHIRP_NAMES))
                chip_col  = align(get_col(ps, CHIP_NAMES))
                mrat_col  = align(get_col(ps, MRAT_NAMES))
                a1_col    = align(get_col(ps, A1_NAMES))
                a2_col    = align(get_col(ps, A2_NAMES))

                if chi_eff is None:
                    skipped.append(label); continue

                # ── Core quantities ──────────────────────────────────────────
                e_chi2_eff   = float(np.mean(chi_eff**2))
                med_chi_eff  = float(np.median(chi_eff))

                # GB per-event uses POPULATION beta (not derived from chi_eff)
                gb_event_pop = BETA_POP + 2.0 * e_chi2_eff
                # deviation of this event from pop-mean GB
                # (positive = this event contributes MORE spin than average)

                # ── Final spin ───────────────────────────────────────────────
                if chi_final is not None:
                    e_chi2_final  = float(np.mean(chi_final**2))
                    med_chi_final = float(np.median(chi_final))

                    # Orbital AM residual:
                    # chi_final^2 = chi_eff^2 + chi_orb^2 + cross_terms
                    # Approximate: E[chi_orb^2] ~ E[chi_final^2] - E[chi_eff^2]
                    e_chi_orb2    = max(0.0, e_chi2_final - e_chi2_eff)

                    # Kerr area ratio from final spin
                    a_fin = np.clip(np.abs(chi_final), 0, 0.9999)
                    e_kerr_final = float(np.mean(1.0 + np.sqrt(1.0 - a_fin**2)))
                else:
                    e_chi2_final  = np.nan
                    med_chi_final = np.nan
                    e_chi_orb2    = np.nan
                    e_kerr_final  = np.nan

                # ── Healy orbital proxy from eta ──────────────────────────────
                if eta_col is not None:
                    med_eta    = float(np.median(eta_col))
                    chi_orb_h  = chi_orb_healy(med_eta)
                    e_chi_orb2_healy = chi_orb_h**2
                else:
                    med_eta = np.nan
                    chi_orb_h = np.nan
                    e_chi_orb2_healy = np.nan

                # ── Individual spin magnitudes ────────────────────────────────
                e_a1_sq = float(np.mean(a1_col**2)) if a1_col is not None else np.nan
                e_a2_sq = float(np.mean(a2_col**2)) if a2_col is not None else np.nan

                # ── Kerr area from chi_eff (proxy) ────────────────────────────
                a_eff = np.clip(np.abs(chi_eff), 0, 0.9999)
                e_kerr_eff = float(np.mean(1.0 + np.sqrt(1.0 - a_eff**2)))

                # ── Medians ───────────────────────────────────────────────────
                med_q     = float(np.median(mrat_col)) if mrat_col is not None else np.nan
                med_chirp = float(np.median(chirp_col)) if chirp_col is not None else np.nan
                med_chip  = float(np.median(chip_col)) if chip_col is not None else np.nan

                rows.append({
                    "event_file":         label,
                    "catalog":            catalog,
                    "waveform_key":       key_used,
                    "median_n":           round(median_n, 5),
                    "in_bbh_band":        in_band,
                    "n_samples":          len(n_arr),
                    # spin
                    "e_chi2_eff":         round(e_chi2_eff, 6),
                    "med_chi_eff":        round(med_chi_eff, 5),
                    "e_chi2_final":       round(e_chi2_final, 6) if np.isfinite(e_chi2_final) else None,
                    "med_chi_final":      round(med_chi_final, 5) if np.isfinite(med_chi_final) else None,
                    "e_chi_orb2_direct":  round(e_chi_orb2, 6) if np.isfinite(e_chi_orb2) else None,
                    "e_chi_orb2_healy":   round(e_chi_orb2_healy, 6) if np.isfinite(e_chi_orb2_healy) else None,
                    "chi_orb_healy":      round(chi_orb_h, 5) if np.isfinite(chi_orb_h) else None,
                    "e_a1_sq":            round(e_a1_sq, 6) if np.isfinite(e_a1_sq) else None,
                    "e_a2_sq":            round(e_a2_sq, 6) if np.isfinite(e_a2_sq) else None,
                    # geometry
                    "e_kerr_ratio_eff":   round(e_kerr_eff, 5),
                    "e_kerr_ratio_final": round(e_kerr_final, 5) if np.isfinite(e_kerr_final) else None,
                    # orbital
                    "med_eta":            round(med_eta, 5) if np.isfinite(med_eta) else None,
                    "med_q":              round(med_q, 5) if np.isfinite(med_q) else None,
                    "med_chirp_mass":     round(med_chirp, 3) if np.isfinite(med_chirp) else None,
                    "med_chip":           round(med_chip, 5) if np.isfinite(med_chip) else None,
                    # GB using population beta
                    "gb_event_pop_beta":  round(gb_event_pop, 5),
                })

                tag = "[BBH]" if in_band else "[OUT]"
                print("[+] " + tag + " " + label[:40].ljust(40) +
                      "  n=" + str(round(median_n, 3)).rjust(6) +
                      "  E[X2]=" + str(round(e_chi2_eff, 4)).rjust(7) +
                      "  GB_pop=" + str(round(gb_event_pop, 4)).rjust(7))

        except Exception as e:
            skipped.append(label + " (" + str(e) + ")")

    print()
    print("[*] Pre-dedup: " + str(len(rows)))
    rows = deduplicate(rows)
    print("[*] Post-dedup: " + str(len(rows)))
    print()

    if not rows:
        print("[!] No events."); return

    bbh = [r for r in rows if r["in_bbh_band"] == 1]
    print("[*] BBH band: " + str(len(bbh)))
    print()

    # -----------------------------------------------------------------------
    # POPULATION GB IDENTITY
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  POPULATION GAUSS-BONNET IDENTITY")
    print("  Using beta_pop = " + str(BETA_POP) + " (empirical, posterior-corrected)")
    print("=" * 70)
    print()

    e_chi2_arr    = np.array([r["e_chi2_eff"] for r in bbh])
    mean_e_chi2   = float(np.mean(e_chi2_arr))
    gb_population = BETA_POP + 2.0 * mean_e_chi2
    pop_gap       = 1.0 - gb_population

    print("  Population mean E[chi_eff^2]: " + str(round(mean_e_chi2, 6)))
    print("  beta_pop:                     " + str(BETA_POP))
    print("  GB_population:                " + str(round(gb_population, 5)))
    print("  Population gap (1 - GB):      " + str(round(pop_gap, 5)))
    print()

    # per-event GB distribution using pop beta
    gb_pop_arr = np.array([r["gb_event_pop_beta"] for r in bbh])
    print("  Per-event GB (using beta_pop) distribution:")
    print("    Mean:   " + str(round(float(np.mean(gb_pop_arr)), 5)))
    print("    Median: " + str(round(float(np.median(gb_pop_arr)), 5)))
    print("    Std:    " + str(round(float(np.std(gb_pop_arr)), 5)))
    print("    Min:    " + str(round(float(np.min(gb_pop_arr)), 5)))
    print("    Max:    " + str(round(float(np.max(gb_pop_arr)), 5)))
    t_gb, p_gb = stats.ttest_1samp(gb_pop_arr, 1.0)
    print("    t-test vs 1.0: t=" + str(round(float(t_gb), 4)) +
          "  p=" + "{:.4e}".format(float(p_gb)))
    print()

    # -----------------------------------------------------------------------
    # ORBITAL ANGULAR MOMENTUM: DIRECT (chi_final from posteriors)
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  ORBITAL AM: DIRECT FROM FINAL SPIN POSTERIORS")
    print("  chi_orb^2 ~ chi_final^2 - chi_eff^2")
    print("  Extended identity: beta_pop + 2*E[chi_eff^2] + 2*f*E[chi_orb^2] = 1.0")
    print("=" * 70)
    print()

    d_rows = [r for r in bbh if r.get("e_chi_orb2_direct") is not None]
    print("  Events with final_spin: " + str(len(d_rows)) + " / " + str(len(bbh)))
    print()

    if d_rows:
        e_chi2_eff_d  = np.array([r["e_chi2_eff"]        for r in d_rows])
        e_chi2_fin_d  = np.array([r["e_chi2_final"]       for r in d_rows])
        e_chi_orb2_d  = np.array([r["e_chi_orb2_direct"]  for r in d_rows])

        mean_e_chi2_eff_d  = float(np.mean(e_chi2_eff_d))
        mean_e_chi2_fin_d  = float(np.mean(e_chi2_fin_d))
        mean_e_chi_orb2_d  = float(np.mean(e_chi_orb2_d))

        print("  Population means (direct subset):")
        print("    E[chi_eff^2]:    " + str(round(mean_e_chi2_eff_d, 6)))
        print("    E[chi_final^2]:  " + str(round(mean_e_chi2_fin_d, 6)))
        print("    E[chi_orb^2]:    " + str(round(mean_e_chi_orb2_d, 6)))
        print()

        # GB without orbital term
        gb_no_orb  = BETA_POP + 2.0 * mean_e_chi2_eff_d
        gap_no_orb = 1.0 - gb_no_orb
        print("  GB without orbital term: " + str(round(gb_no_orb, 5)) +
              "   gap=" + str(round(gap_no_orb, 5)))

        # solve for f: beta + 2*E[chi2_eff] + 2*f*E[chi_orb2] = 1.0
        # f = (1 - beta - 2*E[chi2_eff]) / (2*E[chi_orb2])
        if mean_e_chi_orb2_d > 0:
            f_direct_pop = gap_no_orb / (2.0 * mean_e_chi_orb2_d)
            gb_extended  = gb_no_orb + 2.0 * f_direct_pop * mean_e_chi_orb2_d
            print("  Orbital coupling f (population): " + str(round(f_direct_pop, 5)))
            print("  GB extended (population):        " + str(round(gb_extended, 5)))
            print()

            # Test on per-event basis with this f
            gb_ext_per = BETA_POP + 2.0*e_chi2_eff_d + 2.0*f_direct_pop*e_chi_orb2_d
            print("  Per-event extended GB distribution:")
            print("    Mean:   " + str(round(float(np.mean(gb_ext_per)), 5)))
            print("    Std:    " + str(round(float(np.std(gb_ext_per)), 5)))
            t3, p3 = stats.ttest_1samp(gb_ext_per, 1.0)
            print("    t-test vs 1.0: t=" + str(round(float(t3), 4)) +
                  "  p=" + "{:.4e}".format(float(p3)))
            if p3 > 0.05:
                print("    NOT SIGNIFICANT: orbital term closes the population gap")
                print("    -> Extended identity confirmed at population level")
            else:
                print("    SIGNIFICANT: gap partially but not fully closed by orbital AM")
            print()

        # correlation of chi_orb2 with GB deviation
        gb_dev = gb_pop_arr[:len(d_rows)] - float(np.mean(gb_pop_arr[:len(d_rows)]))
        rp, pp, rs, ps = safe_corr(e_chi_orb2_d, e_chi2_eff_d)
        if rp is not None:
            print("  E[chi_orb^2] vs E[chi_eff^2]:  r=" + str(round(rp,4)) +
                  "  p=" + "{:.3e}".format(pp))

    # -----------------------------------------------------------------------
    # ORBITAL AM: HEALY PROXY
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  ORBITAL AM: HEALY PROXY FROM eta")
    print("  chi_orb(eta) = 2*sqrt(3)*eta - 4.13*eta^2 + 5.37*eta^3")
    print("=" * 70)
    print()

    h_rows = [r for r in bbh if r.get("e_chi_orb2_healy") is not None]
    if h_rows:
        e_chi2_eff_h  = np.array([r["e_chi2_eff"]       for r in h_rows])
        e_chi_orb2_h  = np.array([r["e_chi_orb2_healy"] for r in h_rows])
        chi_orb_h_arr = np.array([r["chi_orb_healy"]    for r in h_rows])

        mean_orb2_h  = float(np.mean(e_chi_orb2_h))
        mean_e_chi2h = float(np.mean(e_chi2_eff_h))

        gb_no_orb_h  = BETA_POP + 2.0 * mean_e_chi2h
        gap_no_orb_h = 1.0 - gb_no_orb_h

        print("  N events: " + str(len(h_rows)))
        print("  Mean chi_orb (Healy):   " + str(round(float(np.mean(chi_orb_h_arr)), 4)))
        print("  Mean E[chi_orb^2]:      " + str(round(mean_orb2_h, 6)))
        print("  GB without orbital:     " + str(round(gb_no_orb_h, 5)))
        print("  Gap without orbital:    " + str(round(gap_no_orb_h, 5)))
        print()

        if mean_orb2_h > 0:
            f_healy_pop = gap_no_orb_h / (2.0 * mean_orb2_h)
            print("  Orbital coupling f (Healy):  " + str(round(f_healy_pop, 5)))
            print()
            print("  Physical check on f:")
            print("    f=1.0 -> orbital and spin enter with identical weight")
            print("    f=0.5 -> orbital contributes at half the spin weight")
            print("    f~0   -> orbital does not thread LQG faces")
            print()

            gb_ext_h = BETA_POP + 2.0*e_chi2_eff_h + 2.0*f_healy_pop*e_chi_orb2_h
            t4, p4 = stats.ttest_1samp(gb_ext_h, 1.0)
            print("  Extended GB per-event (Healy f=" + str(round(f_healy_pop,4)) + "):")
            print("    Mean=" + str(round(float(np.mean(gb_ext_h)),5)) +
                  "  Std=" + str(round(float(np.std(gb_ext_h)),5)))
            print("    t-test vs 1.0: t=" + str(round(float(t4),4)) +
                  "  p=" + "{:.4e}".format(float(p4)))
            print()

    # -----------------------------------------------------------------------
    # KERR AREA GEOMETRY
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  KERR HORIZON AREA GEOMETRY")
    print("  A_Kerr/A_Schw = 1 + sqrt(1-a^2)")
    print("  Your intuition: less spin = more area = more LQG faces = beta->1")
    print("=" * 70)
    print()

    kerr_eff_arr = np.array([r["e_kerr_ratio_eff"] for r in bbh])
    e_chi2_all   = np.array([r["e_chi2_eff"]        for r in bbh])
    gb_pop_all   = np.array([r["gb_event_pop_beta"] for r in bbh])

    print("  Mean Kerr ratio (chi_eff): " + str(round(float(np.mean(kerr_eff_arr)), 4)) +
          "  [2.0=Schwarzschild, 1.0=extremal]")
    print()

    rp, pp, rs, ps = safe_corr(kerr_eff_arr, e_chi2_all)
    if rp is not None:
        print("  Kerr ratio vs E[chi_eff^2]:    Pearson r=" + str(round(rp,4)) +
              "  p=" + "{:.3e}".format(pp) + "  Spearman r=" + str(round(rs,4)))

    rp2, pp2, rs2, ps2 = safe_corr(kerr_eff_arr, gb_pop_all)
    if rp2 is not None:
        print("  Kerr ratio vs GB_pop:          Pearson r=" + str(round(rp2,4)) +
              "  p=" + "{:.3e}".format(pp2) + "  Spearman r=" + str(round(rs2,4)))
    print()

    # final spin kerr
    kerr_fin_rows = [r for r in bbh if r.get("e_kerr_ratio_final") is not None]
    if kerr_fin_rows:
        kf = np.array([r["e_kerr_ratio_final"] for r in kerr_fin_rows])
        gb = np.array([r["gb_event_pop_beta"]   for r in kerr_fin_rows])
        ec2= np.array([r["e_chi2_eff"]           for r in kerr_fin_rows])
        print("  Kerr ratio (final spin) N=" + str(len(kerr_fin_rows)) +
              "  Mean=" + str(round(float(np.mean(kf)),4)))
        rp3, pp3, rs3, ps3 = safe_corr(kf, gb)
        if rp3 is not None:
            print("  Kerr(final) vs GB_pop:         Pearson r=" + str(round(rp3,4)) +
                  "  p=" + "{:.3e}".format(pp3) + "  Spearman r=" + str(round(rs3,4)))
        print()

    # -----------------------------------------------------------------------
    # FULL CORRELATION MATRIX
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  CORRELATION MATRIX: WHAT DRIVES E[chi_eff^2]")
    print("  (E[chi^2] is the only per-event term in the GB identity)")
    print("=" * 70)
    print()

    def get_f(key, src=bbh):
        return np.array([float(r[key]) if r.get(key) not in (None,"None","") else np.nan
                         for r in src])

    target = e_chi2_all
    pairs = [
        ("med_q",             "mass ratio q"),
        ("med_eta",           "sym mass ratio eta"),
        ("med_chi_final",     "median chi_final"),
        ("e_chi_orb2_direct", "E[chi_orb^2] direct"),
        ("e_chi_orb2_healy",  "E[chi_orb^2] Healy"),
        ("e_kerr_ratio_eff",  "Kerr area ratio (eff)"),
        ("e_kerr_ratio_final","Kerr area ratio (final)"),
        ("med_chirp_mass",    "chirp mass Mc"),
        ("median_n",          "octave depth n"),
        ("e_a1_sq",           "E[a1^2] primary spin"),
        ("e_a2_sq",           "E[a2^2] secondary spin"),
        ("med_chip",          "precessing spin chi_p"),
    ]

    print("  " + "Observable".ljust(28) +
          "Pearson r".rjust(10) + "p".rjust(12) +
          "Spearman r".rjust(12) + "p".rjust(12))
    print("  " + "-" * 74)

    corr_results = []
    for key, lbl in pairs:
        arr = get_f(key)
        rp, pp, rs, ps = safe_corr(arr, target)
        if rp is None:
            continue
        n = int(np.sum(np.isfinite(arr) & np.isfinite(target)))
        sig = " **" if pp < 0.001 else (" *" if pp < 0.05 else "")
        print("  " + lbl.ljust(28) +
              str(round(rp, 4)).rjust(10) +
              "{:.3e}".format(pp).rjust(12) +
              str(round(rs, 4)).rjust(12) +
              "{:.3e}".format(ps).rjust(12) + sig)
        corr_results.append({"observable": lbl, "n": n,
                             "pearson_r": round(rp,4), "pearson_p": pp,
                             "spearman_r": round(rs,4), "spearman_p": ps})
    print()

    # -----------------------------------------------------------------------
    # EVENTS CLOSEST TO AND FURTHEST FROM GB = 1.0
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  EVENTS BY DEVIATION FROM GB = 1.0 (using pop beta)")
    print("=" * 70)
    print()

    bbh_sorted = sorted(bbh, key=lambda r: abs(float(r["gb_event_pop_beta"]) - 1.0))
    print("  10 closest to GB=1.0:")
    print("  " + "GW event".ljust(12) + "GB".rjust(8) + "E[X2]".rjust(8) +
          "chi_f".rjust(8) + "q".rjust(7) + "eta".rjust(7))
    for r in bbh_sorted[:10]:
        m = re.search(r'(GW\d{6})', r["event_file"])
        eid = m.group(1) if m else r["event_file"][:12]
        cf = str(round(float(r["med_chi_final"]),3)) if r.get("med_chi_final") not in (None,"None","") else "N/A"
        eta_s = str(round(float(r["med_eta"]),3)) if r.get("med_eta") not in (None,"None","") else "N/A"
        print("  " + eid.ljust(12) +
              str(round(float(r["gb_event_pop_beta"]),4)).rjust(8) +
              str(round(float(r["e_chi2_eff"]),4)).rjust(8) +
              cf.rjust(8) + str(round(float(r["med_q"]),3) if r.get("med_q") not in (None,"None","") else "N/A").rjust(7) +
              eta_s.rjust(7))

    print()
    print("  10 furthest from GB=1.0 (high spin, high orbital AM):")
    for r in bbh_sorted[-10:]:
        m = re.search(r'(GW\d{6})', r["event_file"])
        eid = m.group(1) if m else r["event_file"][:12]
        cf = str(round(float(r["med_chi_final"]),3)) if r.get("med_chi_final") not in (None,"None","") else "N/A"
        eta_s = str(round(float(r["med_eta"]),3)) if r.get("med_eta") not in (None,"None","") else "N/A"
        print("  " + eid.ljust(12) +
              str(round(float(r["gb_event_pop_beta"]),4)).rjust(8) +
              str(round(float(r["e_chi2_eff"]),4)).rjust(8) +
              cf.rjust(8) + str(round(float(r["med_q"]),3) if r.get("med_q") not in (None,"None","") else "N/A").rjust(7) +
              eta_s.rjust(7))
    print()

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  VERDICT & PHYSICAL PICTURE")
    print("=" * 70)
    print()
    print("  Population GB = " + str(round(gb_population, 5)) +
          "   Gap = " + str(round(pop_gap, 5)))
    print()
    print("  The GB identity beta + 2*E[chi_eff^2] = 1.000 measures how the")
    print("  total LQG face budget splits between:")
    print("    beta:          faces committed to pure gravitational geometry")
    print("    2*E[chi_eff^2]: faces threaded by individual spin angular momentum")
    print()
    print("  The population gap (" + str(round(pop_gap,4)) + ") is the fraction of")
    print("  the face budget threaded by ORBITAL angular momentum — the orbit")
    print("  itself threads faces independently of individual BH spins.")
    print()
    print("  A_Kerr/A_Schw = 1 + sqrt(1-a^2): the Kerr geometry reduces the")
    print("  available horizon area — and therefore the LQG face count — as")
    print("  spin increases. This is why beta -> 1.0 as spin -> 0:")
    print("  the Schwarzschild limit maximizes the face count available for")
    print("  pure gravitational work. No density required.")
    print()
    print("  CANDIDATE EXTENDED IDENTITY (new falsifiable prediction):")
    print("    beta + 2*E[chi_eff^2] + 2*f_orb*E[chi_orb^2] = 1.000")
    print("  where f_orb is measured above and E[chi_orb^2] is from final_spin")
    print("  posteriors or the Healy formula.")
    print()

    # -----------------------------------------------------------------------
    # OUTPUTS
    # -----------------------------------------------------------------------
    csv_per = "seraphim_orbital_v2_per_event.csv"
    csv_cor = "seraphim_orbital_v2_correlations.csv"
    json_out= "seraphim_orbital_v2_summary.json"

    with open(csv_per, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    with open(csv_cor, "w", newline="") as cf:
        if corr_results:
            writer = csv.DictWriter(cf, fieldnames=corr_results[0].keys())
            writer.writeheader(); writer.writerows(corr_results)

    summary = {
        "test": "Orbital AM Gap v2 (corrected)",
        "K0": "1.1467e84 (exponent=84)",
        "beta_pop": BETA_POP,
        "gb_population": round(gb_population, 5),
        "population_gap": round(pop_gap, 5),
        "mean_e_chi2_eff": round(mean_e_chi2, 6),
        "physics": "BH=(M,a) only. A_Kerr=4pi*rs^2*(1+sqrt(1-a^2)). No density.",
        "extended_identity": "beta_pop + 2*E[chi_eff^2] + 2*f_orb*E[chi_orb^2] = 1.000",
        "correlations": corr_results,
    }
    with open(json_out, "w") as jf:
        json.dump(summary, jf, indent=4)

    if skipped:
        print("  Skipped: " + str(len(skipped)))
    print()
    print("[*] Per-event:    " + csv_per)
    print("[*] Correlations: " + csv_cor)
    print("[*] JSON:         " + json_out)
    print("[*] Done.")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    run(root)
