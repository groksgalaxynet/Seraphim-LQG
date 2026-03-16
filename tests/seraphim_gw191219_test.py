import os
import csv
import json
import numpy as np
import h5py
from scipy import stats

# ==============================================================================
# SERAPHIM TEST 3: GW191219 POSTERIOR ISOLATION
# 
# GW191219_163120 is the only event in the dataset with median n < n_flip = 3.561.
# Reported median n = 3.272 -- below the geometric floor of the framework.
#
# This test answers three questions:
#   (a) Full posterior range of n -- is sub-floor sustained across the distribution?
#   (b) What fraction of posterior samples fall below n_flip = 3.561?
#   (c) Does the anomaly survive at the 90% credible level?
#
# Secondary questions:
#   (d) What is the event's compactness C implied by n_obs via the linear equation?
#   (e) Is it consistent with an NSBH interpretation (C ~ 0.28-0.34)?
#   (f) How does it compare to GW190814 (the known NSBH reference)?
#   (g) What would n need to be for GW191219 to sit at n_flip exactly?
#
# K_0 = 1.1467e84 Hz^2  (CRITICAL: exponent is 84, not 4)
# ==============================================================================

K_0        = 1.1467e84
NU_PLANCK  = 1.8549e43
ALPHA      = 0.007297
J_SPIN     = 0.5
SQRT_J     = np.sqrt(J_SPIN * (J_SPIN + 1.0))

N_FLIP     = 3.561    # geometric floor (singularity prediction)
N_BBH      = 5.314    # BBH band center
N_BAND_LO  = 4.76     # BBH band lower edge
SLOPE      = 3.506    # compactness equation slope
TARGET_ID  = "GW191219"

MASS1_NAMES  = ["mass_1_source", "mass_1"]
MASS2_NAMES  = ["mass_2_source", "mass_2"]
MFINAL_NAMES = ["final_mass_source", "final_mass",
                "final_mass_source_non_evolved", "final_mass_non_evolved"]
CHIEFF_NAMES = ["chi_eff"]
MCHIRP_NAMES = ["chirp_mass_source", "chirp_mass"]
Q_NAMES      = ["mass_ratio"]
Z_NAMES      = ["redshift"]
M2_NAMES     = ["mass_2_source", "mass_2"]


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


def get_all_posteriors(f):
    """Return ALL waveform keys with posterior_samples for comparison."""
    keys = []
    for key in f.keys():
        if key in ("history", "version"):
            continue
        try:
            _ = f[key]["posterior_samples"]
            keys.append(key)
        except Exception:
            pass
    return keys


def calculate_octave_array(m_total, m_final):
    e_loss = m_total - m_final
    valid  = (e_loss > 0) & (e_loss < m_total) & np.isfinite(e_loss) & np.isfinite(m_total)
    if valid.sum() < 10:
        return np.array([]), valid
    n_star   = e_loss[valid] / (m_total[valid] * ALPHA)
    nu       = np.sqrt(K_0 / (SQRT_J * n_star))
    n_octave = np.log2(NU_PLANCK / nu)
    finite   = np.isfinite(n_octave)
    return n_octave[finite], valid


def implied_compactness(n_obs):
    """Invert linear compactness equation: C = (n - n_flip) / slope"""
    return (n_obs - N_FLIP) / SLOPE


def analyze_posterior(ps, key_label):
    m1      = get_col(ps, MASS1_NAMES)
    m2      = get_col(ps, MASS2_NAMES)
    m_final = get_col(ps, MFINAL_NAMES)
    chi_eff = get_col(ps, CHIEFF_NAMES)
    mchirp  = get_col(ps, MCHIRP_NAMES)
    q       = get_col(ps, Q_NAMES)
    z       = get_col(ps, Z_NAMES)
    m2_col  = get_col(ps, M2_NAMES)

    if m1 is None or m2 is None or m_final is None:
        return None, "missing mass columns"

    m_total = m1 + m2
    n_array, valid_idx = calculate_octave_array(m_total, m_final)

    if len(n_array) < 50:
        return None, "only " + str(len(n_array)) + " valid samples"

    # Core n statistics
    n_median  = float(np.median(n_array))
    n_mean    = float(np.mean(n_array))
    n_std     = float(np.std(n_array))
    n_min     = float(np.min(n_array))
    n_max     = float(np.max(n_array))
    ci_5      = float(np.percentile(n_array, 5))
    ci_10     = float(np.percentile(n_array, 10))
    ci_90     = float(np.percentile(n_array, 90))
    ci_95     = float(np.percentile(n_array, 95))

    # Fraction below n_flip
    frac_below_flip  = float(np.mean(n_array < N_FLIP))
    frac_below_3_0   = float(np.mean(n_array < 3.0))
    frac_below_3_5   = float(np.mean(n_array < 3.5))
    frac_above_flip  = float(np.mean(n_array >= N_FLIP))
    frac_in_nsbh     = float(np.mean((n_array >= 4.0) & (n_array < N_BAND_LO)))
    frac_in_bbh      = float(np.mean((n_array >= N_BAND_LO) & (n_array <= 5.76)))

    # Anomaly survival: does the 90% CI stay below n_flip?
    anomaly_90 = ci_95 < N_FLIP   # upper 90% credible bound still below floor
    anomaly_50 = float(np.percentile(n_array, 75)) < N_FLIP

    # Implied compactness from median n
    C_implied_median = implied_compactness(n_median)
    C_implied_5pct   = implied_compactness(ci_5)
    C_implied_95pct  = implied_compactness(ci_95)

    # Mass parameters
    med_m1  = float(np.median(m1))          if m1      is not None else None
    med_m2  = float(np.median(m2_col))      if m2_col  is not None else None
    med_mc  = float(np.median(mchirp))      if mchirp  is not None else None
    med_q   = float(np.median(q))           if q       is not None else None
    med_chi = float(np.median(chi_eff[valid_idx][:len(n_array)])) if chi_eff is not None else None
    med_z   = float(np.median(z[valid_idx][:len(n_array)]))       if z       is not None else None

    result = {
        "waveform_key":       key_label,
        "n_samples":          int(len(n_array)),
        # Core n distribution
        "n_min":              round(n_min,    5),
        "n_5pct":             round(ci_5,     5),
        "n_10pct":            round(ci_10,    5),
        "n_median":           round(n_median, 5),
        "n_mean":             round(n_mean,   5),
        "n_90pct":            round(ci_90,    5),
        "n_95pct":            round(ci_95,    5),
        "n_max":              round(n_max,    5),
        "n_std":              round(n_std,    5),
        # Floor analysis
        "n_flip":             N_FLIP,
        "frac_below_nflip":   round(frac_below_flip, 4),
        "frac_below_3.5":     round(frac_below_3_5,  4),
        "frac_below_3.0":     round(frac_below_3_0,  4),
        "frac_above_nflip":   round(frac_above_flip, 4),
        "frac_in_nsbh_band":  round(frac_in_nsbh,    4),
        "frac_in_bbh_band":   round(frac_in_bbh,     4),
        # Credible interval survival
        "anomaly_survives_90pct_CI": anomaly_90,
        "anomaly_survives_50pct_CI": anomaly_50,
        # Implied compactness
        "C_implied_median":   round(C_implied_median, 5),
        "C_implied_5pct":     round(C_implied_5pct,   5),
        "C_implied_95pct":    round(C_implied_95pct,  5),
        "C_nsbh_expected_lo": 0.28,
        "C_nsbh_expected_hi": 0.34,
        "C_in_nsbh_range":    0.28 <= C_implied_median <= 0.34,
        # Mass parameters
        "median_m1_msun":     round(med_m1,  3) if med_m1  is not None else None,
        "median_m2_msun":     round(med_m2,  3) if med_m2  is not None else None,
        "median_chirp_msun":  round(med_mc,  3) if med_mc  is not None else None,
        "median_q":           round(med_q,   4) if med_q   is not None else None,
        "median_chi_eff":     round(med_chi, 5) if med_chi is not None else None,
        "median_redshift":    round(med_z,   5) if med_z   is not None else None,
    }
    return result, None


def find_h5_files(root="."):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".h5") or fname.endswith(".hdf5"):
                if TARGET_ID in fname:
                    found.append(os.path.join(dirpath, fname))
    return sorted(found)


def run_gw191219_test(root="."):
    print("=" * 70)
    print("  SERAPHIM TEST 3: GW191219 POSTERIOR ISOLATION")
    print("  Target: " + TARGET_ID + "  |  n_flip floor = " + str(N_FLIP))
    print("  K_0 = 1.1467e84 Hz^2  (exponent = 84)")
    print("=" * 70)
    print()

    h5_files = find_h5_files(root)

    if not h5_files:
        print("[!] No files matching '" + TARGET_ID + "' found under: " + os.path.abspath(root))
        print("    Ensure GWTC-3 HDF5 files are present.")
        return

    print("[*] Found " + str(len(h5_files)) + " file(s) matching " + TARGET_ID)
    for f in h5_files:
        print("    " + f)
    print()

    all_results = []

    for filepath in h5_files:
        filename = os.path.basename(filepath)
        print("-" * 70)
        print("FILE: " + filename)
        print("-" * 70)

        try:
            with h5py.File(filepath, "r") as f:
                wf_keys = get_all_posteriors(f)
                print("[*] Waveform keys with posterior_samples: " + str(wf_keys))
                print()

                for key in wf_keys:
                    ps = f[key]["posterior_samples"]
                    res, err = analyze_posterior(ps, key)

                    if err:
                        print("  [-] " + key + ": " + err)
                        continue

                    res["source_file"] = filename
                    all_results.append(res)

                    print("  [+] Waveform: " + key)
                    print("      Samples:  " + str(res["n_samples"]))
                    print()
                    print("      n DISTRIBUTION:")
                    print("        Min:          " + str(res["n_min"]))
                    print("        5th pct:      " + str(res["n_5pct"]))
                    print("        10th pct:     " + str(res["n_10pct"]))
                    print("        Median:       " + str(res["n_median"]))
                    print("        Mean:         " + str(res["n_mean"]))
                    print("        90th pct:     " + str(res["n_90pct"]))
                    print("        95th pct:     " + str(res["n_95pct"]))
                    print("        Max:          " + str(res["n_max"]))
                    print("        Std:          " + str(res["n_std"]))
                    print()
                    print("      FLOOR ANALYSIS (n_flip = " + str(N_FLIP) + "):")
                    print("        Frac below n_flip:    " + str(round(res["frac_below_nflip"]*100, 2)) + "%")
                    print("        Frac below 3.5:       " + str(round(res["frac_below_3.5"]*100, 2)) + "%")
                    print("        Frac below 3.0:       " + str(round(res["frac_below_3.0"]*100, 2)) + "%")
                    print("        Frac above n_flip:    " + str(round(res["frac_above_nflip"]*100, 2)) + "%")
                    print("        Frac in NSBH band:    " + str(round(res["frac_in_nsbh_band"]*100, 2)) + "%")
                    print("        Frac in BBH band:     " + str(round(res["frac_in_bbh_band"]*100, 2)) + "%")
                    print()
                    print("      CREDIBLE INTERVAL SURVIVAL:")
                    print("        Anomaly survives 90% CI: " + str(res["anomaly_survives_90pct_CI"]))
                    print("        Anomaly survives 50% CI: " + str(res["anomaly_survives_50pct_CI"]))
                    print()
                    print("      IMPLIED COMPACTNESS (from linear eq C=(n-n_flip)/slope):")
                    print("        C(median n):  " + str(res["C_implied_median"]))
                    print("        C(5th pct):   " + str(res["C_implied_5pct"]))
                    print("        C(95th pct):  " + str(res["C_implied_95pct"]))
                    print("        NSBH expected range: [0.28, 0.34]")
                    print("        C in NSBH range:     " + str(res["C_in_nsbh_range"]))
                    print()
                    print("      MASS PARAMETERS:")
                    if res["median_m1_msun"]:
                        print("        m1 (Msun):    " + str(res["median_m1_msun"]))
                    if res["median_m2_msun"]:
                        print("        m2 (Msun):    " + str(res["median_m2_msun"]))
                    if res["median_chirp_msun"]:
                        print("        Mc (Msun):    " + str(res["median_chirp_msun"]))
                    if res["median_q"]:
                        print("        q:            " + str(res["median_q"]))
                    if res["median_chi_eff"] is not None:
                        print("        chi_eff:      " + str(res["median_chi_eff"]))
                    if res["median_redshift"] is not None:
                        print("        z:            " + str(res["median_redshift"]))
                    print()

        except Exception as e:
            print("[!] Error reading " + filename + ": " + str(e))

    if not all_results:
        print("[!] No results produced.")
        return

    # -----------------------------------------------------------------------
    # REFERENCE COMPARISON: GW190814
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  REFERENCE COMPARISON: GW190814 (known NSBH, n=4.189, C~0.18)")
    print("=" * 70)
    print()
    print("  GW190814 sits in the NSBH band at n=4.189, C=0.18")
    print("  GW191219 sits BELOW n_flip at n~3.272")
    print()
    print("  If GW191219 is an NSBH at C~0.28-0.34, predicted n = " +
          str(round(N_FLIP + SLOPE * 0.28, 3)) + " to " +
          str(round(N_FLIP + SLOPE * 0.34, 3)))
    print("  Observed median n = " + str(round(all_results[0]["n_median"], 3)) +
          " -- delta from NSBH band center = " +
          str(round(all_results[0]["n_median"] - (N_FLIP + SLOPE * 0.31), 3)))
    print()

    # What fraction of the posterior sits in each band
    print("  POSTERIOR BAND OCCUPATION (primary waveform):")
    r0 = all_results[0]
    print("    Below n_flip (3.561):  " + str(round(r0["frac_below_nflip"]*100, 1)) + "% of samples")
    print("    NSBH band (4.0-4.76):  " + str(round(r0["frac_in_nsbh_band"]*100, 1)) + "% of samples")
    print("    BBH band (4.76-5.76):  " + str(round(r0["frac_in_bbh_band"]*100, 1)) + "% of samples")
    print()

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    r0 = all_results[0]
    frac_below = r0["frac_below_nflip"]
    survives_90 = r0["anomaly_survives_90pct_CI"]
    survives_50 = r0["anomaly_survives_50pct_CI"]

    if survives_90:
        verdict = "CONFIRMED ANOMALY: sub-floor at 90% credible level"
        detail  = ("The entire 90% credible interval lies below n_flip = " + str(N_FLIP) +
                   ". This event cannot be accommodated by the linear compactness equation "
                   "under any standard compact object interpretation.")
    elif frac_below > 0.50:
        verdict = "PROBABLE ANOMALY: majority of posterior below floor"
        detail  = (str(round(frac_below*100,1)) + "% of posterior samples fall below n_flip = " +
                   str(N_FLIP) + ". The anomaly is present but the posterior extends above "
                   "the floor. Improved NR posteriors required for definitive classification.")
    elif frac_below > 0.10:
        verdict = "MARGINAL: significant posterior weight below floor"
        detail  = (str(round(frac_below*100,1)) + "% of posterior samples below n_flip. "
                   "Not confirmed at 90% CI. Monitor with O5 data.")
    else:
        verdict = "NOT CONFIRMED: most posterior above floor"
        detail  = ("Only " + str(round(frac_below*100,1)) + "% of samples below n_flip. "
                   "Previous sub-floor median may have been posterior median bias.")

    print("  " + verdict)
    print()
    print("  " + detail)
    print()
    print("  n_5pct  = " + str(r0["n_5pct"]) + "   (lower 90% credible bound)")
    print("  n_95pct = " + str(r0["n_95pct"]) + "  (upper 90% credible bound)")
    print("  n_flip  = " + str(N_FLIP))
    print()
    print("  Prediction 2 status: " +
          ("STANDING -- sub-floor anomaly survives full posterior analysis." if frac_below > 0.5
           else "WEAKENED -- anomaly present but not dominant in posterior."))

    # -----------------------------------------------------------------------
    # OUTPUTS
    # -----------------------------------------------------------------------
    csv_path  = "seraphim_gw191219_posterior.csv"
    json_path = "seraphim_gw191219_summary.json"

    with open(csv_path, "w", newline="") as cf:
        if all_results:
            writer = csv.DictWriter(cf, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    summary = {
        "test":          "GW191219 Posterior Isolation",
        "target_event":  TARGET_ID,
        "n_flip":        N_FLIP,
        "n_bbh_pred":    N_BBH,
        "K0_verify":     "1.1467e84 Hz^2 (exponent=84)",
        "files_analyzed": len(h5_files),
        "waveforms_analyzed": len(all_results),
        "primary_result": all_results[0] if all_results else None,
        "verdict":       verdict,
        "nsbh_predicted_range": {
            "C_lo": 0.28, "C_hi": 0.34,
            "n_lo": round(N_FLIP + SLOPE * 0.28, 3),
            "n_hi": round(N_FLIP + SLOPE * 0.34, 3),
        },
        "reference_GW190814": {
            "n_obs": 4.189, "C_approx": 0.18,
            "interpretation": "known NSBH, sits in NSBH band as expected"
        },
    }

    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=4)

    print()
    print("[*] Per-waveform CSV: " + csv_path)
    print("[*] Summary JSON:     " + json_path)
    print("[*] Done.")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    run_gw191219_test(root)
