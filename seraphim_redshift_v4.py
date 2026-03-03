import os
import glob
import json
import csv
import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ==============================================================================
# SERAPHIM REDSHIFT INVARIANCE TEST v4
# Handles GWTC-2.1, GWTC-3, GWTC-4 combined format
# C00: and C01: prefixes, BBH and NSBH, evolved and non-evolved final mass
# K_0 = 1.1467e84 (correct)
# ==============================================================================

ALPHA     = 0.007297
K_0       = 1.1467e84
NU_PLANCK = 1.8549e43
J_SPIN    = 0.5
SQRT_J    = np.sqrt(J_SPIN * (J_SPIN + 1.0))

# Priority order for each parameter - exact names from probe output
MASS1_NAMES   = ["mass_1_source", "mass_1"]
MASS2_NAMES   = ["mass_2_source", "mass_2"]
MFINAL_NAMES  = ["final_mass_source", "final_mass",
                 "final_mass_source_non_evolved", "final_mass_non_evolved"]
Z_NAMES       = ["redshift"]
DL_NAMES      = ["luminosity_distance", "comoving_distance"]
CHIEFF_NAMES  = ["chi_eff"]


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
    # Try C01 BBH first (GWTC-2.1, GWTC-3)
    for key in f.keys():
        if key.startswith("C01") and "XPHM" in key:
            try:
                return f[key]["posterior_samples"], key, "BBH"
            except Exception:
                pass
    # Try C00 XPHM SpinTaylor (GWTC-4 BBH-in-NSBH files)
    for key in f.keys():
        if "XPHM" in key and "Tidal" not in key and "NSBH" not in key:
            try:
                return f[key]["posterior_samples"], key, "BBH"
            except Exception:
                pass
    # Try any IMRPhenomXPHM
    for key in f.keys():
        if "XPHM" in key:
            try:
                return f[key]["posterior_samples"], key, "BBH_mixed"
            except Exception:
                pass
    # Accept NSBH waveforms - useful for compactness equation test
    for key in f.keys():
        if key in ("history", "version"):
            continue
        if "NSBH" in key or "Tidal" in key or "NRTidal" in key:
            try:
                return f[key]["posterior_samples"], key, "NSBH"
            except Exception:
                pass
    # Last resort: first key with posterior_samples
    for key in f.keys():
        if key in ("history", "version"):
            continue
        try:
            ps = f[key]["posterior_samples"]
            return ps, key, "unknown"
        except Exception:
            pass
    return None, None, None


def calculate_octave(m_total, m_final):
    e_loss = m_total - m_final
    valid  = (e_loss > 0) & (e_loss < m_total) & np.isfinite(e_loss) & np.isfinite(m_total)
    if valid.sum() < 10:
        return np.array([]), valid
    n_star   = e_loss[valid] / (m_total[valid] * ALPHA)
    nu       = np.sqrt(K_0 / (SQRT_J * n_star))
    n_octave = np.log2(NU_PLANCK / nu)
    finite   = np.isfinite(n_octave)
    return n_octave[finite], valid


def run_redshift_test(directory="."):
    h5_files = glob.glob(os.path.join(directory, "*.h5"))
    h5_files += glob.glob(os.path.join(directory, "*.hdf5"))
    h5_files = sorted(h5_files)

    if not h5_files:
        print("[!] No HDF5 files found in: " + os.path.abspath(directory))
        return

    print("[*] Found " + str(len(h5_files)) + " HDF5 files")
    print("[*] K_0 = 1.1467e84 Hz^2")
    print("")

    results  = []
    skipped  = []

    for filepath in h5_files:
        filename = os.path.basename(filepath)
        if "Summary" in filename or "Table" in filename:
            skipped.append(filename + " (summary file)")
            continue

        try:
            with h5py.File(filepath, "r") as f:
                ps, key_used, event_type = get_posteriors(f)

                if ps is None:
                    skipped.append(filename + " (no posterior_samples)")
                    continue

                m1      = get_col(ps, MASS1_NAMES)
                m2      = get_col(ps, MASS2_NAMES)
                m_final = get_col(ps, MFINAL_NAMES)
                z       = get_col(ps, Z_NAMES)
                d_l     = get_col(ps, DL_NAMES)
                chi_eff = get_col(ps, CHIEFF_NAMES)

                if m1 is None or m2 is None or m_final is None:
                    skipped.append(filename + " (missing mass cols in " + key_used + ")")
                    continue

                if z is None:
                    skipped.append(filename + " (no redshift col)")
                    continue

                m_total = m1 + m2
                n_array, valid_idx = calculate_octave(m_total, m_final)

                if len(n_array) < 50:
                    skipped.append(filename + " (only " + str(len(n_array)) + " valid samples)")
                    continue

                z_valid = z[valid_idx][:len(n_array)]
                dl_med  = float(np.median(d_l[valid_idx][:len(n_array)])) if d_l is not None else -1.0
                ce_med  = float(np.median(chi_eff[valid_idx][:len(n_array)])) if chi_eff is not None else -99.0

                med_n   = float(np.median(n_array))
                med_z   = float(np.median(z_valid))
                in_band = int(4.76 <= med_n <= 5.76)

                row = {
                    "event_file":          filename,
                    "event_type":          event_type,
                    "waveform_key":        key_used,
                    "median_n":            round(med_n, 5),
                    "std_n":               round(float(np.std(n_array)), 5),
                    "median_redshift":     round(med_z, 5),
                    "median_lum_dist_Mpc": round(dl_med, 3),
                    "median_chi_eff":      round(ce_med, 5),
                    "n_samples":           int(len(n_array)),
                    "in_bbh_band":         in_band
                }
                results.append(row)

                tag = " [BBH]" if in_band else " [OUT]"
                print("[+]" + tag + " " + filename[:52] +
                      "  n=" + str(round(med_n, 3)) +
                      "  z=" + str(round(med_z, 3)) +
                      "  type=" + event_type)

        except Exception as e:
            skipped.append(filename + " (error: " + str(e) + ")")

    print("")
    print("=== PARSE SUMMARY ===")
    print("  Processed: " + str(len(results)))
    print("  Skipped:   " + str(len(skipped)))
    for s in skipped:
        print("    - " + s)

    if len(results) < 3:
        print("[!] Need at least 3 events. Check folder.")
        return

    n_vals  = np.array([r["median_n"]            for r in results])
    z_vals  = np.array([r["median_redshift"]      for r in results])
    dl_vals = np.array([r["median_lum_dist_Mpc"]  for r in results])

    bbh_mask = np.array([r["in_bbh_band"] for r in results], dtype=bool)
    n_bbh    = n_vals[bbh_mask]
    z_bbh    = z_vals[bbh_mask]

    print("")
    print("=== OCTAVE STATS ===")
    print("  All (" + str(len(n_vals)) + " events):")
    print("    mean n=" + str(round(float(np.mean(n_vals)), 4)) +
          "  std="  + str(round(float(np.std(n_vals)), 4)) +
          "  CV="   + str(round(float(np.std(n_vals)/np.mean(n_vals)*100), 3)) + "%")
    if len(n_bbh) > 0:
        print("  BBH band (" + str(len(n_bbh)) + " events, 4.76-5.76):")
        print("    mean n=" + str(round(float(np.mean(n_bbh)), 4)) +
              "  std="  + str(round(float(np.std(n_bbh)), 4)) +
              "  CV="   + str(round(float(np.std(n_bbh)/np.mean(n_bbh)*100), 3)) + "%")

    print("")
    print("=== REDSHIFT INVARIANCE ALL EVENTS ===")
    r_z,  p_z  = pearsonr(z_vals, n_vals)
    sr_z, sp_z = spearmanr(z_vals, n_vals)
    print("  Pearson  r(n,z)  = " + str(round(r_z,  4)) + "  p=" + str(round(p_z,  4)))
    print("  Spearman r(n,z)  = " + str(round(sr_z, 4)) + "  p=" + str(round(sp_z, 4)))

    if dl_vals[0] > 0:
        r_dl,  p_dl  = pearsonr(dl_vals, n_vals)
        sr_dl, sp_dl = spearmanr(dl_vals, n_vals)
        print("  Pearson  r(n,dL) = " + str(round(r_dl,  4)) + "  p=" + str(round(p_dl,  4)))
        print("  Spearman r(n,dL) = " + str(round(sr_dl, 4)) + "  p=" + str(round(sp_dl, 4)))

    if len(n_bbh) >= 3:
        print("")
        print("=== REDSHIFT INVARIANCE BBH BAND ONLY ===")
        r_zb,  p_zb  = pearsonr(z_bbh, n_bbh)
        sr_zb, sp_zb = spearmanr(z_bbh, n_bbh)
        print("  Events in band: " + str(len(n_bbh)))
        print("  Pearson  r(n,z)  = " + str(round(r_zb,  4)) + "  p=" + str(round(p_zb,  4)))
        print("  Spearman r(n,z)  = " + str(round(sr_zb, 4)) + "  p=" + str(round(sp_zb, 4)))

    print("")
    print("=== VERDICT ===")
    if abs(r_z) < 0.2 and p_z > 0.05:
        print("  PASS: Null correlation confirmed.")
        print("  Octave n is redshift-invariant. No cosmological drift.")
    elif abs(r_z) < 0.3 and p_z > 0.05:
        print("  MARGINAL: Weak correlation, not significant.")
        print("  Check BBH-band-only result above.")
    else:
        print("  FLAG: r=" + str(round(r_z, 4)) + " p=" + str(round(p_z, 4)))
        print("  Possible NSBH contamination or selection effect.")
        if len(n_bbh) >= 3:
            print("  See BBH-band result for cleaner picture.")

    csv_path  = "seraphim_redshift_results.csv"
    json_path = "seraphim_redshift_results.json"

    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as jf:
        json.dump(results, jf, indent=4)

    print("")
    print("[*] Saved " + csv_path + " with " + str(len(results)) + " events")
    print("[*] Saved " + json_path)


if __name__ == "__main__":
    run_redshift_test()
