import os
import glob
import csv
import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ==============================================================================
# SERAPHIM CONVENTION FALSIFICATION TEST
#
# Tests two HUP minimum conventions against GWOSC data:
#
#   Robertson (correct): Dx * Dp >= hbar/2   -> K0 = 1.1467e84  -> n_pred = 5.314
#   Heisenberg (wrong?): Dx * Dp >= hbar      -> K0 = 4.5868e84  -> n_pred = 4.314
#
# If Robertson is correct: BBH events cluster tightly at n ~ 5.314
# If Heisenberg is correct: BBH events cluster tightly at n ~ 4.314
# If neither: n scatters with no clustering (CV >> 2%)
#
# The test is falsifiable because BOTH conventions predict tight clustering
# but at different n. Whichever K0 gives lower CV wins.
# ==============================================================================

ALPHA      = 0.007297
NU_PLANCK  = 1.8549e43
J_SPIN     = 0.5
SQRT_J     = np.sqrt(J_SPIN * (J_SPIN + 1.0))

K0_ROBERTSON  = 1.1467e84
K0_HEISENBERG = 4.5868e84
N_PRED_ROBERTSON  = 5.314
N_PRED_HEISENBERG = 4.314

MASS1_NAMES  = ["mass_1_source", "mass_1"]
MASS2_NAMES  = ["mass_2_source", "mass_2"]
MFINAL_NAMES = ["final_mass_source", "final_mass",
                "final_mass_source_non_evolved", "final_mass_non_evolved"]


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
        if key.startswith("C01") and "XPHM" in key:
            try:
                return f[key]["posterior_samples"]
            except Exception:
                pass
    for key in f.keys():
        if "XPHM" in key and "Tidal" not in key and "NSBH" not in key:
            try:
                return f[key]["posterior_samples"]
            except Exception:
                pass
    for key in f.keys():
        if key in ("history", "version"):
            continue
        try:
            return f[key]["posterior_samples"]
        except Exception:
            pass
    return None


def calc_n(m_total, m_final, K0):
    e_loss = m_total - m_final
    valid  = (e_loss > 0) & (e_loss < m_total) & np.isfinite(e_loss)
    if valid.sum() < 10:
        return None
    n_star   = e_loss[valid] / (m_total[valid] * ALPHA)
    nu       = np.sqrt(K0 / (SQRT_J * n_star))
    n_octave = np.log2(NU_PLANCK / nu)
    good     = np.isfinite(n_octave)
    if good.sum() < 10:
        return None
    return float(np.median(n_octave[good]))


def run_test(directory="."):
    h5_files = sorted(
        glob.glob(os.path.join(directory, "*.h5")) +
        glob.glob(os.path.join(directory, "*.hdf5"))
    )

    if not h5_files:
        print("[!] No HDF5 files found in: " + os.path.abspath(directory))
        return

    print("=== SERAPHIM CONVENTION FALSIFICATION TEST ===")
    print()
    print("Robertson K0  = 1.1467e84  -> predicted n = 5.314")
    print("Heisenberg K0 = 4.5868e84  -> predicted n = 4.314")
    print("Files found: " + str(len(h5_files)))
    print()

    results_R = []
    results_H = []
    skipped   = []

    for filepath in h5_files:
        filename = os.path.basename(filepath)
        if "Summary" in filename or "Table" in filename:
            continue
        try:
            with h5py.File(filepath, "r") as f:
                ps = get_posteriors(f)
                if ps is None:
                    skipped.append(filename)
                    continue
                m1      = get_col(ps, MASS1_NAMES)
                m2      = get_col(ps, MASS2_NAMES)
                m_final = get_col(ps, MFINAL_NAMES)
                if m1 is None or m2 is None or m_final is None:
                    skipped.append(filename)
                    continue
                m_total = m1 + m2
                n_R = calc_n(m_total, m_final, K0_ROBERTSON)
                n_H = calc_n(m_total, m_final, K0_HEISENBERG)
                if n_R is None or n_H is None:
                    skipped.append(filename)
                    continue
                results_R.append(n_R)
                results_H.append(n_H)
                print("[+] " + filename[:55] +
                      "  n_R=" + str(round(n_R,3)) +
                      "  n_H=" + str(round(n_H,3)))
        except Exception as e:
            skipped.append(filename + " (" + str(e) + ")")

    print()
    print("Skipped: " + str(len(skipped)))
    print()

    if len(results_R) < 5:
        print("[!] Need at least 5 events.")
        return

    arr_R = np.array(results_R)
    arr_H = np.array(results_H)

    def band_stats(arr, pred, band_lo, band_hi):
        in_band  = arr[(arr > band_lo) & (arr < band_hi)]
        mean     = float(np.mean(in_band)) if len(in_band) > 0 else float(np.mean(arr))
        std      = float(np.std(in_band))  if len(in_band) > 0 else float(np.std(arr))
        cv       = std / mean * 100
        delta    = abs(mean - pred)
        n_band   = len(in_band)
        return mean, std, cv, delta, n_band

    mean_R, std_R, cv_R, delta_R, nb_R = band_stats(arr_R, N_PRED_ROBERTSON,  4.76, 5.76)
    mean_H, std_H, cv_H, delta_H, nb_H = band_stats(arr_H, N_PRED_HEISENBERG, 3.76, 4.76)

    print("=== ROBERTSON CONVENTION (K0 = 1.1467e84, predicted n = 5.314) ===")
    print("  Events total:   " + str(len(arr_R)))
    print("  In band (BBH):  " + str(nb_R))
    print("  Mean n:         " + str(round(mean_R, 4)))
    print("  Std:            " + str(round(std_R,  4)))
    print("  CV:             " + str(round(cv_R,   3)) + "%")
    print("  Delta from pred:" + str(round(delta_R, 4)))
    print()
    print("=== HEISENBERG CONVENTION (K0 = 4.5868e84, predicted n = 4.314) ===")
    print("  Events total:   " + str(len(arr_H)))
    print("  In band:        " + str(nb_H))
    print("  Mean n:         " + str(round(mean_H, 4)))
    print("  Std:            " + str(round(std_H,  4)))
    print("  CV:             " + str(round(cv_H,   3)) + "%")
    print("  Delta from pred:" + str(round(delta_H, 4)))
    print()

    print("=== VERDICT ===")
    print()
    better = "Robertson" if cv_R < cv_H else "Heisenberg"
    if cv_R < cv_H:
        print("Robertson tighter: CV " + str(round(cv_R,3)) + "% vs " + str(round(cv_H,3)) + "%")
        print("Robertson delta closer: " + str(round(delta_R,4)) + " vs " + str(round(delta_H,4)))
        if cv_R < 3.0 and cv_H > 3.0:
            print()
            print("FALSIFICATION RESULT: Robertson CONFIRMED, Heisenberg RULED OUT")
            print("K0 = 1.1467e84 (Dx*Dp >= hbar/2) is the correct convention.")
            print("K0 = 4.5868e84 (Dx*Dp >= hbar)   is ruled out by the data.")
        elif cv_R < 3.0 and cv_H < 3.0:
            print()
            print("AMBIGUOUS: Both conventions produce clustering.")
            print("Distinguish by mean n: Robertson predicts 5.314, Heisenberg predicts 4.314.")
            print("Observed Robertson mean = " + str(round(mean_R,4)) + " (closer to 5.314)")
        else:
            print()
            print("Robertson is better fit but CV > 3%. Check for NSBH contamination.")
    else:
        print("Heisenberg tighter: CV " + str(round(cv_H,3)) + "% vs " + str(round(cv_R,3)) + "%")
        print()
        print("UNEXPECTED: Heisenberg convention fits better. Verify K0 derivation.")

    csv_path = "seraphim_convention_results.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["event_file", "n_Robertson", "n_Heisenberg", "delta_n"])
        for i, (nR, nH) in enumerate(zip(results_R, results_H)):
            fnames = sorted([f for f in os.listdir(directory)
                             if f.endswith(".hdf5") or f.endswith(".h5")
                             if "Summary" not in f and "Table" not in f])
            fname  = fnames[i] if i < len(fnames) else str(i)
            writer.writerow([os.path.basename(fname), round(nR,5), round(nH,5),
                             round(abs(nR-nH),5)])
    print()
    print("[*] Saved " + csv_path)


if __name__ == "__main__":
    run_test()
