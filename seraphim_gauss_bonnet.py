import os
import glob
import csv
import json
import numpy as np
import h5py

# ==============================================================================
# SERAPHIM GAUSS-BONNET IDENTITY TEST
# Tests: beta + 2 * E[chi_eff^2] = 1.000 (Gauss-Bonnet prediction)
# Where beta = 0.814 (measured) and E[chi_eff^2] = mean(chi_eff^2) from
# full posterior samples (proper expectation value, not median^2)
#
# Framework prediction:  population mean of (beta + 2*E[chi_eff^2]) -> 1.0
# Current measurement:   beta_obs = 0.814, so E[chi_eff^2] predicted ~ 0.093
# O5 prediction:         as C is better constrained, beta -> 1.0
#
# K_0 = 1.1467e84 Hz^2  (CRITICAL: exponent is 84, not 4)
# ==============================================================================

K_0       = 1.1467e84           # Hz^2 — VERIFY exponent = 84
NU_PLANCK = 1.8549e43           # Hz
ALPHA     = 0.007297            # fine structure constant
J_SPIN    = 0.5
SQRT_J    = np.sqrt(J_SPIN * (J_SPIN + 1.0))

N_BBH     = 5.314               # predicted octave depth for BBH
BETA_OBS  = 0.814               # measured compactness exponent

# Column name priority lists (handles all three catalog formats)
MASS1_NAMES  = ["mass_1_source", "mass_1"]
MASS2_NAMES  = ["mass_2_source", "mass_2"]
MFINAL_NAMES = ["final_mass_source", "final_mass",
                "final_mass_source_non_evolved", "final_mass_non_evolved"]
CHIEFF_NAMES = ["chi_eff"]


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
    # Priority 1: C01 + XPHM (GWTC-2.1, GWTC-3 BBH)
    for key in f.keys():
        if key.startswith("C01") and "XPHM" in key:
            try:
                return f[key]["posterior_samples"], key, "BBH"
            except Exception:
                pass
    # Priority 2: XPHM without Tidal/NSBH (GWTC-4 BBH)
    for key in f.keys():
        if "XPHM" in key and "Tidal" not in key and "NSBH" not in key:
            try:
                return f[key]["posterior_samples"], key, "BBH"
            except Exception:
                pass
    # Priority 3: any XPHM
    for key in f.keys():
        if "XPHM" in key:
            try:
                return f[key]["posterior_samples"], key, "BBH_mixed"
            except Exception:
                pass
    # Priority 4: NSBH / Tidal waveforms
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
            return f[key]["posterior_samples"], key, "unknown"
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


def find_h5_files(root="."):
    """Recursively find all .h5 and .hdf5 files from root directory."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".h5") or fname.endswith(".hdf5"):
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def run_gauss_bonnet_test(root="."):
    print("=" * 70)
    print("  SERAPHIM GAUSS-BONNET IDENTITY TEST")
    print("  Prediction: beta + 2*E[chi_eff^2] = 1.000")
    print("  K_0 = 1.1467e84 Hz^2  (exponent = 84)")
    print("=" * 70)

    h5_files = find_h5_files(root)
    h5_files = [f for f in h5_files
                if "Summary" not in os.path.basename(f)
                and "Table" not in os.path.basename(f)]

    if not h5_files:
        print("[!] No HDF5 files found under: " + os.path.abspath(root))
        return

    print("[*] Found " + str(len(h5_files)) + " HDF5 files (including subfolders)")
    print("")

    results = []
    skipped = []

    for filepath in h5_files:
        filename  = os.path.basename(filepath)
        subfolder = os.path.relpath(os.path.dirname(filepath), root)
        label     = (subfolder + "/" + filename) if subfolder != "." else filename

        try:
            with h5py.File(filepath, "r") as f:
                ps, key_used, event_type = get_posteriors(f)

                if ps is None:
                    skipped.append(label + " (no posterior_samples)")
                    continue

                m1      = get_col(ps, MASS1_NAMES)
                m2      = get_col(ps, MASS2_NAMES)
                m_final = get_col(ps, MFINAL_NAMES)
                chi_eff = get_col(ps, CHIEFF_NAMES)

                if m1 is None or m2 is None or m_final is None:
                    skipped.append(label + " (missing mass columns in " + key_used + ")")
                    continue

                if chi_eff is None:
                    skipped.append(label + " (no chi_eff column)")
                    continue

                m_total = m1 + m2
                n_array, valid_idx = calculate_octave(m_total, m_final)

                if len(n_array) < 50:
                    skipped.append(label + " (only " + str(len(n_array)) + " valid octave samples)")
                    continue

                # Align chi_eff to same valid rows as n_array
                chi_valid = chi_eff[valid_idx]
                n_len     = len(n_array)
                chi_trim  = chi_valid[:n_len]

                # E[chi_eff^2]: proper expectation value over full posterior
                # NOT median(chi_eff)^2 — that introduces ~+0.13 bias
                e_chi2 = float(np.mean(chi_trim ** 2))

                # Gauss-Bonnet identity value per event
                gb_value = BETA_OBS + 2.0 * e_chi2

                # Also compute with β derived per-event from n (diagnostic)
                # β = (n_BBH - n_obs) / log2(0.5 / C)  — but C is not in posteriors
                # so we report the fixed β_obs = 0.814 version as the primary test
                # and a secondary version using per-event n deviation as proxy

                median_n   = float(np.median(n_array))
                mean_n     = float(np.mean(n_array))
                std_n      = float(np.std(n_array))
                median_chi = float(np.median(chi_trim))
                n_samples  = int(len(n_array))
                in_band    = int(4.76 <= median_n <= 5.76)

                # Delta from identity: how far from 1.0
                delta_from_unity = gb_value - 1.0

                row = {
                    "event_file":         label,
                    "event_type":         event_type,
                    "waveform_key":       key_used,
                    "n_samples":          n_samples,
                    "median_n":           round(median_n,   5),
                    "mean_n":             round(mean_n,     5),
                    "std_n":              round(std_n,      5),
                    "in_bbh_band":        in_band,
                    "median_chi_eff":     round(median_chi, 5),
                    "E_chi_eff_sq":       round(e_chi2,     6),
                    "beta_fixed":         BETA_OBS,
                    "GB_value":           round(gb_value,   6),
                    "delta_from_unity":   round(delta_from_unity, 6),
                }
                results.append(row)

                tag = "[BBH]" if in_band else "[OUT]"
                print("[+] " + tag + " " + label[:50].ljust(50) +
                      "  E[X2]=" + str(round(e_chi2, 4)) +
                      "  GB="    + str(round(gb_value, 4)) +
                      "  d="     + str(round(delta_from_unity, 4)))

        except Exception as e:
            skipped.append(label + " (error: " + str(e) + ")")

    if not results:
        print("[!] No events processed successfully.")
        return

    # -----------------------------------------------------------------------
    # POPULATION STATISTICS
    # -----------------------------------------------------------------------
    gb_all    = np.array([r["GB_value"]       for r in results])
    gb_bbh    = np.array([r["GB_value"]       for r in results if r["in_bbh_band"]])
    echi2_all = np.array([r["E_chi_eff_sq"]   for r in results])
    echi2_bbh = np.array([r["E_chi_eff_sq"]   for r in results if r["in_bbh_band"]])

    n_total = len(results)
    n_bbh   = len(gb_bbh)

    print("")
    print("=" * 70)
    print("  POPULATION RESULTS")
    print("=" * 70)
    print("")
    print("  Events processed:         " + str(n_total))
    print("  Events in BBH band:       " + str(n_bbh))
    print("")
    print("  --- ALL EVENTS ---")
    print("  Mean E[chi_eff^2]:        " + str(round(float(np.mean(echi2_all)), 6)))
    print("  Mean GB value:            " + str(round(float(np.mean(gb_all)),    6)))
    print("  Std  GB value:            " + str(round(float(np.std(gb_all)),     6)))
    print("  GB 95% CI:                [" +
          str(round(float(np.percentile(gb_all, 2.5)),  4)) + ", " +
          str(round(float(np.percentile(gb_all, 97.5)), 4)) + "]")
    print("  Delta from 1.000:         " + str(round(float(np.mean(gb_all)) - 1.0, 6)))
    print("")

    if n_bbh > 0:
        print("  --- BBH BAND ONLY (4.76 <= n <= 5.76) ---")
        print("  Mean E[chi_eff^2]:        " + str(round(float(np.mean(echi2_bbh)), 6)))
        print("  Mean GB value:            " + str(round(float(np.mean(gb_bbh)),    6)))
        print("  Std  GB value:            " + str(round(float(np.std(gb_bbh)),     6)))
        print("  GB 95% CI:                [" +
              str(round(float(np.percentile(gb_bbh, 2.5)),  4)) + ", " +
              str(round(float(np.percentile(gb_bbh, 97.5)), 4)) + "]")
        print("  Delta from 1.000:         " + str(round(float(np.mean(gb_bbh)) - 1.0, 6)))
        print("")

    # Implied beta needed for exact unity, given observed E[chi_eff^2]
    if n_bbh > 0:
        mean_e2 = float(np.mean(echi2_bbh))
    else:
        mean_e2 = float(np.mean(echi2_all))
    beta_for_unity = 1.0 - 2.0 * mean_e2
    print("  Implied beta for GB=1.000: " + str(round(beta_for_unity, 6)))
    print("  Current beta_obs:          " + str(BETA_OBS))
    print("  Gap (beta_needed - beta_obs): " + str(round(beta_for_unity - BETA_OBS, 6)))
    print("")

    # Fraction of events with GB value within 0.05 of 1.0
    frac_near = float(np.mean(np.abs(gb_all - 1.0) < 0.05)) * 100
    print("  Fraction with |GB - 1.0| < 0.05:  " + str(round(frac_near, 1)) + "%")
    frac_near2 = float(np.mean(np.abs(gb_all - 1.0) < 0.10)) * 100
    print("  Fraction with |GB - 1.0| < 0.10:  " + str(round(frac_near2, 1)) + "%")
    print("")

    if skipped:
        print("  --- SKIPPED (" + str(len(skipped)) + " files) ---")
        for s in skipped:
            print("  [-] " + s)
        print("")

    # -----------------------------------------------------------------------
    # OUTPUTS
    # -----------------------------------------------------------------------
    csv_path  = "seraphim_gauss_bonnet_results.csv"
    json_path = "seraphim_gauss_bonnet_summary.json"

    with open(csv_path, "w", newline="") as cf:
        if results:
            writer = csv.DictWriter(cf, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "test": "Gauss-Bonnet Identity: beta + 2*E[chi_eff^2] = 1.000",
        "K_0_exponent_verify": "84 (critical — must be 1.1467e84, not 1.1467e4)",
        "beta_obs_fixed": BETA_OBS,
        "prediction": 1.0,
        "events_total": n_total,
        "events_bbh_band": n_bbh,
        "all_events": {
            "mean_E_chi_eff_sq": round(float(np.mean(echi2_all)), 6),
            "mean_GB":           round(float(np.mean(gb_all)),    6),
            "std_GB":            round(float(np.std(gb_all)),     6),
            "ci_95":             [round(float(np.percentile(gb_all, 2.5)), 4),
                                  round(float(np.percentile(gb_all, 97.5)), 4)],
            "delta_from_unity":  round(float(np.mean(gb_all)) - 1.0, 6),
        },
        "bbh_band_only": {
            "mean_E_chi_eff_sq": round(float(np.mean(echi2_bbh)), 6) if n_bbh > 0 else None,
            "mean_GB":           round(float(np.mean(gb_bbh)),    6) if n_bbh > 0 else None,
            "std_GB":            round(float(np.std(gb_bbh)),     6) if n_bbh > 0 else None,
            "ci_95":             [round(float(np.percentile(gb_bbh, 2.5)), 4),
                                  round(float(np.percentile(gb_bbh, 97.5)), 4)] if n_bbh > 0 else None,
            "delta_from_unity":  round(float(np.mean(gb_bbh)) - 1.0, 6) if n_bbh > 0 else None,
        },
        "implied_beta_for_unity": round(beta_for_unity, 6),
        "gap_to_beta_obs":        round(beta_for_unity - BETA_OBS, 6),
    }

    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=4)

    print("[*] Per-event CSV:  " + csv_path)
    print("[*] Summary JSON:   " + json_path)
    print("[*] Done.")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    run_gauss_bonnet_test(root)
