import os
import glob
import csv
import json
import numpy as np
import h5py
from scipy import stats

# ==============================================================================
# SERAPHIM n DISTRIBUTION SHAPE TEST
# Tests whether the octave depth n distribution in the BBH band is Gaussian.
# Uses: Shapiro-Wilk, D'Agostino-Pearson (K^2), Anderson-Darling
# Also reports: skewness, kurtosis, tail analysis, per-catalog breakdown
#
# Physical interpretation:
#   Gaussian -> clean single-mode prediction, framework producing sharp geometry
#   Skewed low -> NSBH contamination pulling tail below band
#   Skewed high -> redshift or mass-ratio systematics
#   Bimodal -> possible two populations (would be a significant finding)
#
# K_0 = 1.1467e84 Hz^2  (CRITICAL: exponent is 84, not 4)
# ==============================================================================

K_0       = 1.1467e84
NU_PLANCK = 1.8549e43
ALPHA     = 0.007297
J_SPIN    = 0.5
SQRT_J    = np.sqrt(J_SPIN * (J_SPIN + 1.0))

N_BBH_PRED = 5.314
N_BAND_LO  = 4.76
N_BAND_HI  = 5.76

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
    for key in f.keys():
        if key.startswith("C01") and "XPHM" in key:
            try:
                return f[key]["posterior_samples"], key, "BBH"
            except Exception:
                pass
    for key in f.keys():
        if "XPHM" in key and "Tidal" not in key and "NSBH" not in key:
            try:
                return f[key]["posterior_samples"], key, "BBH"
            except Exception:
                pass
    for key in f.keys():
        if "XPHM" in key:
            try:
                return f[key]["posterior_samples"], key, "BBH_mixed"
            except Exception:
                pass
    for key in f.keys():
        if key in ("history", "version"):
            continue
        if "NSBH" in key or "Tidal" in key or "NRTidal" in key:
            try:
                return f[key]["posterior_samples"], key, "NSBH"
            except Exception:
                pass
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
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".h5") or fname.endswith(".hdf5"):
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def infer_catalog(filepath):
    """Infer catalog from path or filename."""
    p = filepath.upper()
    if "6513631" in p or "GWTC-2.1" in p or "GWTC2.1" in p or "2.1" in p:
        return "GWTC-2.1"
    if "8177023" in p or "GWTC-3" in p or "GWTC3" in p:
        return "GWTC-3"
    if "16053484" in p or "GWTC-4" in p or "GWTC4" in p:
        return "GWTC-4"
    return "UNKNOWN"


def normality_report(label, arr):
    """Run all three normality tests and return a results dict."""
    n = len(arr)
    result = {
        "label":    label,
        "n_events": n,
        "mean":     round(float(np.mean(arr)),     5),
        "median":   round(float(np.median(arr)),   5),
        "std":      round(float(np.std(arr)),      5),
        "skewness": round(float(stats.skew(arr)),  5),
        "kurtosis": round(float(stats.kurtosis(arr)), 5),  # excess kurtosis (0 = Gaussian)
    }

    # Shapiro-Wilk (best for n < 5000; if larger, subsample)
    sw_arr = arr if n <= 5000 else np.random.choice(arr, 5000, replace=False)
    sw_stat, sw_p = stats.shapiro(sw_arr)
    result["shapiro_W"]  = round(float(sw_stat), 6)
    result["shapiro_p"]  = float(sw_p)
    result["shapiro_gaussian"] = "YES" if sw_p > 0.05 else "NO"
    if n > 5000:
        result["shapiro_note"] = "subsampled to 5000"

    # D'Agostino-Pearson K^2 (combines skew + kurtosis)
    dp_stat, dp_p = stats.normaltest(arr)
    result["dagostino_K2"] = round(float(dp_stat), 6)
    result["dagostino_p"]  = float(dp_p)
    result["dagostino_gaussian"] = "YES" if dp_p > 0.05 else "NO"

    # Anderson-Darling
    ad = stats.anderson(arr, dist="norm")
    # Report at 5% significance level (index 2)
    ad_stat  = float(ad.statistic)
    ad_crit5 = float(ad.critical_values[2])
    result["anderson_stat"]   = round(ad_stat, 6)
    result["anderson_crit5%"] = round(ad_crit5, 6)
    result["anderson_gaussian"] = "YES" if ad_stat < ad_crit5 else "NO"

    return result


def tail_analysis(label, arr):
    """Count events in tail regions around the BBH band."""
    n = len(arr)
    below_band  = int(np.sum(arr < N_BAND_LO))
    above_band  = int(np.sum(arr > N_BAND_HI))
    in_band     = int(np.sum((arr >= N_BAND_LO) & (arr <= N_BAND_HI)))
    near_pred   = int(np.sum(np.abs(arr - N_BBH_PRED) <= 0.05))  # within 0.05 oct of 5.314
    below_floor = int(np.sum(arr < 3.561))

    return {
        "label":            label,
        "n_events":         n,
        "in_bbh_band":      in_band,
        "pct_in_band":      round(in_band  / n * 100, 1),
        "below_band":       below_band,
        "above_band":       above_band,
        "within_0.05_of_5.314": near_pred,
        "pct_within_0.05":  round(near_pred / n * 100, 1),
        "below_n_flip_3.561": below_floor,
    }


def run_shape_test(root="."):
    print("=" * 70)
    print("  SERAPHIM n DISTRIBUTION SHAPE TEST")
    print("  Shapiro-Wilk + D'Agostino-Pearson + Anderson-Darling")
    print("  K_0 = 1.1467e84 Hz^2  (exponent = 84)")
    print("=" * 70)

    h5_files = find_h5_files(root)
    h5_files = [f for f in h5_files
                if "Summary" not in os.path.basename(f)
                and "Table"   not in os.path.basename(f)]

    if not h5_files:
        print("[!] No HDF5 files found under: " + os.path.abspath(root))
        return

    print("[*] Found " + str(len(h5_files)) + " HDF5 files")
    print("")

    per_event  = []   # one row per event: median_n, catalog, event_type
    skipped    = []
    catalog_ns = {"GWTC-2.1": [], "GWTC-3": [], "GWTC-4": [], "UNKNOWN": []}

    for filepath in h5_files:
        filename  = os.path.basename(filepath)
        subfolder = os.path.relpath(os.path.dirname(filepath), root)
        label     = (subfolder + "/" + filename) if subfolder != "." else filename
        catalog   = infer_catalog(filepath)

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
                    skipped.append(label + " (missing mass columns)")
                    continue

                m_total = m1 + m2
                n_array, valid_idx = calculate_octave(m_total, m_final)

                if len(n_array) < 50:
                    skipped.append(label + " (" + str(len(n_array)) + " valid samples)")
                    continue

                median_n   = float(np.median(n_array))
                std_n      = float(np.std(n_array))
                in_band    = int(N_BAND_LO <= median_n <= N_BAND_HI)
                median_chi = float(np.median(chi_eff[valid_idx][:len(n_array)])) if chi_eff is not None else -99.0

                row = {
                    "event_file":    label,
                    "catalog":       catalog,
                    "event_type":    event_type,
                    "waveform_key":  key_used,
                    "median_n":      round(median_n, 5),
                    "std_n":         round(std_n,    5),
                    "in_bbh_band":   in_band,
                    "median_chi_eff": round(median_chi, 5),
                    "n_samples":     int(len(n_array)),
                }
                per_event.append(row)
                catalog_ns[catalog].append(median_n)

                tag = "[BBH]" if in_band else "[OUT]"
                print("[+] " + tag + " " + label[:50].ljust(50) +
                      "  n=" + str(round(median_n, 3)) +
                      "  cat=" + catalog)

        except Exception as e:
            skipped.append(label + " (error: " + str(e) + ")")

    if not per_event:
        print("[!] No events processed.")
        return

    all_n   = np.array([r["median_n"] for r in per_event])
    bbh_n   = np.array([r["median_n"] for r in per_event if r["in_bbh_band"]])

    print("")
    print("=" * 70)
    print("  NORMALITY TEST RESULTS")
    print("=" * 70)

    norm_results  = []
    tail_results  = []

    def print_norm(res):
        print("")
        print("  [" + res["label"] + "]  N=" + str(res["n_events"]))
        print("  Mean=" + str(res["mean"]) +
              "  Median=" + str(res["median"]) +
              "  Std=" + str(res["std"]))
        print("  Skewness=" + str(res["skewness"]) +
              "  Excess Kurtosis=" + str(res["kurtosis"]))
        note = res.get("shapiro_note", "")
        print("  Shapiro-Wilk:      W=" + str(res["shapiro_W"]) +
              "  p=" + "{:.4e}".format(res["shapiro_p"]) +
              "  Gaussian=" + res["shapiro_gaussian"] +
              (" (" + note + ")" if note else ""))
        print("  D'Agostino K^2:    stat=" + str(res["dagostino_K2"]) +
              "  p=" + "{:.4e}".format(res["dagostino_p"]) +
              "  Gaussian=" + res["dagostino_gaussian"])
        print("  Anderson-Darling:  stat=" + str(res["anderson_stat"]) +
              "  crit(5%)=" + str(res["anderson_crit5%"]) +
              "  Gaussian=" + res["anderson_gaussian"])

    # All events
    if len(all_n) >= 8:
        r = normality_report("ALL EVENTS", all_n)
        norm_results.append(r)
        print_norm(r)
        tail_results.append(tail_analysis("ALL EVENTS", all_n))

    # BBH band only
    if len(bbh_n) >= 8:
        r = normality_report("BBH BAND ONLY (4.76-5.76)", bbh_n)
        norm_results.append(r)
        print_norm(r)
        tail_results.append(tail_analysis("BBH BAND ONLY", bbh_n))

    # Per catalog
    for cat in ["GWTC-2.1", "GWTC-3", "GWTC-4"]:
        ns = np.array(catalog_ns[cat])
        if len(ns) >= 8:
            r = normality_report(cat + " ALL", ns)
            norm_results.append(r)
            print_norm(r)
            # BBH band subset per catalog
            ns_bbh = ns[(ns >= N_BAND_LO) & (ns <= N_BAND_HI)]
            if len(ns_bbh) >= 8:
                r2 = normality_report(cat + " BBH BAND", ns_bbh)
                norm_results.append(r2)
                print_norm(r2)
            tail_results.append(tail_analysis(cat, ns))

    # -----------------------------------------------------------------------
    # TAIL + BAND POPULATION SUMMARY
    # -----------------------------------------------------------------------
    print("")
    print("=" * 70)
    print("  BAND POPULATION SUMMARY")
    print("=" * 70)
    print("")
    for t in tail_results:
        print("  [" + t["label"] + "]  N=" + str(t["n_events"]))
        print("  In BBH band:         " + str(t["in_bbh_band"]) +
              " (" + str(t["pct_in_band"]) + "%)")
        print("  Below band (<4.76):  " + str(t["below_band"]))
        print("  Above band (>5.76):  " + str(t["above_band"]))
        print("  Within 0.05 of 5.314:" + str(t["within_0.05_of_5.314"]) +
              " (" + str(t["pct_within_0.05"]) + "%)")
        print("  Below n_flip (3.561):" + str(t["below_n_flip_3.561"]))
        print("")

    # Physical interpretation
    print("=" * 70)
    print("  PHYSICAL INTERPRETATION")
    print("=" * 70)
    if len(bbh_n) >= 8:
        sk = float(stats.skew(bbh_n))
        ku = float(stats.kurtosis(bbh_n))
        print("")
        if abs(sk) < 0.3:
            print("  Skewness ~ 0: distribution is symmetric in BBH band.")
            print("  No evidence of systematic contamination from one mass regime.")
        elif sk < -0.3:
            print("  Skewness < 0: left tail present — possible NSBH/low-mass contamination")
            print("  pulling events below BBH band. Check events near n ~ 4.76.")
        else:
            print("  Skewness > 0: right tail present — check for redshift or high-mass outliers.")
        print("")
        if abs(ku) < 0.5:
            print("  Kurtosis ~ 0: normal tail weight. Consistent with single Gaussian mode.")
        elif ku > 0.5:
            print("  Excess kurtosis > 0: heavier tails than Gaussian.")
            print("  Possible two-population structure — worth inspecting with a KDE plot.")
        else:
            print("  Excess kurtosis < 0: lighter tails than Gaussian (platykurtic).")

    if skipped:
        print("")
        print("  --- SKIPPED (" + str(len(skipped)) + " files) ---")
        for s in skipped:
            print("  [-] " + s)

    # -----------------------------------------------------------------------
    # OUTPUTS
    # -----------------------------------------------------------------------
    csv_events  = "seraphim_shape_per_event.csv"
    csv_norm    = "seraphim_shape_normality.csv"
    csv_tail    = "seraphim_shape_tail.csv"
    json_path   = "seraphim_shape_summary.json"

    with open(csv_events, "w", newline="") as cf:
        if per_event:
            writer = csv.DictWriter(cf, fieldnames=per_event[0].keys())
            writer.writeheader()
            writer.writerows(per_event)

    with open(csv_norm, "w", newline="") as cf:
        if norm_results:
            writer = csv.DictWriter(cf, fieldnames=norm_results[0].keys())
            writer.writeheader()
            writer.writerows(norm_results)

    with open(csv_tail, "w", newline="") as cf:
        if tail_results:
            writer = csv.DictWriter(cf, fieldnames=tail_results[0].keys())
            writer.writeheader()
            writer.writerows(tail_results)

    summary = {
        "test": "n Distribution Shape Test",
        "K_0_exponent_verify": "84 (must be 1.1467e84)",
        "N_BBH_predicted": N_BBH_PRED,
        "bbh_band": [N_BAND_LO, N_BAND_HI],
        "events_total": len(per_event),
        "events_bbh_band": int(len(bbh_n)),
        "normality_results": norm_results,
        "tail_results": tail_results,
    }

    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=4)

    print("")
    print("[*] Per-event CSV:    " + csv_events)
    print("[*] Normality CSV:    " + csv_norm)
    print("[*] Tail summary CSV: " + csv_tail)
    print("[*] Summary JSON:     " + json_path)
    print("[*] Done.")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    run_shape_test(root)
