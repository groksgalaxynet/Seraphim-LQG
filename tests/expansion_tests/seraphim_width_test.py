import os
import csv
import json
import numpy as np
import h5py
from scipy import stats

# ==============================================================================
# SERAPHIM TEST 4: POSTERIOR WIDTH vs OCTAVE DEPTH SCATTER
#
# Question: Do events with n far from 5.314 also have wider posteriors?
#   - If yes: outliers are poorly constrained, not genuinely anomalous
#   - If no: outliers are real departures, framework structure is clean
#
# Tests:
#   (a) Scatter of sigma_n vs median_n -- Pearson + Spearman correlation
#   (b) Width comparison: BBH band events vs out-of-band events
#   (c) Width comparison: high-spin vs low-spin events
#   (d) Coefficient of variation (CV) per event -- is prediction tight?
#   (e) Sub-floor events: are they wide or narrow? (key question for GW191219)
#   (f) Per-catalog width consistency
#
# Interpretation key:
#   r(sigma_n, |n - 5.314|) >> 0 -> outliers are wide -> posterior noise
#   r(sigma_n, |n - 5.314|) ~ 0  -> outliers are narrow -> genuine signal
#   BBH width << out-of-band width -> band is geometrically preferred
#   BBH width ~ out-of-band width  -> all events equally uncertain
#
# K_0 = 1.1467e84 Hz^2  (CRITICAL: exponent is 84, not 4)
# ==============================================================================

K_0        = 1.1467e84
NU_PLANCK  = 1.8549e43
ALPHA      = 0.007297
J_SPIN     = 0.5
SQRT_J     = np.sqrt(J_SPIN * (J_SPIN + 1.0))

N_BBH      = 5.314
N_FLIP     = 3.561
N_BAND_LO  = 4.76
N_BAND_HI  = 5.76

MASS1_NAMES  = ["mass_1_source", "mass_1"]
MASS2_NAMES  = ["mass_2_source", "mass_2"]
MFINAL_NAMES = ["final_mass_source", "final_mass",
                "final_mass_source_non_evolved", "final_mass_non_evolved"]
CHIEFF_NAMES = ["chi_eff"]
Q_NAMES      = ["mass_ratio"]


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
                if "Summary" not in fname and "Table" not in fname:
                    found.append(os.path.join(dirpath, fname))
    return sorted(found)


def infer_catalog(filepath):
    p = filepath.upper()
    if "6513631" in p: return "GWTC-2.1"
    if "8177023" in p: return "GWTC-3"
    if "16053484" in p or "GWTC4P0" in p or "GWTC4" in p: return "GWTC-4"
    bn = os.path.basename(filepath).upper()
    if "GWTC4P0" in bn or "GWTC-4" in bn: return "GWTC-4"
    return "UNKNOWN"


def corr_report(x, y, label_x, label_y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return None
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    return {
        "comparison":  label_x + " vs " + label_y,
        "n":           len(x),
        "pearson_r":   round(float(r_p), 4),
        "pearson_p":   float(p_p),
        "spearman_r":  round(float(r_s), 4),
        "spearman_p":  float(p_s),
        "significant": (p_p < 0.05 or p_s < 0.05),
    }


def group_stats(arr, label):
    if len(arr) == 0:
        return None
    return {
        "label":   label,
        "n":       len(arr),
        "mean":    round(float(np.mean(arr)),   5),
        "median":  round(float(np.median(arr)), 5),
        "std":     round(float(np.std(arr)),    5),
        "p25":     round(float(np.percentile(arr, 25)), 5),
        "p75":     round(float(np.percentile(arr, 75)), 5),
    }


def run_width_test(root="."):
    print("=" * 70)
    print("  SERAPHIM TEST 4: POSTERIOR WIDTH vs OCTAVE DEPTH")
    print("  K_0 = 1.1467e84 Hz^2  (exponent = 84)")
    print("=" * 70)

    h5_files = find_h5_files(root)
    if not h5_files:
        print("[!] No HDF5 files found under: " + os.path.abspath(root))
        return

    print("[*] Found " + str(len(h5_files)) + " HDF5 files")
    print()

    rows    = []
    skipped = []

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
                q_col   = get_col(ps, Q_NAMES)

                if m1 is None or m2 is None or m_final is None:
                    skipped.append(label + " (missing mass columns)")
                    continue

                m_total = m1 + m2
                n_array, valid_idx = calculate_octave(m_total, m_final)

                if len(n_array) < 50:
                    skipped.append(label + " (" + str(len(n_array)) + " valid samples)")
                    continue

                median_n  = float(np.median(n_array))
                mean_n    = float(np.mean(n_array))
                std_n     = float(np.std(n_array))
                p5_n      = float(np.percentile(n_array, 5))
                p95_n     = float(np.percentile(n_array, 95))
                ci90_width = p95_n - p5_n
                cv_n      = (std_n / abs(mean_n)) * 100 if mean_n != 0 else np.nan
                dev_pred  = abs(median_n - N_BBH)
                in_band   = int(N_BAND_LO <= median_n <= N_BAND_HI)
                sub_floor = int(median_n < N_FLIP)

                med_chi = float(np.median(chi_eff[valid_idx][:len(n_array)])) if chi_eff is not None else np.nan
                med_q   = float(np.median(q_col[valid_idx][:len(n_array)]))   if q_col   is not None else np.nan

                row = {
                    "event_file":    label,
                    "catalog":       catalog,
                    "event_type":    event_type,
                    "waveform_key":  key_used,
                    "n_samples":     int(len(n_array)),
                    "median_n":      round(median_n,   5),
                    "mean_n":        round(mean_n,     5),
                    "std_n":         round(std_n,      5),
                    "p5_n":          round(p5_n,       5),
                    "p95_n":         round(p95_n,      5),
                    "ci90_width":    round(ci90_width, 5),
                    "cv_pct":        round(cv_n,       4),
                    "dev_from_pred": round(dev_pred,   5),
                    "in_bbh_band":   in_band,
                    "sub_floor":     sub_floor,
                    "median_chi_eff":round(med_chi, 5) if np.isfinite(med_chi) else None,
                    "median_q":      round(med_q,   5) if np.isfinite(med_q)   else None,
                }
                rows.append(row)

                tag = "[BBH]" if in_band else ("[FLOOR]" if sub_floor else "[OUT]")
                print("[+] " + tag + " " + label[:48].ljust(48) +
                      "  n=" + str(round(median_n, 3)).rjust(6) +
                      "  σ=" + str(round(std_n, 4)).rjust(7))

        except Exception as e:
            skipped.append(label + " (error: " + str(e) + ")")

    if not rows:
        print("[!] No events processed.")
        return

    # -----------------------------------------------------------------------
    # ARRAYS FOR ANALYSIS
    # -----------------------------------------------------------------------
    all_median = np.array([r["median_n"]      for r in rows])
    all_std    = np.array([r["std_n"]         for r in rows])
    all_ci90   = np.array([r["ci90_width"]    for r in rows])
    all_dev    = np.array([r["dev_from_pred"] for r in rows])
    all_cv     = np.array([r["cv_pct"]        for r in rows if r["cv_pct"] is not None])

    bbh_rows   = [r for r in rows if r["in_bbh_band"]]
    out_rows   = [r for r in rows if not r["in_bbh_band"] and not r["sub_floor"]]
    floor_rows = [r for r in rows if r["sub_floor"]]
    hi_spin    = [r for r in rows if r["median_chi_eff"] is not None and r["median_chi_eff"] >  0.3]
    lo_spin    = [r for r in rows if r["median_chi_eff"] is not None and abs(r["median_chi_eff"]) < 0.05]

    print()
    print("=" * 70)
    print("  CORRELATION: sigma_n vs deviation from prediction")
    print("=" * 70)
    print()

    corr_results = []

    # Core question: does width predict deviation?
    c1 = corr_report(all_std, all_dev, "sigma_n", "|n - 5.314|")
    if c1:
        corr_results.append(c1)
        print("  sigma_n vs |n - 5.314| (ALL events):")
        print("    Pearson  r=" + str(c1["pearson_r"])  + "  p=" + "{:.3e}".format(c1["pearson_p"]))
        print("    Spearman r=" + str(c1["spearman_r"]) + "  p=" + "{:.3e}".format(c1["spearman_p"]))
        if c1["significant"]:
            print("    SIGNIFICANT: wider posteriors correlate with larger deviations")
            print("    -> Some outliers may be poorly constrained rather than genuine")
        else:
            print("    NOT SIGNIFICANT: posterior width does not predict deviation")
            print("    -> Outliers are genuinely anomalous, not just noisy measurements")
        print()

    # Width vs n value directly
    c2 = corr_report(all_std, all_median, "sigma_n", "median_n")
    if c2:
        corr_results.append(c2)
        print("  sigma_n vs median_n (ALL events):")
        print("    Pearson  r=" + str(c2["pearson_r"])  + "  p=" + "{:.3e}".format(c2["pearson_p"]))
        print("    Spearman r=" + str(c2["spearman_r"]) + "  p=" + "{:.3e}".format(c2["spearman_p"]))
        print()

    # BBH band only: width vs position within band
    if len(bbh_rows) > 5:
        bbh_std = np.array([r["std_n"]    for r in bbh_rows])
        bbh_med = np.array([r["median_n"] for r in bbh_rows])
        bbh_dev = np.array([r["dev_from_pred"] for r in bbh_rows])
        c3 = corr_report(bbh_std, bbh_dev, "sigma_n", "|n-5.314| BBH only")
        if c3:
            corr_results.append(c3)
            print("  sigma_n vs |n - 5.314| (BBH BAND ONLY):")
            print("    Pearson  r=" + str(c3["pearson_r"])  + "  p=" + "{:.3e}".format(c3["pearson_p"]))
            print("    Spearman r=" + str(c3["spearman_r"]) + "  p=" + "{:.3e}".format(c3["spearman_p"]))
            if not c3["significant"]:
                print("    NOT SIGNIFICANT: within BBH band, width is independent of position")
                print("    -> The band is uniformly constrained, not a wide smear")
            print()

    # -----------------------------------------------------------------------
    # GROUP WIDTH COMPARISONS
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  GROUP WIDTH COMPARISONS")
    print("=" * 70)
    print()

    groups = [
        ("BBH band (4.76-5.76)",        bbh_rows),
        ("Out-of-band, above floor",     out_rows),
        ("Sub-floor (n < 3.561)",        floor_rows),
        ("High spin (chi_eff > 0.3)",    hi_spin),
        ("Near-zero spin (|chi| < 0.05)",lo_spin),
    ]

    group_stats_list = []
    width_rows_for_ttest = {}

    for label, grp in groups:
        if not grp:
            continue
        std_arr = np.array([r["std_n"]      for r in grp])
        ci_arr  = np.array([r["ci90_width"] for r in grp])
        cv_arr  = np.array([r["cv_pct"]     for r in grp if r["cv_pct"] is not None])
        gs = group_stats(std_arr, label)
        if gs:
            gs["mean_ci90_width"] = round(float(np.mean(ci_arr)), 5)
            gs["mean_cv_pct"]     = round(float(np.mean(cv_arr)), 4) if len(cv_arr) else None
            group_stats_list.append(gs)
            width_rows_for_ttest[label] = std_arr
            print("  [" + label + "]  N=" + str(gs["n"]))
            print("    Mean σ_n:       " + str(gs["mean"]))
            print("    Median σ_n:     " + str(gs["median"]))
            print("    Mean CI90 width:" + str(gs["mean_ci90_width"]))
            print("    Mean CV (%):    " + str(gs["mean_cv_pct"]))
            print()

    # T-test: BBH vs out-of-band width
    if "BBH band (4.76-5.76)" in width_rows_for_ttest and \
       "Out-of-band, above floor" in width_rows_for_ttest:
        a = width_rows_for_ttest["BBH band (4.76-5.76)"]
        b = width_rows_for_ttest["Out-of-band, above floor"]
        t_stat, t_p = stats.ttest_ind(a, b)
        print("  T-test BBH vs out-of-band sigma_n:")
        print("    t=" + str(round(float(t_stat), 3)) + "  p=" + "{:.3e}".format(float(t_p)))
        if t_p < 0.05:
            direction = "BBH band is NARROWER" if np.mean(a) < np.mean(b) else "BBH band is WIDER"
            print("    SIGNIFICANT: " + direction + " than out-of-band events")
        else:
            print("    Not significant: BBH and out-of-band events have similar posterior widths")
        print()

    # -----------------------------------------------------------------------
    # CV DISTRIBUTION IN BBH BAND
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  COEFFICIENT OF VARIATION IN BBH BAND")
    print("=" * 70)
    print()
    if bbh_rows:
        cv_bbh = np.array([r["cv_pct"] for r in bbh_rows if r["cv_pct"] is not None])
        print("  Mean CV:   " + str(round(float(np.mean(cv_bbh)), 3)) + "%")
        print("  Median CV: " + str(round(float(np.median(cv_bbh)), 3)) + "%")
        print("  Std CV:    " + str(round(float(np.std(cv_bbh)), 3)) + "%")
        print("  Max CV:    " + str(round(float(np.max(cv_bbh)), 3)) + "%")
        print("  Min CV:    " + str(round(float(np.min(cv_bbh)), 3)) + "%")
        print()
        tight = np.sum(cv_bbh < 2.0)
        print("  Events with CV < 2%:  " + str(tight) + " (" +
              str(round(tight/len(cv_bbh)*100, 1)) + "%) -- tightly constrained")
        very_tight = np.sum(cv_bbh < 1.0)
        print("  Events with CV < 1%:  " + str(very_tight) + " (" +
              str(round(very_tight/len(cv_bbh)*100, 1)) + "%)")
        wide = np.sum(cv_bbh > 5.0)
        print("  Events with CV > 5%:  " + str(wide) + " (" +
              str(round(wide/len(cv_bbh)*100, 1)) + "%) -- poorly constrained outliers")
        print()

    # -----------------------------------------------------------------------
    # WIDEST AND NARROWEST EVENTS
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  10 NARROWEST EVENTS (most tightly constrained)")
    print("=" * 70)
    narrowest = sorted(rows, key=lambda r: r["std_n"])[:10]
    for r in narrowest:
        import re
        m = re.search(r'(GW\d{6}_\d{6})', r["event_file"])
        eid = m.group(1) if m else r["event_file"][-28:]
        tag = "[BBH]" if r["in_bbh_band"] else ("[FLOOR]" if r["sub_floor"] else "[OUT]")
        print("  " + tag + " " + eid.ljust(24) +
              "  n=" + str(r["median_n"]).rjust(7) +
              "  σ=" + str(r["std_n"]).rjust(8) +
              "  CV=" + str(r["cv_pct"]).rjust(6) + "%")

    print()
    print("=" * 70)
    print("  10 WIDEST EVENTS (least constrained)")
    print("=" * 70)
    widest = sorted(rows, key=lambda r: r["std_n"], reverse=True)[:10]
    for r in widest:
        import re
        m = re.search(r'(GW\d{6}_\d{6})', r["event_file"])
        eid = m.group(1) if m else r["event_file"][-28:]
        tag = "[BBH]" if r["in_bbh_band"] else ("[FLOOR]" if r["sub_floor"] else "[OUT]")
        print("  " + tag + " " + eid.ljust(24) +
              "  n=" + str(r["median_n"]).rjust(7) +
              "  σ=" + str(r["std_n"]).rjust(8) +
              "  CV=" + str(r["cv_pct"]).rjust(6) + "%")

    # -----------------------------------------------------------------------
    # PER-CATALOG WIDTH
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  PER-CATALOG WIDTH CONSISTENCY")
    print("=" * 70)
    print()
    for cat in ["GWTC-2.1", "GWTC-3", "GWTC-4"]:
        cr = [r for r in rows if r["catalog"] == cat]
        if not cr:
            continue
        std_all = np.array([r["std_n"] for r in cr])
        std_bbh = np.array([r["std_n"] for r in cr if r["in_bbh_band"]])
        print("  " + cat + "  N=" + str(len(cr)) + "  BBH=" + str(len(std_bbh)))
        print("    All  mean σ=" + str(round(float(np.mean(std_all)), 5)) +
              "  std σ=" + str(round(float(np.std(std_all)), 5)))
        if len(std_bbh):
            print("    BBH  mean σ=" + str(round(float(np.mean(std_bbh)), 5)) +
                  "  std σ=" + str(round(float(np.std(std_bbh)), 5)))
        print()

    # -----------------------------------------------------------------------
    # PHYSICAL INTERPRETATION
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  PHYSICAL INTERPRETATION")
    print("=" * 70)
    print()
    if c1:
        if not c1["significant"]:
            print("  PRIMARY RESULT: sigma_n is UNCORRELATED with deviation from prediction.")
            print("  Posterior width does not predict how far an event sits from n=5.314.")
            print("  Conclusion: out-of-band events are genuinely anomalous, not noisy.")
            print("  The framework's band boundary is a real geometric structure,")
            print("  not an artifact of measurement precision.")
        else:
            print("  CAUTION: sigma_n correlates with deviation (r=" + str(c1["pearson_r"]) + ").")
            print("  Some out-of-band events may be poorly constrained rather than genuine outliers.")
            print("  Check widest events above for specific cases to flag.")
    print()

    if floor_rows:
        floor_std = np.array([r["std_n"] for r in floor_rows])
        bbh_std_arr = np.array([r["std_n"] for r in bbh_rows]) if bbh_rows else np.array([])
        print("  SUB-FLOOR EVENTS (n < 3.561):  N=" + str(len(floor_rows)))
        print("  Mean sigma_n: " + str(round(float(np.mean(floor_std)), 5)))
        if len(bbh_std_arr):
            ratio = float(np.mean(floor_std)) / float(np.mean(bbh_std_arr))
            print("  BBH mean sigma_n: " + str(round(float(np.mean(bbh_std_arr)), 5)))
            print("  Ratio floor/BBH: " + str(round(ratio, 3)))
            if ratio < 1.5:
                print("  Sub-floor events are NOT wider than BBH events.")
                print("  -> Sub-floor anomaly is tightly constrained. GW191219 is a real outlier.")
            else:
                print("  Sub-floor events are wider than BBH events.")
                print("  -> Sub-floor anomaly may be posterior noise. Confirm with NR waveforms.")
    print()

    if skipped:
        print("  Skipped " + str(len(skipped)) + " files.")

    # -----------------------------------------------------------------------
    # OUTPUTS
    # -----------------------------------------------------------------------
    csv_path   = "seraphim_width_per_event.csv"
    corr_path  = "seraphim_width_correlations.csv"
    group_path = "seraphim_width_groups.csv"
    json_path  = "seraphim_width_summary.json"

    with open(csv_path, "w", newline="") as cf:
        if rows:
            writer = csv.DictWriter(cf, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with open(corr_path, "w", newline="") as cf:
        if corr_results:
            writer = csv.DictWriter(cf, fieldnames=corr_results[0].keys())
            writer.writeheader()
            writer.writerows(corr_results)

    with open(group_path, "w", newline="") as cf:
        if group_stats_list:
            writer = csv.DictWriter(cf, fieldnames=group_stats_list[0].keys())
            writer.writeheader()
            writer.writerows(group_stats_list)

    summary = {
        "test":         "Posterior Width vs Octave Depth Scatter",
        "K0_verify":    "1.1467e84 Hz^2 (exponent=84)",
        "n_events":     len(rows),
        "n_bbh_band":   len(bbh_rows),
        "n_out_band":   len(out_rows),
        "n_sub_floor":  len(floor_rows),
        "correlations": corr_results,
        "group_stats":  group_stats_list,
    }

    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=4)

    print("[*] Per-event CSV:    " + csv_path)
    print("[*] Correlations CSV: " + corr_path)
    print("[*] Group stats CSV:  " + group_path)
    print("[*] Summary JSON:     " + json_path)
    print("[*] Done.")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    run_width_test(root)
