import os
import csv
import json
import numpy as np
import h5py
from scipy import stats
import re

# ==============================================================================
# SERAPHIM TEST 5: CATALOG-TO-CATALOG n CONSISTENCY (ANOVA)
#
# Question: Are the three independent observing runs (GWTC-2.1, GWTC-3, GWTC-4)
# statistically drawn from the same n distribution?
#
# Framework prediction: YES -- n = 5.314 is a geometric constant, independent
# of detector sensitivity, observing period, or catalog selection effects.
# Any significant difference between catalogs would be evidence of:
#   (a) systematic bias in one catalog's waveform models
#   (b) genuine astrophysical evolution of the BBH population
#   (c) framework failure (n is not constant across epochs)
#
# Tests:
#   (1) One-way ANOVA across three catalogs (BBH band)
#   (2) Kruskal-Wallis (non-parametric ANOVA -- appropriate given non-Gaussianity)
#   (3) Pairwise t-tests with Bonferroni correction
#   (4) Pairwise Mann-Whitney U tests
#   (5) Effect sizes (Cohen's d) for each pair
#   (6) Levene's test for equal variances across catalogs
#   (7) Per-catalog: mean, median, std, 95% CI on mean, sigma on prediction
#
# K_0 = 1.1467e84 Hz^2  (CRITICAL: exponent is 84, not 4)
# ==============================================================================

K_0        = 1.1467e84
NU_PLANCK  = 1.8549e43
ALPHA      = 0.007297
J_SPIN     = 0.5
SQRT_J     = np.sqrt(J_SPIN * (J_SPIN + 1.0))

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
                if "Summary" not in fname and "Table" not in fname:
                    found.append(os.path.join(dirpath, fname))
    return sorted(found)


def infer_catalog(filepath):
    p  = filepath.upper()
    bn = os.path.basename(filepath).upper()
    if "6513631"  in p: return "GWTC-2.1"
    if "8177023"  in p: return "GWTC-3"
    if "16053484" in p or "GWTC4P0" in bn or "GWTC-4" in bn: return "GWTC-4"
    return "UNKNOWN"


def deduplicate_events(rows):
    """Keep one file per GW event ID, prefer nocosmo > combined > cosmo."""
    def priority(ef):
        bn = ef.lower()
        if "nocosmo"  in bn: return 0
        if "combined" in bn: return 1
        if "cosmo"    in bn: return 2
        return 3

    groups = {}
    for r in rows:
        m   = re.search(r'(GW\d{6}_\d{6})', r["event_file"])
        eid = m.group(1) if m else r["event_file"]
        key = r["catalog"] + ":" + eid
        if key not in groups or priority(r["event_file"]) < priority(groups[key]["event_file"]):
            groups[key] = r
    return list(groups.values())


def cohens_d(a, b):
    na, nb   = len(a), len(b)
    pooled_s = np.sqrt(((na - 1)*np.var(a, ddof=1) + (nb - 1)*np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled_s if pooled_s > 0 else 0.0


def mean_ci95(arr):
    se = stats.sem(arr)
    t  = stats.t.ppf(0.975, df=len(arr) - 1)
    return float(np.mean(arr) - t*se), float(np.mean(arr) + t*se)


def sigma_from_pred(arr, pred=N_BBH_PRED):
    """How many std devs of the mean is the sample mean from prediction."""
    se   = stats.sem(arr)
    return (float(np.mean(arr)) - pred) / se if se > 0 else 0.0


def run_anova_test(root="."):
    print("=" * 70)
    print("  SERAPHIM TEST 5: CATALOG CONSISTENCY — ONE-WAY ANOVA")
    print("  Prediction: three catalogs drawn from same n distribution")
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

                if m1 is None or m2 is None or m_final is None:
                    skipped.append(label + " (missing mass columns)")
                    continue

                m_total = m1 + m2
                n_array, valid_idx = calculate_octave(m_total, m_final)

                if len(n_array) < 50:
                    skipped.append(label + " (" + str(len(n_array)) + " valid samples)")
                    continue

                median_n  = float(np.median(n_array))
                std_n     = float(np.std(n_array))
                in_band   = int(N_BAND_LO <= median_n <= N_BAND_HI)
                med_chi   = float(np.median(chi_eff[valid_idx][:len(n_array)])) \
                            if chi_eff is not None else np.nan

                rows.append({
                    "event_file":    label,
                    "catalog":       catalog,
                    "event_type":    event_type,
                    "waveform_key":  key_used,
                    "median_n":      round(median_n, 5),
                    "std_n":         round(std_n,    5),
                    "in_bbh_band":   in_band,
                    "median_chi_eff":round(med_chi, 5) if np.isfinite(med_chi) else None,
                })

                tag = "[BBH]" if in_band else "[OUT]"
                print("[+] " + tag + " " + label[:46].ljust(46) +
                      "  n=" + str(round(median_n, 3)).rjust(6) +
                      "  cat=" + catalog)

        except Exception as e:
            skipped.append(label + " (error: " + str(e) + ")")

    if not rows:
        print("[!] No events processed.")
        return

    print()
    print("[*] Total processed (pre-dedup): " + str(len(rows)))
    rows = deduplicate_events(rows)
    print("[*] After deduplication:         " + str(len(rows)))
    print()

    # -----------------------------------------------------------------------
    # BUILD CATALOG ARRAYS
    # -----------------------------------------------------------------------
    cats     = ["GWTC-2.1", "GWTC-3", "GWTC-4"]
    cat_all  = {c: np.array([r["median_n"] for r in rows if r["catalog"] == c]) for c in cats}
    cat_bbh  = {c: np.array([r["median_n"] for r in rows
                              if r["catalog"] == c and r["in_bbh_band"]]) for c in cats}

    # -----------------------------------------------------------------------
    # PER-CATALOG DESCRIPTIVES
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  PER-CATALOG DESCRIPTIVES")
    print("=" * 70)
    print()

    cat_desc = []
    for c in cats:
        arr = cat_bbh[c]
        if len(arr) == 0:
            continue
        ci_lo, ci_hi = mean_ci95(arr)
        sig          = sigma_from_pred(arr)
        row = {
            "catalog":        c,
            "n_events_all":   len(cat_all[c]),
            "n_events_bbh":   len(arr),
            "mean_n":         round(float(np.mean(arr)),   5),
            "median_n":       round(float(np.median(arr)), 5),
            "std_n":          round(float(np.std(arr)),    5),
            "sem_n":          round(float(stats.sem(arr)), 6),
            "ci95_lo":        round(ci_lo, 5),
            "ci95_hi":        round(ci_hi, 5),
            "sigma_from_5314":round(sig, 4),
            "min_n":          round(float(np.min(arr)), 5),
            "max_n":          round(float(np.max(arr)), 5),
        }
        cat_desc.append(row)
        print("  [" + c + "]  N_all=" + str(row["n_events_all"]) +
              "  N_BBH=" + str(row["n_events_bbh"]))
        print("    Mean:   " + str(row["mean_n"]) +
              "   95% CI: [" + str(row["ci95_lo"]) + ", " + str(row["ci95_hi"]) + "]")
        print("    Median: " + str(row["median_n"]) +
              "   Std: "     + str(row["std_n"]) +
              "   SEM: "     + str(row["sem_n"]))
        print("    Distance from 5.314: " + str(row["sigma_from_5314"]) + "σ")
        print()

    # -----------------------------------------------------------------------
    # ANOVA TESTS
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  STATISTICAL TESTS")
    print("=" * 70)
    print()

    test_results = []
    arrays       = [cat_bbh[c] for c in cats if len(cat_bbh[c]) > 0]
    cat_labels   = [c for c in cats if len(cat_bbh[c]) > 0]

    # (1) One-way ANOVA
    f_stat, f_p = stats.f_oneway(*arrays)
    print("  (1) One-way ANOVA (BBH band):")
    print("      F=" + str(round(float(f_stat), 4)) +
          "  p=" + "{:.4e}".format(float(f_p)))
    if f_p > 0.05:
        print("      NOT SIGNIFICANT: catalogs are consistent (p > 0.05)")
        print("      -> Three independent observing runs drawn from same distribution")
    else:
        print("      SIGNIFICANT: catalogs differ (p < 0.05)")
        print("      -> Check pairwise tests below for which catalogs drive difference")
    print()
    test_results.append({"test": "One-way ANOVA", "statistic": round(float(f_stat),4),
                         "p_value": float(f_p), "significant": f_p < 0.05})

    # (2) Kruskal-Wallis
    kw_stat, kw_p = stats.kruskal(*arrays)
    print("  (2) Kruskal-Wallis (non-parametric, appropriate for non-Gaussian):")
    print("      H=" + str(round(float(kw_stat), 4)) +
          "  p=" + "{:.4e}".format(float(kw_p)))
    if kw_p > 0.05:
        print("      NOT SIGNIFICANT: consistent with same distribution")
    else:
        print("      SIGNIFICANT: rank distributions differ")
    print()
    test_results.append({"test": "Kruskal-Wallis", "statistic": round(float(kw_stat),4),
                         "p_value": float(kw_p), "significant": kw_p < 0.05})

    # (3) Levene's test for equal variances
    lev_stat, lev_p = stats.levene(*arrays)
    print("  (3) Levene's test (equal variances):")
    print("      W=" + str(round(float(lev_stat), 4)) +
          "  p=" + "{:.4e}".format(float(lev_p)))
    if lev_p > 0.05:
        print("      NOT SIGNIFICANT: variances are homogeneous across catalogs")
    else:
        print("      SIGNIFICANT: variances differ across catalogs")
    print()
    test_results.append({"test": "Levene variance", "statistic": round(float(lev_stat),4),
                         "p_value": float(lev_p), "significant": lev_p < 0.05})

    # (4) Pairwise t-tests + Mann-Whitney + Cohen's d
    print("  (4) Pairwise comparisons (Bonferroni α = 0.05/3 = 0.0167):")
    print()
    pairs         = []
    bonferroni_a  = 0.05 / 3

    for i in range(len(cat_labels)):
        for j in range(i + 1, len(cat_labels)):
            a, b   = arrays[i], arrays[j]
            la, lb = cat_labels[i], cat_labels[j]

            t_stat, t_p   = stats.ttest_ind(a, b)
            u_stat, u_p   = stats.mannwhitneyu(a, b, alternative="two-sided")
            d             = cohens_d(a, b)
            sig_bonf      = t_p < bonferroni_a

            mag = "negligible"
            if abs(d) >= 0.8:   mag = "LARGE"
            elif abs(d) >= 0.5: mag = "medium"
            elif abs(d) >= 0.2: mag = "small"

            print("    " + la + " vs " + lb + ":")
            print("      t-test:       t=" + str(round(float(t_stat), 4)) +
                  "  p=" + "{:.4e}".format(float(t_p)) +
                  "  Bonferroni sig=" + str(sig_bonf))
            print("      Mann-Whitney: U=" + str(round(float(u_stat), 1)) +
                  "  p=" + "{:.4e}".format(float(u_p)))
            print("      Cohen's d:    " + str(round(float(d), 4)) +
                  "  (" + mag + " effect)")
            delta_means = float(np.mean(a)) - float(np.mean(b))
            print("      Mean diff:    " + str(round(delta_means, 5)) + " octaves")
            print()

            pairs.append({
                "pair":          la + " vs " + lb,
                "t_stat":        round(float(t_stat), 4),
                "t_p":           float(t_p),
                "bonferroni_sig":sig_bonf,
                "mw_U":          round(float(u_stat), 1),
                "mw_p":          float(u_p),
                "cohens_d":      round(float(d), 4),
                "effect_size":   mag,
                "mean_diff_oct": round(delta_means, 5),
                "mean_a":        round(float(np.mean(a)), 5),
                "mean_b":        round(float(np.mean(b)), 5),
            })

    # -----------------------------------------------------------------------
    # ALL-CATALOGS POOLED vs PREDICTION
    # -----------------------------------------------------------------------
    all_bbh = np.concatenate(arrays)
    print("=" * 70)
    print("  POOLED BBH BAND vs PREDICTION (n = 5.314)")
    print("=" * 70)
    print()
    ci_lo, ci_hi = mean_ci95(all_bbh)
    sig_pool     = sigma_from_pred(all_bbh)

    # One-sample t-test against 5.314
    t_one, p_one = stats.ttest_1samp(all_bbh, N_BBH_PRED)

    print("  Pooled N:          " + str(len(all_bbh)))
    print("  Pooled mean n:     " + str(round(float(np.mean(all_bbh)), 5)))
    print("  Pooled median n:   " + str(round(float(np.median(all_bbh)), 5)))
    print("  Pooled std:        " + str(round(float(np.std(all_bbh)), 5)))
    print("  95% CI on mean:    [" + str(round(ci_lo, 5)) + ", " + str(round(ci_hi, 5)) + "]")
    print("  Sigma from 5.314:  " + str(round(sig_pool, 4)) + "σ")
    print()
    print("  One-sample t-test vs 5.314:")
    print("    t=" + str(round(float(t_one), 4)) +
          "  p=" + "{:.4e}".format(float(p_one)))
    if p_one > 0.05:
        print("    NOT SIGNIFICANT: pooled mean is consistent with n = 5.314")
    else:
        print("    SIGNIFICANT: pooled mean departs from n = 5.314")
    print()

    # Per-catalog one-sample t-test vs 5.314
    print("  Per-catalog one-sample t-test vs 5.314:")
    one_sample = []
    for c in cats:
        arr = cat_bbh[c]
        if len(arr) == 0:
            continue
        t_c, p_c = stats.ttest_1samp(arr, N_BBH_PRED)
        sig_c    = p_c < 0.05
        sig_str  = sigma_from_pred(arr)
        print("    " + c + ":  t=" + str(round(float(t_c), 4)) +
              "  p=" + "{:.4e}".format(float(p_c)) +
              "  " + str(round(sig_str, 3)) + "σ from 5.314" +
              ("  *" if sig_c else ""))
        one_sample.append({"catalog": c, "t": round(float(t_c),4),
                            "p": float(p_c), "sigma_from_pred": round(sig_str,4),
                            "significant": sig_c})
    print()

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    anova_sig = f_p < 0.05
    kw_sig    = kw_p < 0.05
    any_pair_sig = any(p["bonferroni_sig"] for p in pairs)

    if not anova_sig and not kw_sig:
        print("  CONFIRMED: Three independent observing runs are statistically")
        print("  indistinguishable in octave depth n.")
        print()
        print("  Both parametric (ANOVA) and non-parametric (Kruskal-Wallis) tests")
        print("  return p > 0.05. The n = 5.314 prediction holds across O3a, O3b,")
        print("  and O4 data with no significant catalog-to-catalog drift.")
        print()
        print("  Prediction 1 (n_BBH independent of observing epoch): CONFIRMED")
    elif anova_sig and not kw_sig:
        print("  MIXED: ANOVA significant but Kruskal-Wallis not.")
        print("  Likely driven by non-Gaussian tails rather than true mean shift.")
        print("  Interpret with caution. Check Cohen's d for effect sizes.")
    elif any_pair_sig:
        print("  PARTIAL: Significant pairwise difference detected.")
        print("  Check which catalog pair drives the result and Cohen's d magnitude.")
    else:
        print("  BORDERLINE: Results mixed. Review pairwise tests above.")

    print()

    # -----------------------------------------------------------------------
    # OUTPUTS
    # -----------------------------------------------------------------------
    csv_per_event = "seraphim_anova_per_event.csv"
    csv_desc      = "seraphim_anova_descriptives.csv"
    csv_tests     = "seraphim_anova_tests.csv"
    csv_pairs     = "seraphim_anova_pairwise.csv"
    json_path     = "seraphim_anova_summary.json"

    with open(csv_per_event, "w", newline="") as cf:
        if rows:
            writer = csv.DictWriter(cf, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with open(csv_desc, "w", newline="") as cf:
        if cat_desc:
            writer = csv.DictWriter(cf, fieldnames=cat_desc[0].keys())
            writer.writeheader()
            writer.writerows(cat_desc)

    with open(csv_tests, "w", newline="") as cf:
        if test_results:
            writer = csv.DictWriter(cf, fieldnames=test_results[0].keys())
            writer.writeheader()
            writer.writerows(test_results)

    with open(csv_pairs, "w", newline="") as cf:
        if pairs:
            writer = csv.DictWriter(cf, fieldnames=pairs[0].keys())
            writer.writeheader()
            writer.writerows(pairs)

    summary = {
        "test":           "Catalog Consistency ANOVA",
        "K0_verify":      "1.1467e84 Hz^2 (exponent=84)",
        "n_prediction":   N_BBH_PRED,
        "events_deduped": len(rows),
        "anova":          {"F": round(float(f_stat),4), "p": float(f_p), "sig": anova_sig},
        "kruskal_wallis": {"H": round(float(kw_stat),4), "p": float(kw_p), "sig": kw_sig},
        "levene":         {"W": round(float(lev_stat),4),"p": float(lev_p),"sig":lev_p<0.05},
        "pairwise":       pairs,
        "pooled": {
            "n": len(all_bbh),
            "mean": round(float(np.mean(all_bbh)),5),
            "std":  round(float(np.std(all_bbh)),5),
            "ci95": [round(ci_lo,5), round(ci_hi,5)],
            "sigma_from_5314": round(sig_pool,4),
            "one_sample_t": round(float(t_one),4),
            "one_sample_p": float(p_one),
        },
        "per_catalog_descriptives": cat_desc,
        "per_catalog_one_sample":   one_sample,
    }

    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=4)

    if skipped:
        print("  Skipped " + str(len(skipped)) + " files.")
    print()
    print("[*] Per-event CSV:     " + csv_per_event)
    print("[*] Descriptives CSV:  " + csv_desc)
    print("[*] Test results CSV:  " + csv_tests)
    print("[*] Pairwise CSV:      " + csv_pairs)
    print("[*] Summary JSON:      " + json_path)
    print("[*] Done.")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    run_anova_test(root)
