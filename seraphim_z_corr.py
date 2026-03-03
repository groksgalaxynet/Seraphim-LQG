"""
seraphim_z_corr.py
------------------
Crawls the current directory and all subfolders for Seraphim CSV files
that contain redshift data (median_redshift / redshift / z column).

Runs the full redshift partial correlation battery:

  Test 1 — r(n, z)             direct Pearson + Spearman, all events
  Test 2 — r(n, z)             BBH band only
  Test 3 — Partial r(z → n | chi_eff)   redshift signal after spin control
  Test 4 — Partial r(chi_eff → n | z)   spin signal after redshift control
  Test 5 — Partial r(z → n | lum_dist)  redshift vs luminosity distance check
  Test 6 — Spearman r(n, z) per octave bin  (nonlinearity check)
  Test 7 — Catalog-level breakdown: mean n, CV, Spearman r(n,z) per file
  Test 8 — Kruskal-Wallis: BBH-band n distribution across catalogs

The key question for the paper:
  Does redshift drift survive controlling for spin and distance?
  If partial r(z|chi_eff) is not significant → geometric invariance holds.
  If it IS significant → framework violation candidate, needs investigation.

Run from any folder:
    python seraphim_z_corr.py

Outputs:
    seraphim_z_corr_results.csv   — per-test numeric results
    (stdout)                       — human-readable summary
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# ── Column aliases ────────────────────────────────────────────────────────────
N_COLS      = ["median_n", "mean_n", "n"]
CHI_COLS    = ["median_chi_eff", "chi_eff", "mean_chi_eff"]
Z_COLS      = ["median_redshift", "redshift", "z"]
DL_COLS     = ["median_lum_dist_Mpc", "lum_dist_Mpc", "luminosity_distance"]
STD_COLS    = ["std_n", "std"]
BAND_COLS   = ["in_bbh_band"]
Q_COLS      = ["median_q", "mass_ratio", "q"]
FILE_COLS   = ["event_file", "filename"]

# ── Prediction anchor ─────────────────────────────────────────────────────────
N_PRED      = 5.314
BBH_LO      = 4.76
BBH_HI      = 5.76

# ─────────────────────────────────────────────────────────────────────────────

def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalise_columns(df):
    renames = {}
    for canon, aliases in [
        ("median_n",          N_COLS),
        ("median_chi_eff",    CHI_COLS),
        ("median_redshift",   Z_COLS),
        ("median_lum_dist_Mpc", DL_COLS),
        ("std_n",             STD_COLS),
        ("in_bbh_band",       BAND_COLS),
        ("median_q",          Q_COLS),
        ("event_file",        FILE_COLS),
    ]:
        if canon not in df.columns:
            for alias in aliases:
                if alias in df.columns and alias != canon:
                    renames[alias] = canon
                    break
    return df.rename(columns=renames)

def find_csvs(root="."):
    found = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".csv"):
                found.append(os.path.join(dirpath, f))
    return sorted(found)

def has_z_data(df):
    z_col = first_col(df, Z_COLS)
    if z_col is None:
        return False
    valid = df[z_col].dropna()
    valid = valid[np.isfinite(valid.astype(float))]
    return len(valid) >= 10

def load_csv(path):
    try:
        df = pd.read_csv(path)
        if first_col(df, N_COLS) is None:
            return None
        df = normalise_columns(df)
        if not has_z_data(df):
            return None
        df["_source_file"] = os.path.basename(path)
        return df
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return None

# ── Stats helpers ─────────────────────────────────────────────────────────────

def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan, np.nan, mask.sum()
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p, mask.sum()

def safe_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan, np.nan, mask.sum()
    r, p = stats.spearmanr(x[mask], y[mask])
    return r, p, mask.sum()

def partial_r(x, y, z_ctrl):
    """Partial correlation of x and y controlling for z_ctrl."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z_ctrl)
    if mask.sum() < 8:
        return np.nan, np.nan, np.nan, mask.sum()
    x, y, z_ctrl = x[mask], y[mask], z_ctrl[mask]
    def resid(a, b):
        s, i, *_ = stats.linregress(b, a)
        return a - (s * b + i)
    rx = resid(x, z_ctrl)
    ry = resid(y, z_ctrl)
    r, _ = stats.pearsonr(rx, ry)
    N = mask.sum()
    t = r * np.sqrt((N - 3) / max(1 - r**2, 1e-12))
    p = 2 * stats.t.sf(abs(t), df=N - 3)
    return r, t, p, N

def sig_stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if (v is not None and np.isfinite(v)) else "—"

# ─────────────────────────────────────────────────────────────────────────────

def run_z_battery(df, label, results):
    print(f"\n{'='*64}")
    print(f"  {label}  (N={len(df)})")
    print(f"{'='*64}")

    n   = df["median_n"].values.astype(float)
    z   = df["median_redshift"].values.astype(float)
    chi = df["median_chi_eff"].values.astype(float) \
          if "median_chi_eff" in df.columns else np.full(len(n), np.nan)
    dl  = df["median_lum_dist_Mpc"].values.astype(float) \
          if "median_lum_dist_Mpc" in df.columns else np.full(len(n), np.nan)
    q   = df["median_q"].values.astype(float) \
          if "median_q" in df.columns else np.full(len(n), np.nan)

    # BBH band mask
    if "in_bbh_band" in df.columns:
        bbh_mask = df["in_bbh_band"].values == 1
    else:
        bbh_mask = (n > BBH_LO) & (n < BBH_HI)

    def record(test, var1, var2, r, t, p, N, rho=None, rho_p=None, note=""):
        results.append({
            "dataset":      label,
            "N_test":       N,
            "test":         test,
            "var1":         var1,
            "var2":         var2,
            "pearson_r":    fmt(r),
            "t_stat":       fmt(t, 3),
            "p_value":      f"{p:.4e}" if (p is not None and np.isfinite(p)) else "—",
            "sig":          sig_stars(p),
            "spearman_rho": fmt(rho),
            "spearman_p":   f"{rho_p:.4e}" if (rho_p is not None and np.isfinite(rho_p)) else "—",
            "note":         note
        })

    # ── TEST 1: Direct r(n, z) all events ────────────────────────────────────
    print("\n[TEST 1] Direct r(n, z) — all events")
    pr, pp, pN = safe_pearson(n, z)
    sr, sp, _  = safe_spearman(n, z)
    pt = pr * np.sqrt(max(pN - 2, 1) / max(1 - pr**2, 1e-12)) if np.isfinite(pr) else np.nan
    record("Direct r(n,z) all", "z", "n", pr, pt, pp, pN, sr, sp)
    print(f"  Pearson  r = {fmt(pr)}  t = {fmt(pt,3)}  p = {pp:.4e} {sig_stars(pp)}")
    print(f"  Spearman ρ = {fmt(sr)}  p = {sp:.4e} {sig_stars(sp)}")
    if np.isfinite(pp) and pp > 0.05:
        print("  --> No significant redshift drift. PASS.")
    elif np.isfinite(pp):
        print("  --> Significant drift detected. Check band composition.")

    # ── TEST 2: r(n, z) BBH band only ────────────────────────────────────────
    nb  = n[bbh_mask];  zb  = z[bbh_mask]
    Nb  = int(bbh_mask.sum())
    print(f"\n[TEST 2] r(n, z) BBH band only (N={Nb})")
    if Nb >= 5:
        pr2, pp2, _ = safe_pearson(nb, zb)
        sr2, sp2, _ = safe_spearman(nb, zb)
        t2 = pr2 * np.sqrt(max(Nb - 2, 1) / max(1 - pr2**2, 1e-12)) if np.isfinite(pr2) else np.nan
        record("r(n,z) BBH band", "z", "n_BBH", pr2, t2, pp2, Nb, sr2, sp2,
               note="BBH band only")
        print(f"  Pearson  r = {fmt(pr2)}  t = {fmt(t2,3)}  p = {pp2:.4e} {sig_stars(pp2)}")
        print(f"  Spearman ρ = {fmt(sr2)}  p = {sp2:.4e} {sig_stars(sp2)}")
        if np.isfinite(pp2) and pp2 > 0.05:
            print("  --> PASS: No redshift drift in BBH band.")
        elif np.isfinite(pp2):
            print("  --> FAIL: Significant drift in BBH band. Framework violation candidate.")
    else:
        print("  --> Insufficient BBH band events.")

    # ── TEST 3: Partial r(z → n | chi_eff) ───────────────────────────────────
    print("\n[TEST 3] Partial r(z → n | chi_eff)  — redshift after spin control?")
    r3, t3, p3, N3 = partial_r(z, n, chi)
    record("Partial r(z|chi)", "z", "n|chi_eff", r3, t3, p3, N3)
    if np.isfinite(r3):
        print(f"  partial r(z|chi_eff) = {fmt(r3)}  t = {fmt(t3,3)}  p = {p3:.4e} {sig_stars(p3)}  (N={N3})")
        if np.isfinite(p3) and p3 > 0.05:
            print("  --> PASS: Redshift signal vanishes after spin control.")
            print("      Geometric invariance confirmed: z drift was spin-mediated.")
        elif np.isfinite(p3):
            print("  --> FAIL: Redshift drift SURVIVES spin control.")
            print("      This is a genuine framework violation candidate.")
    else:
        print("  --> Insufficient overlapping z + chi_eff data.")

    # ── TEST 4: Partial r(chi_eff → n | z) ───────────────────────────────────
    print("\n[TEST 4] Partial r(chi_eff → n | z)  — spin after redshift control?")
    r4, t4, p4, N4 = partial_r(chi, n, z)
    record("Partial r(chi|z)", "chi_eff", "n|z", r4, t4, p4, N4)
    if np.isfinite(r4):
        print(f"  partial r(chi|z) = {fmt(r4)}  t = {fmt(t4,3)}  p = {p4:.4e} {sig_stars(p4)}  (N={N4})")
        if np.isfinite(p4) and p4 < 0.05:
            print("  --> Spin signal survives redshift control. Independent.")
        else:
            print("  --> Spin signal absorbed by redshift.")
    else:
        print("  --> Insufficient overlapping z + chi_eff data.")

    # ── TEST 5: Partial r(z → n | lum_dist) ──────────────────────────────────
    dl_valid = np.isfinite(dl).sum()
    if dl_valid >= 10:
        print("\n[TEST 5] Partial r(z → n | lum_dist)  — redshift vs distance?")
        r5, t5, p5, N5 = partial_r(z, n, dl)
        record("Partial r(z|dl)", "z", "n|lum_dist", r5, t5, p5, N5)
        if np.isfinite(r5):
            print(f"  partial r(z|lum_dist) = {fmt(r5)}  t = {fmt(t5,3)}  p = {p5:.4e} {sig_stars(p5)}  (N={N5})")
            if np.isfinite(p5) and p5 > 0.05:
                print("  --> Redshift and lum_dist carry the same information (expected).")
            else:
                print("  --> Redshift carries signal beyond lum_dist. Investigate.")
        else:
            print("  --> Insufficient lum_dist data.")
    else:
        print(f"\n[TEST 5] Skipped — insufficient lum_dist data ({dl_valid} valid rows).")

    # ── TEST 6: Spearman by redshift bin ─────────────────────────────────────
    z_valid = z[np.isfinite(z) & np.isfinite(n)]
    n_valid = n[np.isfinite(z) & np.isfinite(n)]
    if len(z_valid) >= 20:
        print("\n[TEST 6] r(n, z) per redshift bin (nonlinearity check)")
        bins = np.percentile(z_valid, [0, 25, 50, 75, 100])
        print(f"  {'Bin':<22} {'N':>4}  {'mean n':>8}  {'Spearman ρ':>11}  {'p':>10}")
        for i in range(len(bins) - 1):
            mask_bin = (z_valid >= bins[i]) & (z_valid < bins[i+1] + 1e-9)
            nb_bin = n_valid[mask_bin]
            zb_bin = z_valid[mask_bin]
            if len(nb_bin) < 5:
                continue
            rho_b, p_b = stats.spearmanr(zb_bin, nb_bin)
            label_bin = f"z=[{bins[i]:.2f}, {bins[i+1]:.2f}]"
            print(f"  {label_bin:<22} {len(nb_bin):>4}  {nb_bin.mean():>8.4f}  "
                  f"{rho_b:>+11.4f}  {p_b:>10.4e} {sig_stars(p_b)}")
            record(f"Spearman bin {i+1}", "z", f"n_bin{i+1}", np.nan, np.nan, p_b,
                   len(nb_bin), rho_b, p_b, note=f"z=[{bins[i]:.2f},{bins[i+1]:.2f}]")

    # ── TEST 7 not here — done at catalog level below ─────────────────────────

    return results


def catalog_breakdown(combined):
    print(f"\n{'='*64}")
    print("  [TEST 7] Per-catalog redshift summary")
    print(f"{'='*64}")
    print(f"  {'File':<48} {'N':>4}  {'BBH':>4}  {'mean n':>8}  {'CV':>6}  {'ρ(n,z)':>8}  {'p':>10}")
    for src, grp in combined.groupby("_source_file"):
        grp = grp.dropna(subset=["median_n"])
        n_all = grp["median_n"].values.astype(float)
        z_all = grp["median_redshift"].values.astype(float) \
                if "median_redshift" in grp.columns else np.full(len(n_all), np.nan)

        if "in_bbh_band" in grp.columns:
            n_bbh = grp[grp["in_bbh_band"] == 1]["median_n"].values.astype(float)
        else:
            n_bbh = n_all[(n_all > BBH_LO) & (n_all < BBH_HI)]

        mean_n = n_bbh.mean() if len(n_bbh) > 0 else np.nan
        cv     = (n_bbh.std() / mean_n * 100) if (len(n_bbh) > 1 and np.isfinite(mean_n)) else np.nan

        mask_z = np.isfinite(z_all) & np.isfinite(n_all)
        if mask_z.sum() >= 5:
            rho, p = stats.spearmanr(z_all[mask_z], n_all[mask_z])
        else:
            rho, p = np.nan, np.nan

        print(f"  {src:<48} {len(grp):>4}  {len(n_bbh):>4}  "
              f"{mean_n:>8.4f}  {cv:>5.2f}%  {rho:>+8.4f}  {p:>10.4e} {sig_stars(p)}")


def main():
    print("=" * 64)
    print("  Seraphim Redshift Partial Correlation Test")
    print("  Crawling for CSVs with z data from:", os.path.abspath("."))
    print("=" * 64)

    csv_files = find_csvs(".")
    print(f"\nFound {len(csv_files)} CSV file(s) — checking for redshift columns...")

    frames = []
    skipped_no_n  = []
    skipped_no_z  = []

    for path in csv_files:
        try:
            raw = pd.read_csv(path)
        except Exception as e:
            print(f"  [skip] {path}: {e}")
            continue

        if first_col(raw, N_COLS) is None:
            skipped_no_n.append(os.path.basename(path))
            continue

        raw = normalise_columns(raw)
        if not has_z_data(raw):
            skipped_no_z.append(os.path.basename(path))
            continue

        raw["_source_file"] = os.path.basename(path)
        frames.append(raw)
        n_z = raw["median_redshift"].dropna().shape[0]
        print(f"  [load] {path}  ({len(raw)} rows, {n_z} with z)")

    if skipped_no_n:
        print(f"\n  [skip — no n col]  : {', '.join(skipped_no_n)}")
    if skipped_no_z:
        print(f"  [skip — no z data] : {', '.join(skipped_no_z)}")

    if not frames:
        print("\nNo CSVs with both median_n and redshift data found.")
        print("Run seraphim_redshift_v4.py first to generate the redshift results CSV,")
        print("then run this script from the same folder.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate by event_file
    if "event_file" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=["event_file"], keep="first")
        after = len(combined)
        if before != after:
            print(f"\n  [dedup] Removed {before - after} duplicate rows (same event_file).")

    combined = combined.dropna(subset=["median_n", "median_redshift"])
    print(f"\nPooled: {len(combined)} events with valid n and z")

    # Column inventory
    cols_present = []
    for col, name in [("median_chi_eff","chi_eff"), ("median_q","q"),
                      ("median_lum_dist_Mpc","lum_dist"), ("std_n","std_n"),
                      ("in_bbh_band","band_flag")]:
        if col in combined.columns:
            n_valid = combined[col].dropna().shape[0]
            cols_present.append(f"{name}({n_valid})")
    print(f"Extra columns: {', '.join(cols_present) if cols_present else 'none'}")

    results = []

    # ── Run battery on full pooled set ────────────────────────────────────────
    run_z_battery(combined, "ALL CATALOGS POOLED", results)

    # ── Run per source file ───────────────────────────────────────────────────
    if len(frames) > 1:
        for src, grp in combined.groupby("_source_file"):
            grp = grp.dropna(subset=["median_n", "median_redshift"])
            if len(grp) >= 15:
                run_z_battery(grp, src, results)

    # ── Catalog breakdown ─────────────────────────────────────────────────────
    catalog_breakdown(combined)

    # ── Kruskal-Wallis ────────────────────────────────────────────────────────
    if "in_bbh_band" in combined.columns:
        bbh_all = combined[combined["in_bbh_band"] == 1]
    else:
        bbh_all = combined[(combined["median_n"] > BBH_LO) & (combined["median_n"] < BBH_HI)]

    groups = [grp["median_n"].dropna().values
              for _, grp in bbh_all.groupby("_source_file")
              if len(grp) >= 5]
    if len(groups) >= 2:
        stat, p = stats.kruskal(*groups)
        print(f"\n[TEST 8] Kruskal-Wallis BBH-band n across {len(groups)} catalog(s): "
              f"H={stat:.3f}, p={p:.4e} {sig_stars(p)}")
        if p > 0.05:
            print("  --> Catalogs statistically identical in BBH band. PASS.")
        else:
            print("  --> Catalogs differ. Check NSBH contamination in band definition.")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  REDSHIFT INVARIANCE VERDICT")
    print(f"{'='*64}")

    if "in_bbh_band" in combined.columns:
        bbh = combined[combined["in_bbh_band"] == 1]["median_n"].dropna()
    else:
        bbh = combined[(combined["median_n"] > BBH_LO) & (combined["median_n"] < BBH_HI)]["median_n"].dropna()

    if len(bbh) > 0:
        mean_n = bbh.mean()
        cv     = bbh.std() / mean_n * 100
        delta  = abs(mean_n - N_PRED)
        sigma  = delta / (bbh.std() / np.sqrt(len(bbh)))
        print(f"  BBH band N      : {len(bbh)}")
        print(f"  Mean n          : {mean_n:.4f}  (predicted {N_PRED}, Δ={delta:.4f})")
        print(f"  CV              : {cv:.3f}%")
        print(f"  Offset          : {sigma:.2f}σ")

    # Pull key results
    key_tests = [r for r in results if r["test"] in
                 ("r(n,z) BBH band", "Partial r(z|chi)", "Direct r(n,z) all")]
    print("\n  Key results summary:")
    for r in key_tests:
        print(f"  {r['test']:<35} r={r['pearson_r']:>8}  ρ={r['spearman_rho']:>8}  "
              f"p={r['p_value']:>12}  {r['sig']}")

    # Overall pass/fail
    partial_z_chi = next((r for r in results
                          if r["test"] == "Partial r(z|chi)" and r["dataset"] == "ALL CATALOGS POOLED"),
                         None)
    if partial_z_chi and partial_z_chi["p_value"] not in ("—", None):
        p_val = float(partial_z_chi["p_value"])
        if p_val > 0.05:
            print("\n  FRAMEWORK STATUS: PASS")
            print("  Redshift drift does not survive spin control.")
            print("  Prediction 7 (redshift invariance) confirmed in partial correlation.")
        else:
            print("\n  FRAMEWORK STATUS: INVESTIGATE")
            print("  Redshift signal survives spin control — genuine drift candidate.")
            print("  Check NSBH contamination and lum_dist correlation before flagging.")
    else:
        print("\n  FRAMEWORK STATUS: INCOMPLETE — need chi_eff column alongside z.")
        print("  Run seraphim_redshift_v4.py (which outputs both) and retry.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = "seraphim_z_corr_results.csv"
    if results:
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"\n[+] Results saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
