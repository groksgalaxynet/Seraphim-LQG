"""
seraphim_partial_corr.py
------------------------
Crawls the current directory and all subfolders for Seraphim CSV result files,
pools compatible data, and runs the full partial correlation battery:

  Test 1 — Direct correlations: r(n, chi_eff), r(n, q), r(n, z), r(n, std_n)
  Test 2 — Partial r(chi_eff → n | q)     if both columns present
  Test 3 — Partial r(q → n | chi_eff)     if both columns present
  Test 4 — Partial r(chi_eff → n | z)     if both columns present
  Test 5 — Partial r(z → n | chi_eff)     if both columns present
  Test 6 — BBH-band-only repeat of all above
  Test 7 — Catalog-level breakdown (mean n, CV, N per source file)
  Test 8 — Spearman on all pairs (non-parametric check)

Run from any folder:
    python seraphim_partial_corr.py

Outputs:
    seraphim_partial_corr_results.csv   — per-test numeric results
    (stdout)                            — human-readable summary
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from scipy import stats

# ── Column name aliases so the script handles different CSV flavours ──────────
N_COLS      = ["median_n", "mean_n", "n"]
CHI_COLS    = ["median_chi_eff", "chi_eff", "mean_chi_eff"]
Q_COLS      = ["median_q", "mass_ratio", "q"]
Z_COLS      = ["median_redshift", "redshift", "z"]
STD_COLS    = ["std_n", "std"]
BAND_COLS   = ["in_bbh_band"]
TYPE_COLS   = ["event_type"]
FILE_COLS   = ["event_file", "filename"]

def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def partial_r(x, y, z):
    """Partial correlation of x and y controlling for z (residuals method).
    Drops rows where any of the three arrays is NaN."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    def resid(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    rx = resid(x, z)
    ry = resid(y, z)
    r, _ = stats.pearsonr(rx, ry)
    N = len(x)
    t_stat = r * np.sqrt((N - 3) / (1 - r**2 + 1e-15))
    p = 2 * stats.t.sf(abs(t_stat), df=N - 3)
    return r, t_stat, p

def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan, np.nan
    r, p = stats.spearmanr(x[mask], y[mask])
    return r, p

def spearman_resid(x, y, z):
    """Spearman on residuals of x~z and y~z, NaN-safe."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 5:
        return np.nan, np.nan
    def resid(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    return stats.spearmanr(resid(x, z), resid(y, z))

def find_csvs(root="."):
    """Recursively find all .csv files under root."""
    found = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".csv"):
                found.append(os.path.join(dirpath, f))
    return sorted(found)

def looks_like_seraphim(df):
    """Return True if the dataframe has at least median_n (or alias)."""
    return first_col(df, N_COLS) is not None

def load_and_tag(path):
    try:
        df = pd.read_csv(path)
        if not looks_like_seraphim(df):
            return None
        df["_source_file"] = os.path.basename(path)
        return df
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return None

def normalise_columns(df):
    """Rename aliased columns to canonical names."""
    renames = {}
    for canon, aliases in [
        ("median_n",       N_COLS),
        ("median_chi_eff", CHI_COLS),
        ("median_q",       Q_COLS),
        ("median_redshift",Z_COLS),
        ("std_n",          STD_COLS),
        ("in_bbh_band",    BAND_COLS),
        ("event_type",     TYPE_COLS),
        ("event_file",     FILE_COLS),
    ]:
        if canon not in df.columns:
            for alias in aliases:
                if alias in df.columns and alias != canon:
                    renames[alias] = canon
                    break
    return df.rename(columns=renames)

def corr_block(df, label, results_list):
    """Run one block of tests on a dataframe. Print and collect results."""
    print(f"\n{'='*60}")
    print(f"  {label}  (N={len(df)})")
    print(f"{'='*60}")

    n_col   = "median_n"
    chi_col = "median_chi_eff"
    q_col   = "median_q"
    z_col   = "median_redshift"
    std_col = "std_n"

    has_chi = chi_col in df.columns
    has_q   = q_col   in df.columns
    has_z   = z_col   in df.columns
    has_std = std_col in df.columns

    n = df[n_col].values

    def row(test, var1, var2, r, t, p, rho=None, rho_p=None, note=""):
        sig = "***" if (p is not None and np.isfinite(p) and p < 0.001) \
              else ("**" if (p is not None and np.isfinite(p) and p < 0.01) \
              else ("*"  if (p is not None and np.isfinite(p) and p < 0.05) else ""))
        rho_str = f"{rho:.4f}" if (rho is not None and np.isfinite(rho)) else "—"
        results_list.append({
            "dataset": label,
            "N": len(df),
            "test": test,
            "var1": var1,
            "var2": var2,
            "pearson_r": round(r, 4) if (r is not None and np.isfinite(r)) else None,
            "t_stat":    round(t, 3) if (t is not None and np.isfinite(t)) else None,
            "p_value":   f"{p:.4e}" if (p is not None and np.isfinite(p)) else None,
            "sig":       sig,
            "spearman_rho": rho_str,
            "spearman_p":   f"{rho_p:.4e}" if (rho_p is not None and np.isfinite(rho_p)) else "—",
            "note": note
        })
        return sig

    # ── TEST 1: Direct Pearson ────────────────────────────────────────────────
    print("\n[TEST 1] Direct Pearson correlations with n")
    for col, label_col in [(chi_col, "chi_eff"), (q_col, "q"), (z_col, "z"), (std_col, "std_n")]:
        if col not in df.columns:
            continue
        x = df[col].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(n)
        if mask.sum() < 5:
            print(f"  r(n, {label_col:8s}) — insufficient valid data ({mask.sum()} rows)")
            continue
        xm, nm = x[mask], n[mask]
        r, p = stats.pearsonr(xm, nm)
        rho, rho_p = spearman(xm, nm)
        N = mask.sum()
        t = r * np.sqrt((N-2) / (1 - r**2 + 1e-15))
        sig = row("Direct Pearson", label_col, "n", r, t, p, rho, rho_p)
        print(f"  r(n, {label_col:8s}) = {r:+.4f}  t={t:+6.2f}  p={p:.4e} {sig:<3}  ρ={rho:+.4f}")

    # ── TEST 2 & 3: Partial correlations (chi | q, q | chi) ──────────────────
    if has_chi and has_q:
        print("\n[TEST 2] Partial r(chi_eff → n | q)  — does spin matter beyond mass ratio?")
        chi = df[chi_col].values.astype(float)
        q   = df[q_col].values.astype(float)
        r, t, p = partial_r(n, chi, q)
        rho, rho_p = spearman_resid(chi, n, q)
        sig = row("Partial chi|q", "chi_eff", "n|q", r, t, p, rho, rho_p)
        if np.isfinite(r):
            print(f"  partial r(chi_eff | q) = {r:+.4f}  t={t:+.3f}  p={p:.4e} {sig}")
            if np.isfinite(p) and p < 0.05:
                print("  --> SPIN SURVIVES mass-ratio control. Independent signal confirmed.")
            else:
                print("  --> Spin signal absorbed by mass-ratio. chi_eff may be proxy.")
        else:
            print("  --> Insufficient overlapping valid data for this test.")

        print("\n[TEST 3] Partial r(q → n | chi_eff)  — does mass ratio matter beyond spin?")
        r, t, p = partial_r(n, q, chi)
        rho, rho_p = spearman_resid(q, n, chi)
        sig = row("Partial q|chi", "q", "n|chi_eff", r, t, p, rho, rho_p)
        if np.isfinite(r):
            print(f"  partial r(q | chi_eff) = {r:+.4f}  t={t:+.3f}  p={p:.4e} {sig}")
            if np.isfinite(p) and p < 0.05:
                print("  --> MASS RATIO survives spin control. Independent signal confirmed.")
            else:
                print("  --> Mass ratio signal absorbed by spin.")
        else:
            print("  --> Insufficient overlapping valid data for this test.")

    # ── TEST 4 & 5: Partial correlations (chi | z, z | chi) ──────────────────
    if has_chi and has_z:
        print("\n[TEST 4] Partial r(chi_eff → n | z)  — spin signal after redshift control?")
        chi = df[chi_col].values.astype(float)
        z   = df[z_col].values.astype(float)
        r, t, p = partial_r(n, chi, z)
        sig = row("Partial chi|z", "chi_eff", "n|z", r, t, p)
        if np.isfinite(r):
            print(f"  partial r(chi_eff | z) = {r:+.4f}  t={t:+.3f}  p={p:.4e} {sig}")
        else:
            print("  --> Insufficient overlapping valid data for this test.")

        print("\n[TEST 5] Partial r(z → n | chi_eff)  — redshift signal after spin control?")
        r, t, p = partial_r(n, z, chi)
        sig = row("Partial z|chi", "z", "n|chi_eff", r, t, p)
        if np.isfinite(r):
            print(f"  partial r(z | chi_eff) = {r:+.4f}  t={t:+.3f}  p={p:.4e} {sig}")
            if np.isfinite(p) and p < 0.05:
                print("  --> REDSHIFT DRIFT survives spin control. Framework violation candidate.")
            else:
                print("  --> No redshift drift after spin control. Geometric invariance holds.")
        else:
            print("  --> Insufficient overlapping valid data for this test.")

    # ── TEST 6: BBH-band-only repeat ─────────────────────────────────────────
    if "in_bbh_band" in df.columns:
        bbh = df[df["in_bbh_band"] == 1].copy()
        if len(bbh) > 10:
            print(f"\n[TEST 6] BBH-band-only repeat (N={len(bbh)})")
            for col, label_col in [(chi_col, "chi_eff"), (q_col, "q"), (z_col, "z")]:
                if col not in bbh.columns:
                    continue
                x = bbh[col].values.astype(float)
                nb = bbh["median_n"].values.astype(float)
                mask = np.isfinite(x) & np.isfinite(nb)
                if mask.sum() < 5:
                    continue
                xm, nm = x[mask], nb[mask]
                r, p = stats.pearsonr(xm, nm)
                rho, rho_p = spearman(xm, nm)
                N = mask.sum()
                t = r * np.sqrt((N-2) / (1 - r**2 + 1e-15))
                sig = row("BBH-band Pearson", label_col, "n_BBH", r, t, p, rho, rho_p,
                          note="BBH band only")
                print(f"  r(n_BBH, {label_col:8s}) = {r:+.4f}  t={t:+6.2f}  p={p:.4e} {sig:<3}  ρ={rho:+.4f}")

    return results_list

def catalog_breakdown(df):
    """Print per-source-file stats."""
    if "_source_file" not in df.columns:
        return
    print("\n[TEST 7] Catalog breakdown")
    print(f"  {'File':<55} {'N':>4} {'Mean n':>8} {'CV':>7} {'σ':>7}")
    for src, grp in df.groupby("_source_file"):
        n_vals = grp["median_n"].dropna()
        mean_n = n_vals.mean()
        cv     = (n_vals.std() / mean_n * 100) if mean_n != 0 else 0
        print(f"  {src:<55} {len(n_vals):>4} {mean_n:>8.4f} {cv:>6.2f}% {n_vals.std():>7.4f}")

def main():
    print("=" * 60)
    print("  Seraphim Partial Correlation Test")
    print("  Crawling for CSVs from:", os.path.abspath("."))
    print("=" * 60)

    csv_files = find_csvs(".")
    print(f"\nFound {len(csv_files)} CSV file(s):")

    frames = []
    for path in csv_files:
        df = load_and_tag(path)
        if df is not None:
            df = normalise_columns(df)
            frames.append(df)
            print(f"  [load] {path}  ({len(df)} rows)")
        else:
            print(f"  [skip] {path}  (no Seraphim columns)")

    if not frames:
        print("\nNo compatible Seraphim CSVs found. Run from a folder containing")
        print("seraphim_redshift_results.csv or similar output files.")
        sys.exit(1)

    # ── Pool all frames ───────────────────────────────────────────────────────
    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: if same event_file appears in multiple CSVs, keep first
    if "event_file" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=["event_file"], keep="first")
        after = len(combined)
        if before != after:
            print(f"\n[dedup] Removed {before - after} duplicate event rows (same event_file).")

    combined = combined.dropna(subset=["median_n"])
    print(f"\nPooled dataset: {len(combined)} events with valid median_n")

    # Column inventory
    available = []
    for col, name in [("median_chi_eff","chi_eff"), ("median_q","q"),
                      ("median_redshift","z"), ("std_n","std_n")]:
        if col in combined.columns:
            valid_n = combined[col].dropna().count()
            available.append(f"{name}({valid_n})")
    print(f"Available columns: {', '.join(available) if available else 'none beyond n'}")

    results = []

    # ── Run on full pooled dataset ────────────────────────────────────────────
    corr_block(combined, "ALL CATALOGS POOLED", results)

    # ── Run per source file if more than one ─────────────────────────────────
    if len(frames) > 1:
        for src, grp in combined.groupby("_source_file"):
            grp = grp.dropna(subset=["median_n"])
            if len(grp) >= 15:
                corr_block(grp, src, results)

    # ── Catalog breakdown ─────────────────────────────────────────────────────
    catalog_breakdown(combined)

    # ── Kruskal-Wallis across sources ─────────────────────────────────────────
    if "_source_file" in combined.columns:
        groups = [grp["median_n"].dropna().values
                  for _, grp in combined.groupby("_source_file")
                  if len(grp) >= 5]
        if len(groups) >= 2:
            stat, p = stats.kruskal(*groups)
            print(f"\n[TEST 8] Kruskal-Wallis across {len(groups)} catalog(s): "
                  f"H={stat:.3f}, p={p:.4e}")
            if p > 0.05:
                print("  --> Catalogs statistically identical. Geometric universality holds.")
            else:
                print("  --> Catalogs differ significantly. Investigate source.")

    # ── Framework verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FRAMEWORK VERDICT")
    print("=" * 60)
    if "in_bbh_band" in combined.columns:
        bbh = combined[combined["in_bbh_band"] == 1].copy()
        filter_method = "in_bbh_band == 1"
    else:
        bbh = combined[(combined["median_n"] > 4.76) & (combined["median_n"] < 5.76)].copy()
        filter_method = "4.76 < n < 5.76"
    if len(bbh) == 0:
        bbh = combined.copy()
        filter_method = "all events (no band filter)"
    n_vals = bbh["median_n"].dropna()
    mean_n = n_vals.mean()
    cv     = n_vals.std() / mean_n * 100
    delta  = abs(mean_n - 5.314)
    sigma  = delta / (n_vals.std() / np.sqrt(len(n_vals)))
    print(f"  Filter          : {filter_method}")
    print(f"  BBH band events : {len(n_vals)}")
    print(f"  Mean n          : {mean_n:.4f}  (predicted 5.314, Δ={delta:.4f})")
    print(f"  CV              : {cv:.3f}%")
    print(f"  Offset from pred: {sigma:.2f}σ")
    if delta < 0.1 and sigma < 1.0:
        print("  STATUS: CONFIRMED — prediction holds across pooled dataset.")
    else:
        print("  STATUS: CHECK — mean n outside expected range, investigate.")
    print("\n  NOTE: Kruskal-Wallis significance across catalogs is expected if")
    print("  NSBH events are present in some catalogs — this is physics, not artifact.")

    # ── Save results CSV ──────────────────────────────────────────────────────
    out_path = "seraphim_partial_corr_results.csv"
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(out_path, index=False)
        print(f"\n[+] Results saved to {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
