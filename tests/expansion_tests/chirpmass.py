import os
import glob
import h5py
import numpy as np
from scipy import stats
import csv

# === SERAPHIM CHIRP MASS INDEPENDENCE TEST v3 ===
# Correct column names confirmed from O3a HDF5 files:
#   total_mass_source, final_mass_source, chirp_mass_source

ALPHA     = 0.007297
K_0       = 1.1467e84
NU_PLANCK = 1.8549e43
J_SPIN    = 0.5
SQRT_J    = np.sqrt(J_SPIN * (J_SPIN + 1.0))

TARGET_APPROX = "C01:IMRPhenomXPHM"

def calculate_octave(m_total, m_final):
    e_loss = m_total - m_final
    valid  = (e_loss > 0) & (e_loss < m_total)
    n_star = np.where(valid, e_loss / (m_total * ALPHA), np.nan)
    nu     = np.where(valid, np.sqrt(K_0 / (SQRT_J * n_star)), np.nan)
    n_oct  = np.where(valid, np.log2(NU_PLANCK / nu), np.nan)
    good   = valid & np.isfinite(n_oct) & (n_oct > 3.0) & (n_oct < 8.0)
    return n_oct, good

def partial_r(x, y, z):
    rx = np.polyfit(z, x, 1)
    ry = np.polyfit(z, y, 1)
    return stats.pearsonr(x - np.polyval(rx, z), y - np.polyval(ry, z))

def process_files(directory="."):
    h5_files = sorted(
        glob.glob(os.path.join(directory, "*.h5")) +
        glob.glob(os.path.join(directory, "*.hdf5"))
    )
    if not h5_files:
        print("[!] No HDF5 files found in:", os.path.abspath(directory))
        return

    print("[*] Seraphim Chirp Mass Independence Test v3")
    print(f"    Files found: {len(h5_files)}")
    print()

    results = []
    skipped = 0

    for filepath in h5_files:
        filename = os.path.basename(filepath)
        try:
            with h5py.File(filepath, "r") as f:
                if TARGET_APPROX not in f:
                    skipped += 1
                    continue
                ps = f[TARGET_APPROX]["posterior_samples"]
                m_total = ps["total_mass_source"][:]
                m_final = ps["final_mass_source"][:]
                mc      = ps["chirp_mass_source"][:]
                q       = ps["mass_ratio"][:]
                chi     = ps["chi_eff"][:]
                n_oct, good = calculate_octave(m_total, m_final)
                if good.sum() < 100:
                    skipped += 1
                    continue
                results.append({
                    "file":       filename,
                    "median_n":   float(np.median(n_oct[good])),
                    "median_mc":  float(np.median(mc[good])),
                    "median_q":   float(np.median(q[good])),
                    "median_chi": float(np.median(chi[good])),
                    "n_samples":  int(good.sum())
                })
        except Exception as e:
            print(f"  [!] {filename}: {e}")
            skipped += 1

    print(f"  Processed: {len(results)},  Skipped: {skipped}")
    print()
    if not results:
        print("[!] No valid results.")
        return

    n_arr   = np.array([r["median_n"]   for r in results])
    mc_arr  = np.array([r["median_mc"]  for r in results])
    q_arr   = np.array([r["median_q"]   for r in results])
    chi_arr = np.array([r["median_chi"] for r in results])

    bbh     = n_arr > 4.76
    n_bbh   = n_arr[bbh];   mc_bbh  = mc_arr[bbh]
    q_bbh   = q_arr[bbh];   chi_bbh = chi_arr[bbh]
    lnq_bbh = np.log(q_bbh)

    mean_n = np.mean(n_bbh); std_n = np.std(n_bbh)
    cv = std_n / mean_n * 100

    print(f"  BBH in band:  {bbh.sum()},  Outside band: {(~bbh).sum()}")
    print(f"  Mean n = {mean_n:.4f},  Std = {std_n:.4f},  CV = {cv:.4f}%")
    print(f"  Mc range = [{mc_bbh.min():.1f}, {mc_bbh.max():.1f}] Msun")
    print()

    r, p       = stats.pearsonr(mc_bbh, n_bbh)
    r_sp, p_sp = stats.spearmanr(mc_bbh, n_bbh)
    print(f"  Pearson  r(Mc, n)              = {r:.4f},  p = {p:.4f}")
    print(f"  Spearman r(Mc, n)              = {r_sp:.4f},  p = {p_sp:.4f}")

    r_chi, p_chi = partial_r(mc_bbh, n_bbh, chi_bbh)
    r_q,   p_q   = partial_r(mc_bbh, n_bbh, lnq_bbh)
    print(f"  Partial  r(Mc, n | chi_eff)    = {r_chi:.4f},  p = {p_chi:.4f}")
    print(f"  Partial  r(Mc, n | ln_q)       = {r_q:.4f},  p = {p_q:.4f}")
    print()

    q25, q50, q75 = np.percentile(mc_bbh, [25, 50, 75])
    groups = []
    print("  Octave n by chirp mass quartile:")
    for mask, label in [
        (mc_bbh <= q25,                    f"Q1 Mc<={q25:.1f}"),
        ((mc_bbh>q25)&(mc_bbh<=q50),       f"Q2 Mc<={q50:.1f}"),
        ((mc_bbh>q50)&(mc_bbh<=q75),       f"Q3 Mc<={q75:.1f}"),
        (mc_bbh > q75,                      f"Q4 Mc> {q75:.1f}"),
    ]:
        g = n_bbh[mask]; groups.append(g)
        print(f"    {label}: mean={np.mean(g):.4f}  std={np.std(g):.4f}  N={len(g)}")

    f_stat, f_p = stats.f_oneway(*groups)
    print(f"\n  ANOVA p = {f_p:.4f}  (want >> 0.05 for mass-independence)")
    print()

    if abs(r) < 0.20 and p > 0.05:
        print("  RESULT: PASS — octave is independent of chirp mass.")
    elif abs(r) >= 0.20 and p < 0.05:
        print("  RESULT: FLAG — chirp mass correlation present.")
        print("          Check partial correlations — if they drop near zero")
        print("          after controlling for q, the correlation is mass-ratio driven.")
    else:
        print("  RESULT: WEAK PASS — marginal.")

    outfile = "seraphim_chirpmass_results.csv"
    with open(outfile, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=[
            "file","median_n","median_mc","median_q","median_chi","n_samples"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Saved: {outfile}")
    print("[*] Done.")

process_files(directory=".")
