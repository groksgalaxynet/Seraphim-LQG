#!/usr/bin/env python3
"""
SERAPHIM TIDAL DEFORMABILITY TEST  (BNS edition)
==================================================
Direct test of the compactness equation:

    n(C) = 3.561 + 3.506 * C

using LVK-computed compactness values from NRTidal waveform posteriors.

RUN:
    python seraphim_tidal_test.py /path/to/folder_or_file.h5

COLUMNS CONFIRMED IN GW190425 NRTidal posteriors (from seraphim_cols.py):
    mass_1_source, mass_2_source          -- gravitational masses (Msun)
    total_mass_source                     -- total gravitational mass
    baryonic_mass_1_source, baryonic_mass_2_source  -- baryonic masses
    compactness_1, compactness_2          -- LVK-computed compactness
    lambda_1, lambda_2, lambda_tilde      -- tidal deformabilities
    chirp_mass_source, mass_ratio         -- event parameters

NOTE ON ENERGY LOSS FOR BNS:
    NRTidal waveforms do NOT produce a final_mass column — the remnant is
    a neutron star or collapsar, not a BH ringdown. Therefore the standard
    BBH n_observed calculation cannot run directly.

    We use TWO proxy approaches and are explicit about their limitations:
    
    Method A — Binding energy: 
        E_bind = sum(baryonic_mass_i - gravitational_mass_i)
        This is an UPPER BOUND on available energy. For BNS,
        E_GW << E_bind, so this gives n << n_BBH as expected.

    Method B — Published radiated fraction:
        For GW190425, NR simulations estimate E_rad ~ 0.05–0.15% of M_total.
        (Dietrich et al., Bernuzzi et al.) We propagate this through the
        K0 formula and report the n_obs range.

    The primary test is therefore:
    -- Does the compactness equation predict n in the physically correct
       sub-BBH range for the actual LVK C values?
    -- Are those C values consistent with the Yagi-Yunes Lambda-C relation?

VERDICT THRESHOLDS (pre-stated before running):
    |residual| < 0.15 oct  = Strong confirmation
    |residual| < 0.30 oct  = Consistent
    |residual| < 0.50 oct  = Weak / posterior-dominated
    |residual| > 0.50 oct  = Tension with framework
"""

import os
import sys
import glob
import h5py
import numpy as np

# =============================================================================
# PHYSICAL CONSTANTS (unchanged from paper)
# =============================================================================
ALPHA       = 0.007297353
NU_PLANCK   = 1.8549e43         # Hz
J_SPIN      = 0.5
SQRT_J      = np.sqrt(J_SPIN * (J_SPIN + 1.0))
K0          = 1.1467e84         # Robertson K0 [Hz^2]

# =============================================================================
# COMPACTNESS EQUATION PARAMETERS (from paper — not tuned here)
# =============================================================================
N_FLIP      = 3.561
SLOPE       = 3.506
N_BBH       = 5.314
BETA        = 0.814
C_BH        = 0.5
BBH_LO      = 4.76
BBH_HI      = 5.76

# Pre-stated verdict thresholds
THRESH_STRONG  = 0.15
THRESH_CONSIST = 0.30
THRESH_WEAK    = 0.50

# GW190425 radiated energy fraction estimates from NR (Dietrich+, Bernuzzi+)
ERAD_LO  = 0.0005
ERAD_MED = 0.0008
ERAD_HI  = 0.0015


# =============================================================================
# UTILITIES
# =============================================================================

def get_col(ps, *names):
    for name in names:
        try:
            if hasattr(ps, 'dtype') and ps.dtype.names and name in ps.dtype.names:
                arr = np.array(ps[name], dtype=float)
                if np.any(np.isfinite(arr)):
                    return arr
        except Exception:
            pass
    return None


def calc_n_from_efrac(m_total, e_frac):
    """Compute n-octave from total mass and fractional radiated energy."""
    e_loss = e_frac * m_total
    valid  = (e_loss > 0) & np.isfinite(e_loss) & np.isfinite(m_total)
    n_out  = np.full(len(m_total), np.nan)
    if valid.sum() < 10:
        return n_out
    n_star            = e_loss[valid] / (m_total[valid] * ALPHA)
    nu                = np.sqrt(K0 / (SQRT_J * n_star))
    n_octave          = np.log2(NU_PLANCK / nu)
    good              = np.isfinite(n_octave)
    n_out[np.where(valid)[0][good]] = n_octave[good]
    return n_out


def calc_n_from_binding(m1g, m2g, mb1, mb2):
    """Compute n using binding energy E_bind = (mb1+mb2) - (m1+m2).
    This is an UPPER BOUND — much larger than actual E_GW for BNS."""
    m_total  = m1g + m2g
    e_bind   = (mb1 + mb2) - m_total
    valid    = (e_bind > 0) & np.isfinite(e_bind) & np.isfinite(m_total)
    n_out    = np.full(len(m_total), np.nan)
    if valid.sum() < 10:
        return n_out
    n_star   = e_bind[valid] / (m_total[valid] * ALPHA)
    nu       = np.sqrt(K0 / (SQRT_J * n_star))
    n_octave = np.log2(NU_PLANCK / nu)
    good     = np.isfinite(n_octave)
    n_out[np.where(valid)[0][good]] = n_octave[good]
    return n_out


def n_linear(C):
    return N_FLIP + SLOPE * float(C)


def n_beta(C):
    C = float(C)
    if C <= 0:
        return np.nan
    return N_BBH - BETA * np.log2(C_BH / C)


def summarize(arr, label, indent=4):
    pad = " " * indent
    fin = arr[np.isfinite(arr)]
    if len(fin) == 0:
        print(f"{pad}{label}: NO VALID DATA")
        return None
    med = float(np.median(fin))
    std = float(np.std(fin))
    p5, p16, p84, p95 = np.percentile(fin, [5, 16, 84, 95])
    print(f"{pad}{label}:")
    print(f"{pad}  N={len(fin)}  median={med:.4f}  std={std:.4f}  "
          f"68%CI=[{p16:.4f},{p84:.4f}]  90%CI=[{p5:.4f},{p95:.4f}]")
    return med


def verdict_str(r):
    if r < THRESH_STRONG:
        return f"STRONG CONFIRMATION  (|r|={r:.3f} < 0.15)"
    elif r < THRESH_CONSIST:
        return f"CONSISTENT           (|r|={r:.3f} < 0.30)"
    elif r < THRESH_WEAK:
        return f"WEAK/BROAD           (|r|={r:.3f} < 0.50)"
    else:
        return f"TENSION              (|r|={r:.3f} > 0.50)"


def get_tidal_keys(f):
    keys = []
    for k in f.keys():
        if k in ("history", "version", "prior"):
            continue
        if "Tidal" in k or "tidal" in k:
            try:
                ps = f[k]["posterior_samples"]
                keys.append((k, ps))
            except Exception:
                pass
    return keys


# =============================================================================
# MAIN TEST
# =============================================================================

def run(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.h5")) +
                       glob.glob(os.path.join(path, "*.hdf5")))
    else:
        files = [path]

    if not files:
        print(f"No HDF5 files found at: {path}"); return

    print("=" * 72)
    print("SERAPHIM TIDAL DEFORMABILITY TEST")
    print("=" * 72)
    print()
    print("COMPACTNESS EQUATION (pre-stated, not tuned to this data):")
    print(f"  Linear:  n(C) = {N_FLIP} + {SLOPE} * C")
    print(f"  Beta:    n(C) = {N_BBH} - {BETA} * log2(0.5/C)")
    print()
    print("VERDICT THRESHOLDS (pre-stated):")
    print("  <0.15=Strong  <0.30=Consistent  <0.50=Weak  >0.50=Tension")
    print()

    summary_rows = []

    for filepath in files:
        fname = os.path.basename(filepath)
        print("=" * 72)
        print(f"FILE: {fname}")
        print("=" * 72)

        try:
            with h5py.File(filepath, "r") as f:
                tidal_keys = get_tidal_keys(f)
                if not tidal_keys:
                    print("  No NRTidal keys found — skipping.")
                    continue
                print(f"  NRTidal keys: {[k for k,_ in tidal_keys]}")
                print()

                for wf_key, ps in tidal_keys:
                    print(f"  {'─'*68}")
                    print(f"  WAVEFORM: {wf_key}")
                    print(f"  {'─'*68}")
                    print(f"  Posterior samples: {len(ps)}")
                    print()

                    # --- pull columns ---
                    m1g  = get_col(ps, "mass_1_source",  "mass_1")
                    m2g  = get_col(ps, "mass_2_source",  "mass_2")
                    mb1  = get_col(ps, "baryonic_mass_1_source", "baryonic_mass_1")
                    mb2  = get_col(ps, "baryonic_mass_2_source", "baryonic_mass_2")
                    C1   = get_col(ps, "compactness_1")
                    C2   = get_col(ps, "compactness_2")
                    lam1 = get_col(ps, "lambda_1")
                    lam2 = get_col(ps, "lambda_2")
                    lamt = get_col(ps, "lambda_tilde")
                    mc   = get_col(ps, "chirp_mass_source", "chirp_mass")
                    q    = get_col(ps, "mass_ratio")
                    z    = get_col(ps, "redshift")

                    col_status = {
                        "mass_1_source": m1g, "mass_2_source": m2g,
                        "baryonic_mass_1_source": mb1, "baryonic_mass_2_source": mb2,
                        "compactness_1": C1, "compactness_2": C2,
                        "lambda_1": lam1, "lambda_2": lam2, "lambda_tilde": lamt,
                    }
                    print("  COLUMNS FOUND:")
                    for cn, cv in col_status.items():
                        print(f"    {cn:<35} {'YES' if cv is not None else 'NO'}")
                    print()

                    # --- metadata ---
                    print("  EVENT METADATA:")
                    if mc is not None:
                        mc_fin = mc[np.isfinite(mc)]
                        if len(mc_fin): print(f"    Chirp mass (source): {np.median(mc_fin):.4f} ± {np.std(mc_fin):.4f} Msun")
                    if q is not None:
                        q_fin = q[np.isfinite(q)]
                        if len(q_fin): print(f"    Mass ratio:          {np.median(q_fin):.4f}")
                    if m1g is not None and m2g is not None:
                        m1_med = float(np.median(m1g[np.isfinite(m1g)]))
                        m2_med = float(np.median(m2g[np.isfinite(m2g)]))
                        print(f"    m1 (source):         {m1_med:.4f} Msun")
                        print(f"    m2 (source):         {m2_med:.4f} Msun")
                        print(f"    Total mass:          {m1_med+m2_med:.4f} Msun")
                    if z is not None:
                        z_fin = z[np.isfinite(z)]
                        if len(z_fin): print(f"    Redshift:            {np.median(z_fin):.4f}")
                    print()

                    if C1 is None or C2 is None:
                        print("  [!] compactness columns missing — cannot run main test.")
                        print()
                        continue

                    # =========================================================
                    # SECTION 1: LVK compactness values
                    # =========================================================
                    print("  ─── SECTION 1: LVK COMPACTNESS ───")
                    C1_med = summarize(C1, "compactness_1 (primary NS)")
                    C2_med = summarize(C2, "compactness_2 (secondary NS)")

                    # mass-weighted C
                    C_mw_med = None
                    if m1g is not None and m2g is not None:
                        joint = (np.isfinite(C1) & (C1>0) & np.isfinite(C2) & (C2>0) &
                                 np.isfinite(m1g) & np.isfinite(m2g))
                        if joint.sum() > 10:
                            C_mw_arr = (m1g[joint]*C1[joint] + m2g[joint]*C2[joint]) / \
                                       (m1g[joint] + m2g[joint])
                            C_mw_med = float(np.median(C_mw_arr))
                            p16_c, p84_c = np.percentile(C_mw_arr, [16, 84])
                            print(f"    mass-weighted C: median={C_mw_med:.4f}  "
                                  f"68%CI=[{p16_c:.4f},{p84_c:.4f}]")

                    if C_mw_med is None:
                        C_mw_med = (C1_med + C2_med) / 2.0
                    print()

                    # =========================================================
                    # SECTION 2: Predicted n
                    # =========================================================
                    print("  ─── SECTION 2: PREDICTED n FROM COMPACTNESS EQUATION ───")
                    print()
                    print(f"  {'Method':<32}  {'C':>6}  {'n_linear':>10}  {'n_beta':>9}  {'in BBH band?'}")
                    print("  " + "─"*72)
                    for lbl, C_val in [
                        ("C1  (primary NS)",   C1_med),
                        ("C2  (secondary NS)", C2_med),
                        ("C_mw (mass-weighted)", C_mw_med),
                    ]:
                        nl = n_linear(C_val)
                        nb = n_beta(C_val)
                        in_band = "YES" if BBH_LO < nl < BBH_HI else "no (sub-band as expected)"
                        print(f"  {lbl:<32}  {C_val:>6.4f}  {nl:>10.4f}  {nb:>9.4f}  {in_band}")
                    print()

                    # =========================================================
                    # SECTION 3: Observed n proxies
                    # =========================================================
                    print("  ─── SECTION 3: OBSERVED n PROXIES ───")
                    print()
                    print("  [A] BINDING ENERGY PROXY (upper bound on available energy):")
                    n_bind_med = None
                    if mb1 is not None and mb2 is not None and m1g is not None and m2g is not None:
                        n_bind = calc_n_from_binding(m1g, m2g, mb1, mb2)
                        n_bind_med = summarize(n_bind, "n_obs proxy — binding energy")
                        if mb1 is not None:
                            mb1_fin = mb1[np.isfinite(mb1)]
                            mb2_fin = mb2[np.isfinite(mb2)]
                            m1_fin  = m1g[np.isfinite(m1g)]
                            m2_fin  = m2g[np.isfinite(m2g)]
                            if len(mb1_fin) and len(m1_fin):
                                be1 = float(np.median(mb1_fin)) - float(np.median(m1_fin))
                                be2 = float(np.median(mb2_fin)) - float(np.median(m2_fin))
                                be_tot = be1 + be2
                                m_tot = float(np.median(m1_fin)) + float(np.median(m2_fin))
                                print(f"    E_bind_1 / M_total = {be1/m_tot:.5f}")
                                print(f"    E_bind_2 / M_total = {be2/m_tot:.5f}")
                                print(f"    E_bind_total / M_total = {be_tot/m_tot:.5f}  "
                                      f"(compare: actual E_GW ~ 0.001)")
                    else:
                        print("    [!] baryonic_mass columns not available")
                    print()

                    print("  [B] PUBLISHED NR RADIATED ENERGY FRACTIONS:")
                    print(f"      Source: Dietrich et al. / Bernuzzi et al. BNS NR simulations")
                    n_rad_meds = {}
                    if m1g is not None and m2g is not None:
                        mt = (m1g + m2g)
                        mt = mt[np.isfinite(mt)]
                        for label, frac in [
                            (f"E_rad/M={ERAD_LO:.4f} (conservative)", ERAD_LO),
                            (f"E_rad/M={ERAD_MED:.4f} (median NR)",    ERAD_MED),
                            (f"E_rad/M={ERAD_HI:.4f} (upper NR)",      ERAD_HI),
                        ]:
                            n_arr = calc_n_from_efrac(mt, frac)
                            med   = summarize(n_arr, label)
                            n_rad_meds[frac] = med
                    print()

                    # =========================================================
                    # SECTION 4: Tidal deformabilities
                    # =========================================================
                    print("  ─── SECTION 4: TIDAL DEFORMABILITIES ───")
                    if lam1 is not None: summarize(lam1, "lambda_1")
                    if lam2 is not None: summarize(lam2, "lambda_2")
                    if lamt is not None: summarize(lamt, "lambda_tilde")
                    print()

                    # Yagi-Yunes C-Lambda consistency check
                    # C ~ 0.371 * Lambda^(-1/5)  (from I-Love-Q universal relations)
                    if lam1 is not None and lam2 is not None:
                        l1_m = float(np.median(lam1[np.isfinite(lam1) & (lam1>0)]))
                        l2_m = float(np.median(lam2[np.isfinite(lam2) & (lam2>0)]))
                        C1_yy = 0.371 * l1_m**(-1.0/5.0)
                        C2_yy = 0.371 * l2_m**(-1.0/5.0)
                        print("  YAGI-YUNES LAMBDA-COMPACTNESS CONSISTENCY CHECK:")
                        print(f"    C1_from_lambda = {C1_yy:.4f}  vs  C1_LVK = {C1_med:.4f}  "
                              f"diff = {abs(C1_yy-C1_med):.4f}")
                        print(f"    C2_from_lambda = {C2_yy:.4f}  vs  C2_LVK = {C2_med:.4f}  "
                              f"diff = {abs(C2_yy-C2_med):.4f}")
                        if abs(C1_yy - C1_med) < 0.05 and abs(C2_yy - C2_med) < 0.05:
                            print("    => LVK compactness consistent with I-Love-Q relation ✓")
                        else:
                            print("    => Discrepancy > 0.05 — LVK may use different EOS assumption")
                        print()

                    # =========================================================
                    # SECTION 5: RESIDUALS & VERDICTS
                    # =========================================================
                    print("  ─── SECTION 5: RESIDUALS & VERDICTS ───")
                    print()

                    # vs binding energy proxy
                    if n_bind_med is not None:
                        for lbl, C_val in [("C1",C1_med),("C2",C2_med),("C_mw",C_mw_med)]:
                            nl = n_linear(C_val)
                            nb = n_beta(C_val)
                            res_l = abs(nl - n_bind_med)
                            res_b = abs(nb - n_bind_med)
                            print(f"  vs binding-energy proxy n_obs={n_bind_med:.4f}  [{lbl}]:")
                            print(f"    linear  residual: {nl:.4f} - {n_bind_med:.4f} = "
                                  f"{nl-n_bind_med:+.4f}  => {verdict_str(res_l)}")
                            print(f"    beta    residual: {nb:.4f} - {n_bind_med:.4f} = "
                                  f"{nb-n_bind_med:+.4f}  => {verdict_str(res_b)}")
                        print()

                    # vs NR radiated energy proxy (median)
                    n_nr_med = n_rad_meds.get(ERAD_MED)
                    if n_nr_med is not None:
                        for lbl, C_val in [("C1",C1_med),("C2",C2_med),("C_mw",C_mw_med)]:
                            nl = n_linear(C_val)
                            nb = n_beta(C_val)
                            res_l = abs(nl - n_nr_med)
                            res_b = abs(nb - n_nr_med)
                            print(f"  vs NR-radiated proxy n_obs={n_nr_med:.4f}  [{lbl}]:")
                            print(f"    linear  residual: {nl:.4f} - {n_nr_med:.4f} = "
                                  f"{nl-n_nr_med:+.4f}  => {verdict_str(res_l)}")
                            print(f"    beta    residual: {nb:.4f} - {n_nr_med:.4f} = "
                                  f"{nb-n_nr_med:+.4f}  => {verdict_str(res_b)}")
                        print()

                    # Framework range check
                    print("  FRAMEWORK RANGE CHECK:")
                    for lbl, C_val in [("C1",C1_med),("C2",C2_med),("C_mw",C_mw_med)]:
                        nl = n_linear(C_val)
                        in_range = N_FLIP < nl < N_BBH
                        below_bbh = nl < BBH_LO
                        print(f"    n_linear({lbl}={C_val:.4f}) = {nl:.4f}  "
                              f"in [n_flip, n_BBH]: {'YES ✓' if in_range else 'NO !'}  "
                              f"below BBH band: {'YES ✓' if below_bbh else 'NO !'}")
                    print()

                    # Compare to v11 NSBH table predictions
                    print("  v11 NSBH TABLE COMPARISON:")
                    print("  (What C value does v11 assign these known NSBH events?)")
                    nsbh_table = [
                        ("GW190814", 0.18, 4.189),
                        ("GW200105", 0.32, 4.688),
                        ("GW200115", 0.34, 4.741),
                        ("GW230518", 0.28, 4.552),
                    ]
                    print(f"  {'Event':<12}  {'C_implied':>10}  {'n_obs(v11)':>10}  {'n_pred':>8}")
                    for ev, C_ev, n_ev in nsbh_table:
                        print(f"  {ev:<12}  {C_ev:>10.2f}  {n_ev:>10.3f}  {n_linear(C_ev):>8.4f}")
                    print(f"  {'GW190425':12}  {C_mw_med:>10.4f}  {'(this run)':>10}  "
                          f"{n_linear(C_mw_med):>8.4f}")
                    print()

                    # collect lambda/YY values for CSV
                    yy_data = {}
                    if lam1 is not None and lam2 is not None:
                        l1_m2 = float(np.median(lam1[np.isfinite(lam1) & (lam1>0)])) if np.any(np.isfinite(lam1) & (lam1>0)) else 1.0
                        l2_m2 = float(np.median(lam2[np.isfinite(lam2) & (lam2>0)])) if np.any(np.isfinite(lam2) & (lam2>0)) else 1.0
                        C1_yy2 = 0.371 * l1_m2**(-1.0/5.0)
                        C2_yy2 = 0.371 * l2_m2**(-1.0/5.0)
                        yy_data = {"C1_yy": C1_yy2, "C2_yy": C2_yy2,
                                   "diff1": abs(C1_yy2 - C1_med), "diff2": abs(C2_yy2 - C2_med)}
                    lam1_m = float(np.median(lam1[np.isfinite(lam1)])) if lam1 is not None else None
                    lam2_m = float(np.median(lam2[np.isfinite(lam2)])) if lam2 is not None else None
                    lamt_m = float(np.median(lamt[np.isfinite(lamt)])) if lamt is not None else None

                    summary_rows.append({
                        "file": fname, "wf_key": wf_key,
                        "C1": C1_med, "C2": C2_med, "C_mw": C_mw_med,
                        "n_lin_mw": n_linear(C_mw_med),
                        "n_beta_mw": n_beta(C_mw_med),
                        "n_obs_binding": n_bind_med,
                        "n_obs_nr_med": n_nr_med,
                        "yagi_yunes": yy_data,
                        "lam1_med": lam1_m, "lam2_med": lam2_m, "lamt_med": lamt_m,
                        "n_samples": len(ps),
                    })

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print()

    if not summary_rows:
        print("No events processed."); return

    print(f"{'Waveform':<40}  {'C_mw':>6}  {'n_lin':>7}  {'n_beta':>7}  "
          f"{'n_obs_bind':>12}  {'n_obs_NR':>10}")
    print("─" * 90)
    for r in summary_rows:
        nob = f"{r['n_obs_binding']:.4f}" if r['n_obs_binding'] else "  N/A  "
        nnr = f"{r['n_obs_nr_med']:.4f}"  if r['n_obs_nr_med'] else "  N/A  "
        print(f"{r['wf_key']:<40}  {r['C_mw']:>6.4f}  {r['n_lin_mw']:>7.4f}  "
              f"{r['n_beta_mw']:>7.4f}  {nob:>12}  {nnr:>10}")
    print()

    # =========================================================================
    # CSV EXPORT
    # =========================================================================
    import csv as csv_mod
    csv_path = "seraphim_tidal_results.csv"
    fieldnames = [
        "event_file","wf_key","n_samples",
        "C1_median","C2_median","C_mw_median",
        "n_pred_linear_C1","n_pred_linear_C2","n_pred_linear_Cmw",
        "n_pred_beta_C1","n_pred_beta_C2","n_pred_beta_Cmw",
        "n_obs_binding_proxy","n_obs_NR_median_proxy",
        "resid_linear_Cmw_vs_binding","resid_linear_Cmw_vs_NR",
        "resid_beta_Cmw_vs_binding","resid_beta_Cmw_vs_NR",
        "framework_in_range_C1","framework_in_range_C2","framework_in_range_Cmw",
        "below_BBH_band_C1","below_BBH_band_C2","below_BBH_band_Cmw",
        "C1_YY","C2_YY","YY_diff_C1","YY_diff_C2",
        "lambda1_median","lambda2_median","lambda_tilde_median",
    ]
    def _r6(x): return round(float(x), 6) if x is not None and x == x else ""
    def _res(pred, obs): return _r6(pred - obs) if obs else ""
    def _inr(nl): return "YES" if N_FLIP < nl < N_BBH else "NO"
    def _sub(nl): return "YES" if nl < BBH_LO else "NO"

    with open(csv_path, "w", newline="") as cf:
        writer = csv_mod.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            C1, C2, Cmw = r["C1"], r["C2"], r["C_mw"]
            nl1, nl2, nlm = n_linear(C1), n_linear(C2), n_linear(Cmw)
            nb1, nb2, nbm = n_beta(C1),   n_beta(C2),   n_beta(Cmw)
            nob, nnr = r["n_obs_binding"], r["n_obs_nr_med"]
            yy = r.get("yagi_yunes", {})
            writer.writerow({
                "event_file": r["file"], "wf_key": r["wf_key"],
                "n_samples": r.get("n_samples",""),
                "C1_median": _r6(C1), "C2_median": _r6(C2), "C_mw_median": _r6(Cmw),
                "n_pred_linear_C1": _r6(nl1), "n_pred_linear_C2": _r6(nl2), "n_pred_linear_Cmw": _r6(nlm),
                "n_pred_beta_C1":   _r6(nb1), "n_pred_beta_C2":   _r6(nb2), "n_pred_beta_Cmw":   _r6(nbm),
                "n_obs_binding_proxy": _r6(nob), "n_obs_NR_median_proxy": _r6(nnr),
                "resid_linear_Cmw_vs_binding": _res(nlm, nob),
                "resid_linear_Cmw_vs_NR":      _res(nlm, nnr),
                "resid_beta_Cmw_vs_binding":   _res(nbm, nob),
                "resid_beta_Cmw_vs_NR":        _res(nbm, nnr),
                "framework_in_range_C1":  _inr(nl1), "framework_in_range_C2": _inr(nl2), "framework_in_range_Cmw": _inr(nlm),
                "below_BBH_band_C1": _sub(nl1), "below_BBH_band_C2": _sub(nl2), "below_BBH_band_Cmw": _sub(nlm),
                "C1_YY": _r6(yy.get("C1_yy")) if yy else "",
                "C2_YY": _r6(yy.get("C2_yy")) if yy else "",
                "YY_diff_C1": _r6(yy.get("diff1")) if yy else "",
                "YY_diff_C2": _r6(yy.get("diff2")) if yy else "",
                "lambda1_median":      _r6(r.get("lam1_med")),
                "lambda2_median":      _r6(r.get("lam2_med")),
                "lambda_tilde_median": _r6(r.get("lamt_med")),
            })
    print(f"[*] Results saved to: {csv_path}")
    print()

    print("=" * 72)
    print("WHAT THIS TEST CAN AND CANNOT CONCLUDE")
    print("=" * 72)
    print()
    print("CAN conclude:")
    print("  1. Whether LVK compactness values are consistent with Yagi-Yunes")
    print("     I-Love-Q relation (internal posterior sanity check)")
    print("  2. Whether n_predicted(C) falls in the physically expected sub-BBH")
    print("     range for a BNS (n_flip < n < 4.76)")
    print("  3. Where GW190425 sits on the compactness equation line n(C)")
    print("  4. How that predicted n compares to the v11 NSBH table (C~0.18-0.34)")
    print()
    print("CANNOT cleanly conclude:")
    print("  - A direct residual falsification, because NRTidal has no final_mass.")
    print("  - The binding energy n is an upper bound (E_bind >> E_GW for BNS).")
    print("  - The NR radiated fraction is from simulations, not this posterior.")
    print()
    print("TO CLOSE THIS TEST CLEANLY:")
    print("  - GW170817 with full NR BNS waveform (if final_mass is in posteriors)")
    print("  - O5 BNS event with matched BBH surrogate mass comparison")
    print("  - Use tidal Lambda => NICER EOS => R_NS => C => predict n,")
    print("    then compare against n_obs from a hypothetical surrogate final_mass")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python seraphim_tidal_test.py /path/to/folder_or_file.h5")
        sys.exit(1)
    run(sys.argv[1])
