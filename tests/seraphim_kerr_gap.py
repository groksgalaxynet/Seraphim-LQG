"""
seraphim_kerr_gap.py
=====================
Make-or-break test: does replacing the Schwarzschild ISCO frequency
with the exact Kerr ISCO frequency (using per-event chi_eff posteriors)
reduce the scatter in the gap and move the median gap closer to 1/alpha?

TWO GAPS COMPUTED PER EVENT:

  gap_schwarz = n_carrier_schwarz - n_seraphim
                n_carrier_schwarz = log2(nu_P / f_ISCO_Schwarz)
                f_ISCO_Schwarz = c^3 / (6*sqrt(6)*pi*G*M_total)

  gap_kerr    = n_carrier_kerr - n_seraphim
                n_carrier_kerr = log2(nu_P / f_ISCO_Kerr)
                f_ISCO_Kerr uses exact Bardeen-Press-Teukolsky (1972)
                formula with per-event median chi_eff

THE VERDICT:
  If gap_kerr std < gap_schwarz std  -> Kerr is the missing physics
  If gap_kerr median closer to 1/alpha -> spin closes the offset
  If neither                         -> other physics dominates

ALSO TESTS:
  - Per-spin-bin gap comparison (retrograde / low / prograde)
  - Does chi_eff partial correlation on gap_kerr vanish?
    (if spin was the driver it should disappear after correction)
  - Per-catalog consistency

WALKS current directory and ALL subdirectories for .h5 / .hdf5 files.

USAGE:
    cd /path/to/your/hdf5/folder
    python seraphim_kerr_gap.py

    Or:
    python seraphim_kerr_gap.py --dir /path/to/folder

OUTPUTS:
    seraphim_kerr_results.csv     <- per-event, both gaps
    seraphim_kerr_summary.csv     <- per-catalog statistics
"""

import os, sys, glob, math, re, csv, argparse
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("ERROR: h5py not installed.  pip install h5py")

try:
    from scipy.stats import spearmanr, pearsonr, ttest_1samp, levene, kruskal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — pip install scipy for full analysis")
    print()

# ── Constants ─────────────────────────────────────────────────────────────────
G         = 6.67430e-11
C_LIGHT   = 2.99792458e8
M_SUN     = 1.98847e30
ALPHA     = 0.007297352569
ALPHA_INV = 1.0 / ALPHA          # 137.0360
K0        = 1.1467e84
NU_P      = 1.8549e43
SQRT_J    = math.sqrt(0.75)
N_BBH     = math.log2(NU_P / math.sqrt(K0 / (SQRT_J * 6.09)))

E_NR_COEFF = 0.0842  # Healy et al. 2014 fallback for GWTC-2.1

print("=" * 68)
print("  SERAPHIM KERR GAP TEST — Schwarzschild vs Kerr ISCO")
print("=" * 68)
print()
print("  Constants:")
print("    n_BBH       = %.6f" % N_BBH)
print("    1/alpha     = %.6f" % ALPHA_INV)
print()

# ── Kerr ISCO physics ─────────────────────────────────────────────────────────
def kerr_isco_rg(chi):
    """
    ISCO radius in gravitational radii (r_g = G*M/c^2).
    Bardeen, Press & Teukolsky (1972), ApJ 178, 347.
    chi = dimensionless spin parameter, signed (-1 to +1).
    Prograde orbit for chi > 0, retrograde for chi < 0.
    """
    a = min(abs(chi), 0.9999)  # spin magnitude, clamped
    Z1 = 1.0 + (1.0 - a**2)**(1.0/3.0) * (
         (1.0 + a)**(1.0/3.0) + (1.0 - a)**(1.0/3.0))
    Z2 = math.sqrt(3.0 * a**2 + Z1**2)
    s  = 1.0 if chi >= 0.0 else -1.0
    r  = 3.0 + Z2 - s * math.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0*Z2))
    return r

def f_gw_schwarz(M_msun):
    """GW frequency at Schwarzschild ISCO (quadrupole, Hz)."""
    return C_LIGHT**3 / (6.0 * math.sqrt(6.0) * math.pi
                         * G * M_msun * M_SUN)

def f_gw_kerr(M_msun, chi):
    """
    GW frequency at Kerr ISCO (quadrupole, Hz).
    f_GW = 2 * f_orbital = Omega_orb / pi
    Omega_orb = c^3/(G*M) * 1/(r_isco^1.5 + a)
    where r_isco is in units of r_g and a = chi (signed).
    Reduces exactly to Schwarzschild at chi=0.
    """
    M_kg   = M_msun * M_SUN
    r_isco = kerr_isco_rg(chi)
    a      = max(-0.9999, min(0.9999, chi))
    Omega  = C_LIGHT**3 / (G * M_kg) / (r_isco**1.5 + a)
    return Omega / math.pi   # = 2*Omega/(2*pi)

def n_from_f(f_hz):
    if f_hz and f_hz > 0:
        return math.log2(NU_P / f_hz)
    return None

# ── HDF5 navigation ───────────────────────────────────────────────────────────
PREFER_WF = ["IMRPhenomXPHM","IMRPhenomXP","IMRPhenomPv2",
             "SEOBNRv4PHM","SEOBNRv4P","NRSur7dq4",
             "IMRPhenomD","SEOBNRv4"]
NON_BBH   = ["NRTidal","NRTidalv2","NSBH","BNS"]

def is_non_bbh(wf):
    ku = (wf or "").upper()
    return any(x.upper() in ku for x in NON_BBH)

def get_posteriors(hf):
    keys = list(hf.keys())
    if "posterior_samples" in keys:
        ps = hf["posterior_samples"]
        if hasattr(ps,"dtype") and ps.dtype is not None:
            return ps, "posterior_samples"
    wf_cands = [k for k in keys
                if any(x in k for x in ["IMR","SEOBNR","NR","EOB"])]
    wf_cands.sort(key=lambda k:(
        next((i for i,p in enumerate(PREFER_WF) if p.upper() in k.upper()),999)
        + (100 if is_non_bbh(k) else 0)))
    for wf in wf_cands:
        grp = hf[wf]
        if hasattr(grp,"keys") and "posterior_samples" in grp:
            ps = grp["posterior_samples"]
            if hasattr(ps,"dtype") and ps.dtype is not None:
                return ps, wf
        if hasattr(grp,"dtype") and grp.dtype is not None:
            return grp, wf
        if hasattr(grp,"keys"):
            for sub in grp:
                item = grp[sub]
                if hasattr(item,"dtype") and item.dtype is not None:
                    return item, wf
    return None, None

def col_arr(ps, *names):
    avail = ps.dtype.names or []
    for n in names:
        if n in avail:
            try:
                a = np.array(ps[n][:], dtype=float)
                a = a[np.isfinite(a)]
                if len(a) > 0: return a
            except Exception: pass
    return None

def med_col(ps, *names):
    a = col_arr(ps, *names)
    return float(np.median(a)) if a is not None else None

# ── Catalog / event detection ─────────────────────────────────────────────────
def detect_catalog(fpath):
    p = fpath.replace("\\","/")
    if "6513631"  in p or "gwtc2p1" in p.lower(): return "GWTC-2.1"
    if "8177023"  in p or "gwtc3"   in p.lower(): return "GWTC-3"
    if "16053484" in p or "gwtc4"   in p.lower(): return "GWTC-4"
    b = os.path.basename(fpath).upper()
    if "GWTC2" in b or "GWTC2P1" in b: return "GWTC-2.1"
    if "GWTC3" in b:                    return "GWTC-3"
    if "GWTC4" in b:                    return "GWTC-4"
    return "UNKNOWN"

def event_name(fpath):
    m = re.search(r"(GW\d{6}(?:_\d{6})?)", os.path.basename(fpath))
    return m.group(1) if m else os.path.basename(fpath)[:24]

# ── Seraphim n from energy loss ───────────────────────────────────────────────
def seraphim_n(E_rad_arr, M_total_arr):
    L     = min(len(E_rad_arr), len(M_total_arr))
    E, M  = E_rad_arr[:L], M_total_arr[:L]
    valid = (E > 0) & (M > 0) & (E < M)
    if valid.sum() < 10: return None, None
    n_star = E[valid] / (M[valid] * ALPHA)
    nu_s   = np.sqrt(K0 / (SQRT_J * n_star))
    n_s    = np.log2(NU_P / nu_s)
    ok     = np.isfinite(n_s) & (n_s > 2.0) & (n_s < 10.0)
    if ok.sum() < 10: return None, None
    return float(np.median(n_s[ok])), float(np.std(n_s[ok]))

# ── Per-file processing ───────────────────────────────────────────────────────
def process(fpath):
    try:
        with h5py.File(fpath,"r") as hf:
            ps, wf = get_posteriors(hf)
            if ps is None: return None, "no posterior dataset"
            if is_non_bbh(wf): return None, "non-BBH waveform"

            avail = set(ps.dtype.names or [])

            # Total mass
            M_tot_arr = col_arr(ps,"total_mass_source","total_mass")
            if M_tot_arr is None: return None, "no total_mass"
            M_tot_med = float(np.median(M_tot_arr))

            # chi_eff — need full posterior array for per-sample Kerr correction
            chi_arr = col_arr(ps,"chi_eff")
            chi_med = float(np.median(chi_arr)) if chi_arr is not None else 0.0

            # Radiated energy — strategy waterfall
            E_arr, M_aligned, strategy = None, None, ""
            for fm in ["final_mass_source_non_evolved","final_mass_source",
                       "final_mass_non_evolved","final_mass"]:
                if fm in avail:
                    fm_arr = col_arr(ps, fm)
                    if fm_arr is not None:
                        L  = min(len(M_tot_arr), len(fm_arr))
                        e  = M_tot_arr[:L] - fm_arr[:L]
                        ok = (e>0)&(e<M_tot_arr[:L])
                        if ok.sum()>=10:
                            E_arr=e[ok]; M_aligned=M_tot_arr[:L][ok]
                            strategy="M_total-%s"%fm; break

            if E_arr is None:
                for rc in ["radiated_energy_non_evolved","radiated_energy"]:
                    if rc in avail:
                        ra = col_arr(ps,rc)
                        if ra is not None:
                            L  = min(len(M_tot_arr),len(ra))
                            ok = (ra[:L]>0)&(ra[:L]<M_tot_arr[:L])
                            if ok.sum()>=10:
                                E_arr=ra[:L][ok]; M_aligned=M_tot_arr[:L][ok]
                                strategy=rc; break

            if E_arr is None:
                mc_arr = col_arr(ps,"chirp_mass_source","chirp_mass")
                if mc_arr is not None:
                    L  = min(len(M_tot_arr),len(mc_arr))
                    e  = E_NR_COEFF * mc_arr[:L]
                    ok = (e>0)&(e<M_tot_arr[:L])
                    if ok.sum()>=10:
                        E_arr=e[ok]; M_aligned=M_tot_arr[:L][ok]
                        strategy="NR_fit"

            if E_arr is None: return None, "no energy loss"

            n_sph, n_sph_std = seraphim_n(E_arr, M_aligned)
            if n_sph is None: return None, "seraphim_n failed"

            # Schwarzschild carrier
            f_s  = f_gw_schwarz(M_tot_med)
            n_cs = n_from_f(f_s)

            # Kerr carrier — median chi
            f_k  = f_gw_kerr(M_tot_med, chi_med)
            n_ck = n_from_f(f_k)

            # Kerr carrier — per-sample chi (more precise)
            # Use per-sample chi_eff if available, else fall back to median
            n_ck_ps = None
            if chi_arr is not None and len(chi_arr) >= 10:
                # Compute per-sample Kerr f and take median n
                L  = min(len(chi_arr), len(M_tot_arr))
                ns_kerr = []
                for i in range(0, L, max(1, L//2000)):  # sample at most 2000 points
                    try:
                        fki = f_gw_kerr(float(M_tot_arr[i]), float(chi_arr[i]))
                        nki = n_from_f(fki)
                        if nki and 130 < nki < 145:
                            ns_kerr.append(nki)
                    except Exception:
                        pass
                if len(ns_kerr) >= 10:
                    n_ck_ps = float(np.median(ns_kerr))

            # Use per-sample if available, else median-chi Kerr
            n_ck_best = n_ck_ps if n_ck_ps is not None else n_ck

            # Kerr correction magnitude
            kerr_delta_n = n_ck_best - n_cs if (n_cs and n_ck_best) else None

            return dict(
                wf=wf, strategy=strategy,
                M_tot=M_tot_med,
                chi_med=chi_med,
                chi_std=float(np.std(chi_arr)) if chi_arr is not None else None,
                redshift=med_col(ps,"redshift"),
                lum_dist=med_col(ps,"luminosity_distance"),
                mass_ratio=med_col(ps,"mass_ratio","inverted_mass_ratio"),
                n_seraphim=n_sph, n_sph_std=n_sph_std,
                n_carrier_schwarz=n_cs,
                n_carrier_kerr=n_ck_best,
                kerr_delta_n=kerr_delta_n,
                used_per_sample_chi=(n_ck_ps is not None),
            ), None

    except Exception as ex:
        return None, str(ex)

# ── File discovery ────────────────────────────────────────────────────────────
def find_files(root):
    found = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".h5") or fn.lower().endswith(".hdf5"):
                found.append(os.path.join(dp, fn))
    return sorted(set(found))

# ── Main ──────────────────────────────────────────────────────────────────────
def run(root):
    print("  Scanning: %s" % os.path.abspath(root))
    files = find_files(root)
    print("  Found %d HDF5 files" % len(files))
    if not files:
        sys.exit("No HDF5 files found under %s" % root)
    print()

    rows, skipped = [], []
    for fpath in files:
        ename = event_name(fpath)
        cat   = detect_catalog(fpath)
        res, reason = process(fpath)
        if res is None:
            skipped.append((ename, reason)); continue

        n_sph = res["n_seraphim"]
        n_cs  = res["n_carrier_schwarz"]
        n_ck  = res["n_carrier_kerr"]
        if not all([n_sph, n_cs, n_ck]):
            skipped.append((ename,"null n value")); continue

        gap_s = n_cs - n_sph
        gap_k = n_ck - n_sph

        def f4(v): return round(float(v),4) if v is not None else ""
        def f6(v): return round(float(v),6) if v is not None else ""
        def fb(v): return str(v)

        rows.append({
            "event"               : ename,
            "catalog"             : cat,
            "waveform"            : (res["wf"] or "")[:50],
            "erad_strategy"       : res["strategy"],
            "per_sample_chi"      : fb(res["used_per_sample_chi"]),
            "M_total_Msun"        : f4(res["M_tot"]),
            "chi_eff_med"         : f4(res["chi_med"]),
            "chi_eff_std"         : f4(res["chi_std"]),
            "redshift"            : f4(res["redshift"]),
            "lum_dist_Mpc"        : f4(res["lum_dist"]),
            "mass_ratio"          : f4(res["mass_ratio"]),
            "n_seraphim"          : f6(n_sph),
            "n_sph_std"           : f6(res["n_sph_std"]),
            "n_carrier_schwarz"   : f6(n_cs),
            "n_carrier_kerr"      : f6(n_ck),
            "kerr_delta_n"        : f6(res["kerr_delta_n"]),
            "n_BBH_theory"        : round(N_BBH,6),
            "gap_schwarz"         : f6(gap_s),
            "gap_kerr"            : f6(gap_k),
            "alpha_inv"           : round(ALPHA_INV,6),
            "delta_schwarz"       : f6(gap_s - ALPHA_INV),
            "delta_kerr"          : f6(gap_k - ALPHA_INV),
            "log2_M"              : f6(math.log2(res["M_tot"])),
        })

    print("  Processed: %d events" % len(rows))
    print("  Skipped:   %d" % len(skipped))
    if skipped:
        from collections import Counter
        for r, c in Counter(r for _,r in skipped).most_common(6):
            print("    [%3d] %s" % (c, r[:70]))
    print()
    if not rows: sys.exit("No events processed.")

    # Write per-event CSV
    out = "seraphim_kerr_results.csv"
    with open(out,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("  Per-event CSV: %s  (%d rows)" % (out, len(rows)))
    print()

    # ── Arrays ────────────────────────────────────────────────────────────
    def arr(c):
        return np.array([float(r[c]) for r in rows
                         if r.get(c,"") not in ("",None,"None")])
    def al2(c1,c2):
        p=[(float(r[c1]),float(r[c2])) for r in rows
           if r.get(c1,"") not in ("",None) and r.get(c2,"") not in ("",None)]
        if not p: return np.array([]),np.array([])
        a,b=zip(*p); return np.array(a),np.array(b)
    def al3(c1,c2,c3):
        p=[(float(r[c1]),float(r[c2]),float(r[c3])) for r in rows
           if all(r.get(c,"") not in ("",None) for c in [c1,c2,c3])]
        if not p: return np.array([]),np.array([]),np.array([])
        a,b,c=zip(*p); return np.array(a),np.array(b),np.array(c)

    gs  = arr("gap_schwarz")
    gk  = arr("gap_kerr")
    ds  = arr("delta_schwarz")
    dk  = arr("delta_kerr")
    chi = arr("chi_eff_med")

    sep = "=" * 68
    print(sep)
    print("  THE MAIN EVENT: Schwarzschild vs Kerr gap comparison")
    print(sep)
    print()
    print("  %-28s  %10s  %10s  %10s" % ("Metric","Schwarz","Kerr","Change"))
    print("  " + "-"*62)

    med_s = np.median(gs); med_k = np.median(gk)
    std_s = np.std(gs);    std_k = np.std(gk)
    pct_s = abs(np.median(ds))/ALPHA_INV*100
    pct_k = abs(np.median(dk))/ALPHA_INV*100

    print("  %-28s  %10.4f  %10.4f  %+10.4f" % (
          "Median gap", med_s, med_k, med_k-med_s))
    print("  %-28s  %10.4f  %10.4f  %+10.4f" % (
          "Std of gap", std_s, std_k, std_k-std_s))
    print("  %-28s  %10.4f  %10.4f  %+10.4f" % (
          "Delta from 1/alpha", np.median(ds), np.median(dk),
          np.median(dk)-np.median(ds)))
    print("  %-28s  %9.3f%%  %9.3f%%  %+9.3f%%" % (
          "Pct from 1/alpha", pct_s, pct_k, pct_k-pct_s))
    print()
    print("  1/alpha = %.6f" % ALPHA_INV)
    print()

    # Std reduction
    std_reduction = (std_s - std_k) / std_s * 100
    if std_reduction > 0:
        print("  ** Kerr REDUCES gap scatter by %.2f%% **" % std_reduction)
    else:
        print("  Kerr INCREASES scatter by %.2f%%" % abs(std_reduction))

    # Median shift
    med_shift = abs(np.median(dk)) - abs(np.median(ds))
    if med_shift < 0:
        print("  ** Kerr CLOSES gap toward 1/alpha by %.4f octaves **" % abs(med_shift))
    else:
        print("  Kerr moves gap AWAY from 1/alpha by %.4f octaves" % med_shift)
    print()

    if HAS_SCIPY:
        # Does chi_eff correlation vanish after Kerr correction?
        g_c_s, c_s = al2("gap_schwarz","chi_eff_med")
        g_c_k, c_k = al2("gap_kerr","chi_eff_med")
        if len(g_c_s)>=5:
            rho_s, p_s = spearmanr(c_s, g_c_s)
            rho_k, p_k = spearmanr(c_k, g_c_k)
            print("  Chi_eff partial test (does spin signal disappear?):")
            print("    r(chi, gap_schwarz) = %+.4f   p=%.2e" % (rho_s, p_s))
            print("    r(chi, gap_kerr)    = %+.4f   p=%.2e" % (rho_k, p_k))
            if abs(rho_k) < abs(rho_s) * 0.5:
                print("    ** YES: spin correlation halved after Kerr correction **")
                print("       Kerr ISCO is accounting for the spin-gap relationship")
            elif p_k > 0.05:
                print("    ** YES: spin correlation vanishes after Kerr correction **")
                print("       Spin was entirely the ISCO shift — Kerr accounts for it")
            else:
                print("    Spin correlation persists — other spin physics remains")
            print()

        # Partial: chi | log2M on both gaps
        g3s, c3s, m3s = al3("gap_schwarz","chi_eff_med","log2_M")
        g3k, c3k, m3k = al3("gap_kerr",   "chi_eff_med","log2_M")
        if len(g3s)>=5:
            def partial_r(x,y,z):
                cx=np.polyfit(z,x,1); cy=np.polyfit(z,y,1)
                return pearsonr(x-np.polyval(cx,z), y-np.polyval(cy,z))
            rps, pps = partial_r(c3s, g3s, m3s)
            rpk, ppk = partial_r(c3k, g3k, m3k)
            print("  Partial r(chi, gap | log2M):")
            print("    Schwarzschild: r=%+.4f  p=%.2e" % (rps, pps))
            print("    Kerr:          r=%+.4f  p=%.2e" % (rpk, ppk))
            if ppk > 0.05 and pps < 0.05:
                print("    ** SPIN SIGNAL REMOVED by Kerr correction **")
            elif abs(rpk) < abs(rps):
                print("    Partial spin correlation reduced by Kerr correction")
            print()

        # Levene test: is Kerr gap distribution tighter?
        stat_lev, p_lev = levene(gs, gk)
        print("  Levene test (equal variance Schwarz vs Kerr): p=%.4e" % p_lev)
        if p_lev < 0.05:
            if std_k < std_s:
                print("  ** Kerr gap is SIGNIFICANTLY tighter (p<0.05) **")
            else:
                print("  Kerr gap is significantly WIDER (p<0.05)")
        else:
            print("  No significant variance difference between Schwarz and Kerr")
        print()

    # ── Spin bins ─────────────────────────────────────────────────────────
    print("  Gap by spin bin:")
    bins = [
        ("retrograde  chi < -0.1", lambda r: float(r["chi_eff_med"]) < -0.1),
        ("near-zero  |chi|<= 0.1", lambda r: abs(float(r["chi_eff_med"])) <= 0.1),
        ("prograde    chi >  0.1", lambda r: float(r["chi_eff_med"]) > 0.1),
    ]
    spin_groups_s, spin_groups_k = [], []
    for label, fn in bins:
        sub = [r for r in rows if r.get("chi_eff_med","") not in ("",None)
               and r.get("gap_schwarz","") not in ("",None)
               and fn(r)]
        if not sub: continue
        gs_b = np.array([float(r["gap_schwarz"]) for r in sub])
        gk_b = np.array([float(r["gap_kerr"])    for r in sub])
        spin_groups_s.append(gs_b)
        spin_groups_k.append(gk_b)
        print("  %-36s  N=%3d  gap_S=%.4f(±%.4f)  gap_K=%.4f(±%.4f)"
              "  delta_S=%+.4f  delta_K=%+.4f" % (
              label, len(sub),
              np.median(gs_b), np.std(gs_b),
              np.median(gk_b), np.std(gk_b),
              np.median(gs_b)-ALPHA_INV,
              np.median(gk_b)-ALPHA_INV))
    print()

    if HAS_SCIPY and len(spin_groups_s) >= 2:
        stat_s, p_s = kruskal(*spin_groups_s)
        stat_k, p_k = kruskal(*spin_groups_k)
        print("  Kruskal-Wallis across spin bins:")
        print("    Schwarzschild gap: p=%.4e  %s" % (
              p_s, "** differs across spin bins **" if p_s<0.05 else "consistent"))
        print("    Kerr gap:          p=%.4e  %s" % (
              p_k, "** still differs **" if p_k<0.05 else "** spin structure removed **"))
        print()

    # ── Per-catalog ────────────────────────────────────────────────────────
    print("  Per-catalog:")
    sum_rows = []
    for cat in ["GWTC-2.1","GWTC-3","GWTC-4","UNKNOWN"]:
        sub = [r for r in rows if r["catalog"]==cat
               and r.get("gap_schwarz","") not in ("",None)]
        if not sub: continue
        gs_c = np.array([float(r["gap_schwarz"]) for r in sub])
        gk_c = np.array([float(r["gap_kerr"])    for r in sub])
        zs   = [float(r["redshift"]) for r in sub
                if r.get("redshift","") not in ("",None)]
        print("    %s  N=%3d  "
              "gap_S=%.4f(±%.4f) d=%+.4f  "
              "gap_K=%.4f(±%.4f) d=%+.4f  "
              "z=%.3f" % (
              cat, len(sub),
              np.median(gs_c), np.std(gs_c), np.median(gs_c)-ALPHA_INV,
              np.median(gk_c), np.std(gk_c), np.median(gk_c)-ALPHA_INV,
              np.median(zs) if zs else -1))
        sum_rows.append({
            "catalog"             : cat,
            "N"                   : len(sub),
            "median_gap_schwarz"  : round(float(np.median(gs_c)),4),
            "std_gap_schwarz"     : round(float(np.std(gs_c)),4),
            "delta_schwarz"       : round(float(np.median(gs_c))-ALPHA_INV,4),
            "pct_schwarz"         : round(abs(float(np.median(gs_c))-ALPHA_INV)/ALPHA_INV*100,4),
            "median_gap_kerr"     : round(float(np.median(gk_c)),4),
            "std_gap_kerr"        : round(float(np.std(gk_c)),4),
            "delta_kerr"          : round(float(np.median(gk_c))-ALPHA_INV,4),
            "pct_kerr"            : round(abs(float(np.median(gk_c))-ALPHA_INV)/ALPHA_INV*100,4),
            "std_reduction_pct"   : round((float(np.std(gs_c))-float(np.std(gk_c)))/float(np.std(gs_c))*100,3),
            "alpha_inv"           : round(ALPHA_INV,6),
            "median_redshift"     : round(np.median(zs),4) if zs else "",
        })
    print()

    # ALL row
    zs_all = [float(r["redshift"]) for r in rows
              if r.get("redshift","") not in ("",None)]
    sum_rows.append({
        "catalog"           : "ALL",
        "N"                 : len(rows),
        "median_gap_schwarz": round(float(np.median(gs)),4),
        "std_gap_schwarz"   : round(float(np.std(gs)),4),
        "delta_schwarz"     : round(float(np.median(ds)),4),
        "pct_schwarz"       : round(pct_s,4),
        "median_gap_kerr"   : round(float(np.median(gk)),4),
        "std_gap_kerr"      : round(float(np.std(gk)),4),
        "delta_kerr"        : round(float(np.median(dk)),4),
        "pct_kerr"          : round(pct_k,4),
        "std_reduction_pct" : round((std_s-std_k)/std_s*100,3),
        "alpha_inv"         : round(ALPHA_INV,6),
        "median_redshift"   : round(np.median(zs_all),4) if zs_all else "",
    })

    out_s = "seraphim_kerr_summary.csv"
    with open(out_s,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
        w.writeheader(); w.writerows(sum_rows)
    print("  Summary CSV: %s" % out_s)
    print()

    # ── VERDICT ───────────────────────────────────────────────────────────
    print(sep)
    print("  VERDICT")
    print(sep)
    print()
    print("  Schwarzschild:  median gap=%.4f  std=%.4f  delta=%+.4f (%.3f%%)" % (
          med_s, std_s, np.median(ds), pct_s))
    print("  Kerr:           median gap=%.4f  std=%.4f  delta=%+.4f (%.3f%%)" % (
          med_k, std_k, np.median(dk), pct_k))
    print()
    print("  Std change:     %+.4f octaves  (%+.2f%%)" % (std_k-std_s, -std_reduction))
    print("  Median shift:   %+.4f octaves toward 1/alpha" % (-med_shift))
    print()

    if std_k < std_s and abs(np.median(dk)) < abs(np.median(ds)):
        print("  *** KERR IS THE CORRECTION ***")
        print("  Gap tightens AND moves toward 1/alpha with Kerr ISCO.")
        print("  The rotating black hole geometry is embedded in the bridge")
        print("  between LQG activation depth and EM carrier frequency.")
    elif std_k < std_s:
        print("  ** Kerr tightens the gap scatter **")
        print("  Spin accounts for some of the n_carrier spread.")
        print("  But median shift direction needs interpretation.")
    elif abs(np.median(dk)) < abs(np.median(ds)):
        print("  ** Kerr moves median gap toward 1/alpha **")
        print("  But scatter is not reduced — other physics also in the gap.")
    else:
        print("  x Kerr correction does not improve the gap.")
        print("  The spin signal in the partial correlation is real,")
        print("  but ISCO shift is not the dominant mechanism.")
        print("  Consider: spin modulates n_seraphim (face activation),")
        print("  not just n_carrier. That effect is already in v17.2.")
    print()
    print("  Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir","-d", default=".",
        help="Root dir to scan (default: current). All subdirs searched.")
    run(ap.parse_args().dir)
