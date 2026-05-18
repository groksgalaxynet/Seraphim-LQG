"""
seraphim_kerr_remnant.py
========================
Repeat the Kerr gap test using REMNANT spin (final_spin or
final_spin_non_evolved from the posteriors) instead of binary chi_eff.

The previous test used chi_eff (binary effective spin) which averages
near zero. The actual ISCO determining n_carrier is the REMNANT BH,
whose spin a_f ~ 0.686 even for non-spinning binaries.

Hypothesis: using the correct remnant spin tightens the gap std
and shifts the median toward 1/alpha by ~1 octave.

If true: the Robertson vs Heisenberg 1-octave separation and the
Kerr remnant correction are the SAME topological fact (chi(S^2)=2)
seen from two different sides of the framework.

OUTPUTS:
    seraphim_kerr_remnant_results.csv
    seraphim_kerr_remnant_summary.csv

USAGE:
    python seraphim_kerr_remnant.py          # scans current dir
    python seraphim_kerr_remnant.py --dir /path/to/hdf5
"""

import os, sys, re, csv, math, argparse
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("pip install h5py")

try:
    from scipy.stats import spearmanr, pearsonr, levene, kruskal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Constants ─────────────────────────────────────────────────────────────────
G          = 6.67430e-11
C_LIGHT    = 2.99792458e8
M_SUN      = 1.98847e30
ALPHA      = 0.007297352569
ALPHA_INV  = 1.0 / ALPHA        # 137.03600
K0         = 1.1467e84
NU_P       = 1.8549e43
SQRT_J     = math.sqrt(0.75)    # sqrt(j(j+1)) at j=1/2
N_BBH      = math.log2(NU_P / math.sqrt(K0 / (SQRT_J * 6.09)))
E_NR_COEFF = 0.0842             # Healy fallback

print("=" * 68)
print("  SERAPHIM KERR REMNANT SPIN TEST")
print("=" * 68)
print()
print("  n_BBH theory = %.6f" % N_BBH)
print("  1/alpha      = %.6f" % ALPHA_INV)
print("  Robertson-Heisenberg separation = exactly 1.000 octave")
print("  Kerr(a_f=0.666)/Schwarz         = exactly 1.000 octave")
print("  Hypothesis: these are the same topological fact")
print()

# ── Kerr ISCO ─────────────────────────────────────────────────────────────────
def kerr_isco_rg(chi):
    a = min(abs(chi), 0.9999)
    Z1 = 1+(1-a**2)**(1/3)*((1+a)**(1/3)+(1-a)**(1/3))
    Z2 = math.sqrt(3*a**2+Z1**2)
    s  = 1 if chi >= 0 else -1
    return 3+Z2-s*math.sqrt((3-Z1)*(3+Z1+2*Z2))

def f_gw_schwarz(M_msun):
    return C_LIGHT**3/(6*math.sqrt(6)*math.pi*G*M_msun*M_SUN)

def f_gw_kerr(M_msun, chi):
    Mk = M_msun*M_SUN
    r  = kerr_isco_rg(chi)
    a  = max(-0.9999, min(0.9999, chi))
    return C_LIGHT**3/(G*Mk)/(r**1.5+a)/math.pi

def n_from_f(f):
    return math.log2(NU_P/f) if f and f>0 else None

# Healy et al. 2014 remnant spin fit (fallback when final_spin missing)
def healy_remnant_spin(eta, chi_eff):
    """
    Simple Healy 2014 fit for final spin.
    eta = symmetric mass ratio = q/(1+q)^2
    chi_eff = effective spin
    Returns a_f (dimensionless remnant spin 0-1)
    """
    # Healy & Lousto 2017 simplified:
    a_f = (abs(chi_eff) * (1-3.4*eta) +
           math.sqrt(12)*eta*(1-0.93*eta))
    return min(max(a_f, 0.0), 0.9999)

# ── HDF5 helpers ─────────────────────────────────────────────────────────────
PREFER_WF = ["IMRPhenomXPHM","IMRPhenomXP","IMRPhenomPv2",
             "SEOBNRv4PHM","SEOBNRv4P","NRSur7dq4",
             "IMRPhenomD","SEOBNRv4"]
NON_BBH   = ["NRTidal","NRTidalv2","NSBH","BNS"]

def is_non_bbh(wf): return any(x.upper() in (wf or "").upper() for x in NON_BBH)

def get_posteriors(hf):
    keys = list(hf.keys())
    if "posterior_samples" in keys:
        ps = hf["posterior_samples"]
        if hasattr(ps,"dtype") and ps.dtype is not None: return ps,"posterior_samples"
    wf_cands = [k for k in keys if any(x in k for x in ["IMR","SEOBNR","NR","EOB"])]
    wf_cands.sort(key=lambda k:(
        next((i for i,p in enumerate(PREFER_WF) if p.upper() in k.upper()),999)
        +(100 if is_non_bbh(k) else 0)))
    for wf in wf_cands:
        grp = hf[wf]
        if hasattr(grp,"keys") and "posterior_samples" in grp:
            ps = grp["posterior_samples"]
            if hasattr(ps,"dtype") and ps.dtype is not None: return ps,wf
        if hasattr(grp,"dtype") and grp.dtype is not None: return grp,wf
        if hasattr(grp,"keys"):
            for sub in grp:
                item = grp[sub]
                if hasattr(item,"dtype") and item.dtype is not None: return item,wf
    return None,None

def col_arr(ps, *names):
    avail = ps.dtype.names or []
    for n in names:
        if n in avail:
            try:
                a = np.array(ps[n][:], dtype=float)
                a = a[np.isfinite(a)]
                if len(a)>0: return a
            except: pass
    return None

def med_col(ps, *names):
    a = col_arr(ps, *names)
    return float(np.median(a)) if a is not None else None

# ── Catalog/event detection ───────────────────────────────────────────────────
def detect_catalog(fp):
    p = fp.replace("\\","/")
    if "6513631"  in p or "gwtc2p1" in p.lower(): return "GWTC-2.1"
    if "8177023"  in p or "gwtc3"   in p.lower(): return "GWTC-3"
    if "16053484" in p or "gwtc4"   in p.lower(): return "GWTC-4"
    b = os.path.basename(fp).upper()
    if "GWTC2" in b or "GWTC2P1" in b: return "GWTC-2.1"
    if "GWTC3" in b: return "GWTC-3"
    if "GWTC4" in b: return "GWTC-4"
    return "UNKNOWN"

def event_name(fp):
    m = re.search(r"(GW\d{6}(?:_\d{6})?)", os.path.basename(fp))
    return m.group(1) if m else os.path.basename(fp)[:24]

# ── Seraphim n ────────────────────────────────────────────────────────────────
def seraphim_n(E_arr, M_arr):
    L = min(len(E_arr),len(M_arr))
    E,M = E_arr[:L],M_arr[:L]
    valid = (E>0)&(M>0)&(E<M)
    if valid.sum()<10: return None,None
    ns = np.log2(NU_P/np.sqrt(K0/(SQRT_J*(E[valid]/(M[valid]*ALPHA)))))
    ok = np.isfinite(ns)&(ns>2)&(ns<10)
    if ok.sum()<10: return None,None
    return float(np.median(ns[ok])),float(np.std(ns[ok]))

# ── Per-file processing ───────────────────────────────────────────────────────
def process(fp):
    try:
        with h5py.File(fp,"r") as hf:
            ps,wf = get_posteriors(hf)
            if ps is None: return None,"no posterior dataset"
            if is_non_bbh(wf): return None,"non-BBH waveform"
            avail = set(ps.dtype.names or [])

            M_tot_arr = col_arr(ps,"total_mass_source","total_mass")
            if M_tot_arr is None: return None,"no total_mass"
            M_tot_med = float(np.median(M_tot_arr))

            chi_arr = col_arr(ps,"chi_eff")
            chi_med = float(np.median(chi_arr)) if chi_arr is not None else 0.0

            # REMNANT SPIN — this is the key new column
            # Try final_spin, final_spin_non_evolved, final_spin_source
            af_arr = col_arr(ps,
                "final_spin","final_spin_non_evolved",
                "final_spin_source","remnant_spin","a_f")
            af_med = float(np.median(af_arr)) if af_arr is not None else None
            af_source = "posterior" if af_arr is not None else "healy_fit"

            # If not in posteriors, use Healy fit
            if af_med is None:
                eta_arr = col_arr(ps,"symmetric_mass_ratio","eta")
                eta_med = float(np.median(eta_arr)) if eta_arr is not None else 0.25
                af_med = healy_remnant_spin(eta_med, chi_med)
                af_source = "healy_fit"

            # Clamp to physical range
            af_med = min(max(af_med, 0.0), 0.9999)

            # Energy loss waterfall
            E_arr,M_aligned,strategy = None,None,""
            for fm in ["final_mass_source_non_evolved","final_mass_source",
                       "final_mass_non_evolved","final_mass"]:
                if fm in avail:
                    fm_arr = col_arr(ps,fm)
                    if fm_arr is not None:
                        L = min(len(M_tot_arr),len(fm_arr))
                        e = M_tot_arr[:L]-fm_arr[:L]
                        ok = (e>0)&(e<M_tot_arr[:L])
                        if ok.sum()>=10:
                            E_arr=e[ok]; M_aligned=M_tot_arr[:L][ok]
                            strategy="M_total-%s"%fm; break

            if E_arr is None:
                for rc in ["radiated_energy_non_evolved","radiated_energy"]:
                    if rc in avail:
                        ra = col_arr(ps,rc)
                        if ra is not None:
                            L=min(len(M_tot_arr),len(ra))
                            ok=(ra[:L]>0)&(ra[:L]<M_tot_arr[:L])
                            if ok.sum()>=10:
                                E_arr=ra[:L][ok]; M_aligned=M_tot_arr[:L][ok]
                                strategy=rc; break

            if E_arr is None:
                mc_arr = col_arr(ps,"chirp_mass_source","chirp_mass")
                if mc_arr is not None:
                    L=min(len(M_tot_arr),len(mc_arr))
                    e=E_NR_COEFF*mc_arr[:L]
                    ok=(e>0)&(e<M_tot_arr[:L])
                    if ok.sum()>=10:
                        E_arr=e[ok]; M_aligned=M_tot_arr[:L][ok]
                        strategy="NR_fit"

            if E_arr is None: return None,"no energy loss"

            n_sph,n_sph_std = seraphim_n(E_arr,M_aligned)
            if n_sph is None: return None,"seraphim_n failed"

            # Three carrier frequencies
            f_s  = f_gw_schwarz(M_tot_med)             # Schwarzschild
            f_kb = f_gw_kerr(M_tot_med, chi_med)       # Kerr / binary chi
            f_kr = f_gw_kerr(M_tot_med, af_med)        # Kerr / remnant spin

            n_cs  = n_from_f(f_s)
            n_ckb = n_from_f(f_kb)
            n_ckr = n_from_f(f_kr)

            return dict(
                wf=wf, strategy=strategy, af_source=af_source,
                M_tot=M_tot_med,
                chi_med=chi_med,
                af_med=af_med,
                redshift=med_col(ps,"redshift"),
                n_seraphim=n_sph, n_sph_std=n_sph_std,
                n_carrier_schwarz=n_cs,
                n_carrier_kerr_binary=n_ckb,
                n_carrier_kerr_remnant=n_ckr,
                kerr_remnant_delta=n_ckr-n_cs if n_ckr and n_cs else None,
            ), None

    except Exception as ex:
        return None,str(ex)

# ── File discovery ────────────────────────────────────────────────────────────
def find_files(root):
    found=[]
    for dp,_,fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".h5") or fn.lower().endswith(".hdf5"):
                found.append(os.path.join(dp,fn))
    return sorted(set(found))

# ── Main ──────────────────────────────────────────────────────────────────────
def run(root):
    print("  Scanning: %s" % os.path.abspath(root))
    files = find_files(root)
    print("  Found %d HDF5 files" % len(files))
    print()

    rows,skipped=[],[]
    for fp in files:
        ename=event_name(fp); cat=detect_catalog(fp)
        res,reason=process(fp)
        if res is None:
            skipped.append((ename,reason)); continue

        n_sph=res["n_seraphim"]
        n_cs=res["n_carrier_schwarz"]
        n_ckb=res["n_carrier_kerr_binary"]
        n_ckr=res["n_carrier_kerr_remnant"]
        if not all([n_sph,n_cs,n_ckb,n_ckr]):
            skipped.append((ename,"null n value")); continue

        gap_s=n_cs-n_sph; gap_kb=n_ckb-n_sph; gap_kr=n_ckr-n_sph
        log2M=math.log2(res["M_tot"])

        rows.append({
            "event":ename,"catalog":cat,
            "waveform":(res["wf"] or "")[:40],
            "af_source":res["af_source"],
            "M_total_Msun":round(res["M_tot"],4),
            "chi_eff_med":round(res["chi_med"],4),
            "a_final_med":round(res["af_med"],4),
            "redshift":round(res["redshift"],4) if res["redshift"] else "",
            "n_seraphim":round(n_sph,6),
            "n_sph_std":round(res["n_sph_std"],6),
            "n_carrier_schwarz":round(n_cs,6),
            "n_carrier_kerr_binary":round(n_ckb,6),
            "n_carrier_kerr_remnant":round(n_ckr,6),
            "kerr_remnant_delta":round(res["kerr_remnant_delta"],6),
            "gap_schwarz":round(gap_s,6),
            "gap_kerr_binary":round(gap_kb,6),
            "gap_kerr_remnant":round(gap_kr,6),
            "delta_schwarz":round(gap_s-ALPHA_INV,6),
            "delta_kerr_binary":round(gap_kb-ALPHA_INV,6),
            "delta_kerr_remnant":round(gap_kr-ALPHA_INV,6),
            "log2_M":round(log2M,6),
            "alpha_inv":round(ALPHA_INV,6),
            "n_BBH_theory":round(N_BBH,6),
        })

    print("  Processed: %d   Skipped: %d" % (len(rows),len(skipped)))
    if skipped:
        from collections import Counter
        for r,c in Counter(r for _,r in skipped).most_common(5):
            print("    [%3d] %s" % (c,r[:70]))
    print()
    if not rows: sys.exit("No events processed.")

    # Write CSV
    out="seraphim_kerr_remnant_results.csv"
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("  Per-event CSV: %s" % out)
    print()

    # ── Analysis ──────────────────────────────────────────────────────────
    def arr(c): return np.array([float(r[c]) for r in rows if r.get(c,"") not in ("",None)])
    def al2(c1,c2):
        p=[(float(r[c1]),float(r[c2])) for r in rows
           if all(r.get(c,"") not in ("",None) for c in [c1,c2])]
        if not p: return np.array([]),np.array([])
        a,b=zip(*p); return np.array(a),np.array(b)
    def partial_r(x,y,z):
        cx=np.polyfit(z,x,1); cy=np.polyfit(z,y,1)
        return pearsonr(x-np.polyval(cx,z), y-np.polyval(cy,z))

    gs  = arr("gap_schwarz")
    gkb = arr("gap_kerr_binary")
    gkr = arr("gap_kerr_remnant")
    af  = arr("a_final_med")
    chi = arr("chi_eff_med")
    krd = arr("kerr_remnant_delta")

    sep="="*68
    print(sep)
    print("  CORE COMPARISON: THREE GAP MEASURES")
    print(sep)
    print()
    print("  %-28s  %10s  %10s  %10s" % ("Metric","Schwarz","Kerr(chi)","Kerr(a_f)"))
    print("  "+"-"*62)

    for label,g in [("Median gap",
                     [np.median(gs),np.median(gkb),np.median(gkr)]),
                    ("Std of gap",
                     [np.std(gs),np.std(gkb),np.std(gkr)]),
                    ("Delta from 1/alpha",
                     [np.median(gs)-ALPHA_INV,
                      np.median(gkb)-ALPHA_INV,
                      np.median(gkr)-ALPHA_INV]),
                    ("Pct from 1/alpha",
                     [abs(np.median(gs)-ALPHA_INV)/ALPHA_INV*100,
                      abs(np.median(gkb)-ALPHA_INV)/ALPHA_INV*100,
                      abs(np.median(gkr)-ALPHA_INV)/ALPHA_INV*100])]:
        print("  %-28s  %10.4f  %10.4f  %10.4f" % (label,g[0],g[1],g[2]))

    print()
    print("  Remnant spin: median=%.4f  mean=%.4f  std=%.4f" % (
          np.median(af),np.mean(af),np.std(af)))
    print("  Kerr remnant delta n: median=%.4f  (theory: ~-1.0 oct)" % np.median(krd))
    print()

    # Std reduction from Schwarzschild to Kerr remnant
    std_red_remnant = (np.std(gs)-np.std(gkr))/np.std(gs)*100
    med_shift = abs(np.median(gkr)-ALPHA_INV) - abs(np.median(gs)-ALPHA_INV)
    print("  Std reduction (Schwarz->Kerr_remnant): %+.4f%%" % std_red_remnant)
    print("  Median shift toward 1/alpha: %+.4f oct" % (-med_shift))
    print()

    # Levene
    if HAS_SCIPY:
        _,p_lev = levene(gs,gkr)
        print("  Levene test (Schwarz vs Kerr_remnant): p=%.4e  %s" % (
              p_lev,"SIGNIFICANT" if p_lev<0.05 else "not significant"))
        print()

    # Spin correlations — the key test
    print(sep)
    print("  SPIN CORRELATION TEST")
    print(sep)
    print()

    if HAS_SCIPY:
        for gname,g in [("gap_schwarz",gs),("gap_kerr_binary",gkb),
                        ("gap_kerr_remnant",gkr)]:
            # raw chi_eff
            g2,c2 = al2(gname,"chi_eff_med")
            if len(g2)>=5:
                rho,p = spearmanr(c2,g2)
                print("  Raw r(chi_eff, %s):  %+.4f  p=%.2e" % (gname,rho,p))

        print()
        # Partial correlations
        lm = arr("log2_M")
        Ls=min(len(gs),len(chi),len(lm))
        for gname,g in [("Schwarz",gs[:Ls]),
                        ("Kerr_binary",gkb[:Ls]),
                        ("Kerr_remnant",gkr[:Ls])]:
            rp,pp = partial_r(chi[:Ls],g,lm[:Ls])
            print("  Partial r(chi | log2M) for %-16s: %+.4f  p=%.2e  %s" % (
                  gname,rp,pp,"SIG" if pp<0.05 else "ns"))

        print()
        # a_f correlations
        Lf=min(len(gs),len(af),len(lm))
        for gname,g in [("Schwarz",gs[:Lf]),
                        ("Kerr_binary",gkb[:Lf]),
                        ("Kerr_remnant",gkr[:Lf])]:
            rp,pp = partial_r(af[:Lf],g,lm[:Lf])
            print("  Partial r(a_f  | log2M) for %-16s: %+.4f  p=%.2e  %s" % (
                  gname,rp,pp,"SIG" if pp<0.05 else "ns"))
        print()

    # How many events have posterior a_f vs Healy fit?
    n_post  = sum(1 for r in rows if r["af_source"]=="posterior")
    n_healy = sum(1 for r in rows if r["af_source"]=="healy_fit")
    print("  Remnant spin source: %d from posteriors, %d from Healy fit" % (
          n_post,n_healy))
    print()

    # Per-catalog
    print(sep)
    print("  PER-CATALOG")
    print(sep)
    print()
    sum_rows=[]
    for cat in ["GWTC-2.1","GWTC-3","GWTC-4"]:
        sub=[r for r in rows if r["catalog"]==cat]
        if not sub: continue
        gs_c=np.array([float(r["gap_schwarz"])       for r in sub])
        gkr_c=np.array([float(r["gap_kerr_remnant"]) for r in sub])
        af_c=np.array([float(r["a_final_med"])        for r in sub])
        zs=[float(r["redshift"]) for r in sub if r.get("redshift","") not in ("",None)]
        std_red_c=(np.std(gs_c)-np.std(gkr_c))/np.std(gs_c)*100
        print("  %s  N=%d  af_med=%.3f" % (cat,len(sub),np.median(af_c)))
        print("    Schwarz:      gap=%.4f ± %.4f  d=%+.4f" % (
              np.median(gs_c),np.std(gs_c),np.median(gs_c)-ALPHA_INV))
        print("    Kerr_remnant: gap=%.4f ± %.4f  d=%+.4f  std_red=%+.2f%%" % (
              np.median(gkr_c),np.std(gkr_c),np.median(gkr_c)-ALPHA_INV,std_red_c))
        print()
        sum_rows.append({
            "catalog":cat,"N":len(sub),
            "median_af":round(float(np.median(af_c)),4),
            "gap_schwarz_med":round(float(np.median(gs_c)),4),
            "gap_schwarz_std":round(float(np.std(gs_c)),4),
            "delta_schwarz":round(float(np.median(gs_c))-ALPHA_INV,4),
            "gap_kerr_remnant_med":round(float(np.median(gkr_c)),4),
            "gap_kerr_remnant_std":round(float(np.std(gkr_c)),4),
            "delta_kerr_remnant":round(float(np.median(gkr_c))-ALPHA_INV,4),
            "std_reduction_pct":round(std_red_c,3),
            "median_z":round(np.median(zs),4) if zs else "",
        })

    if sum_rows:
        out_s="seraphim_kerr_remnant_summary.csv"
        with open(out_s,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(sum_rows[0].keys()))
            w.writeheader(); w.writerows(sum_rows)
        print("  Summary CSV: %s" % out_s)
    print()

    # ── VERDICT ───────────────────────────────────────────────────────────
    print(sep)
    print("  VERDICT")
    print(sep)
    print()
    print("  Schwarzschild:   median=%.4f  std=%.4f  d=%+.4f" % (
          np.median(gs),np.std(gs),np.median(gs)-ALPHA_INV))
    print("  Kerr (binary):   median=%.4f  std=%.4f  d=%+.4f" % (
          np.median(gkb),np.std(gkb),np.median(gkb)-ALPHA_INV))
    print("  Kerr (remnant):  median=%.4f  std=%.4f  d=%+.4f" % (
          np.median(gkr),np.std(gkr),np.median(gkr)-ALPHA_INV))
    print()
    print("  1/alpha = %.4f" % ALPHA_INV)
    print()
    print("  Robertson-Heisenberg: 1.000 octave separation (exact)")
    print("  Kerr(a_f)/Schwarz:    %.4f octave shift (median)" % abs(np.median(krd)))
    print()

    if abs(np.median(krd) - (-1.0)) < 0.1:
        print("  ** Kerr remnant shift ~ 1 octave **")
        print("  ** Matches Robertson-Heisenberg separation **")
        print("  Both trace to chi(S^2) = 2: the Euler characteristic")
        print("  of the event horizon topology.")
    else:
        print("  Remnant Kerr shift = %.4f oct (not exactly 1.0)" % abs(np.median(krd)))

    if std_red_remnant > 1.0:
        print()
        print("  ** GAP TIGHTENS with remnant Kerr correction **")
        print("  Remnant spin is accounting for real scatter.")
    if med_shift < -0.1:
        print()
        print("  ** MEDIAN MOVES TOWARD 1/alpha **")
    print()
    print("  Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir","-d",default=".",
        help="Root dir to scan (default: current)")
    run(ap.parse_args().dir)
