"""
seraphim_alpha_test.py  (fixed)
================================
THREE-PART TEST against all three GWTC catalogs:

  Test 1: Does the GW ISCO carrier frequency sit at octave depth n ~ 1/alpha?
          1/alpha = 137.036  (fine structure constant inverse)

  Test 2: Does the gap (n_carrier - n_BBH) correlate with total mass?
          Heavier systems -> lower f_ISCO -> higher n_carrier -> larger gap.

  Test 3: Low-mass BBH convergence (M < 40 Msun closest to 1/alpha).

OUTPUTS:
  seraphim_alpha_results.csv   -- one row per event
  seraphim_alpha_summary.csv   -- statistics by catalog / event type

USAGE:
  python seraphim_alpha_test.py --dir /path/to/hdf5/files
"""

import os, sys, glob, math, argparse, csv, re
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("ERROR: h5py not installed.  pip install h5py")

try:
    from scipy.stats import spearmanr, pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found -- correlation tests skipped.")

# Constants
G         = 6.67430e-11
C_SPEED   = 2.99792458e8
M_SUN     = 1.98847e30
H_PLANCK  = 6.62607015e-34
HBAR      = 1.054571817e-34
lP        = 1.616255e-35
tP        = 5.391247e-44
ALPHA     = 7.2973525693e-3
ALPHA_INV = 1.0 / ALPHA             # 137.035999

GAMMA_AREA = 0.2375
K0         = (HBAR**2 * C_SPEED**2) / (32 * GAMMA_AREA * lP**2 * H_PLANCK**2)
NU_PLANCK  = 1.0 / tP
J_SPIN     = 0.5
SQRT_J     = math.sqrt(J_SPIN * (J_SPIN + 1.0))
N_SUN      = 6.09
N_BBH      = math.log2(NU_PLANCK / math.sqrt(K0 / (SQRT_J * N_SUN)))

print("=" * 60)
print("  Seraphim Alpha Gap Test")
print("=" * 60)
print(f"  K0         = {K0:.6e} Hz^2")
print(f"  n_BBH      = {N_BBH:.6f}")
print(f"  1/alpha    = {ALPHA_INV:.6f}")
print()

def f_isco(total_mass_msun):
    M = total_mass_msun * M_SUN
    return (C_SPEED**3) / (6.0 * math.sqrt(6.0) * math.pi * G * M)

def n_oct(freq_hz):
    return math.log2(NU_PLANCK / freq_hz) if freq_hz > 0 else None

BBH_PREFERRED = [
    'C01:IMRPhenomXPHM', 'C01:IMRPhenomPv2', 'C01:SEOBNRv4PHM',
    'C00:IMRPhenomXPHM', 'IMRPhenomXPHM', 'IMRPhenomPv2', 'SEOBNRv4PHM',
]

def get_posteriors(f):
    top_keys = list(f.keys())
    # Direct root posterior_samples (old format)
    if 'posterior_samples' in top_keys:
        return f['posterior_samples'], 'posterior_samples'
    # Try preferred BBH keys
    for pref in BBH_PREFERRED:
        for k in top_keys:
            if pref in k:
                try:
                    return f[k]['posterior_samples'], k
                except (KeyError, AttributeError, TypeError):
                    continue
    # Any key with posterior_samples child
    for k in top_keys:
        try:
            return f[k]['posterior_samples'], k
        except (KeyError, AttributeError, TypeError):
            continue
    return None, None

def classify_key(key):
    ku = key.upper()
    if 'NSBH' in ku:
        return 'NSBH'
    if 'NRTIDAL' in ku and 'NSBH' not in ku and 'XPHM' not in ku:
        return 'BNS_candidate'
    return 'BBH'

def safe_col(ps, *names):
    try:
        col_names = (ps.dtype.names if hasattr(ps, 'dtype') and ps.dtype.names
                     else list(ps.keys()))
    except Exception:
        return None
    for name in names:
        if name in col_names:
            try:
                arr = np.array(ps[name][:], dtype=float)
                arr = arr[np.isfinite(arr)]
                return arr if len(arr) > 0 else None
            except Exception:
                continue
    return None

def med(arr):
    return float(np.median(arr)) if arr is not None and len(arr) > 0 else None

def catalog_from_path(path):
    b = os.path.basename(path).upper()
    if 'GWTC2P1' in b or 'GWTC2.1' in b: return 'GWTC-2.1'
    if 'GWTC3' in b or 'GWTC-3' in b:    return 'GWTC-3'
    if 'GWTC4' in b or 'GWTC-4' in b:    return 'GWTC-4'
    return 'UNKNOWN'

def event_from_path(path):
    m = re.search(r'(GW\d{6}(?:_\d{6})?)', os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)[:20]

def run(search_dir):
    files = []
    for pat in ['*.h5', '*.hdf5', '*.hdf']:
        files += glob.glob(os.path.join(search_dir, pat))
        files += glob.glob(os.path.join(search_dir, '**', pat), recursive=True)
    files = sorted(set(files))

    if not files:
        sys.exit(f"ERROR: No HDF5 files found under {search_dir}")

    print(f"Found {len(files)} HDF5 files")
    print()

    rows, skipped = [], 0

    for fpath in files:
        ename   = event_from_path(fpath)
        catalog = catalog_from_path(fpath)

        try:
            with h5py.File(fpath, 'r') as f:
                ps, wfkey = get_posteriors(f)
                if ps is None:
                    print(f"  SKIP {ename}: no posterior_samples found")
                    skipped += 1
                    continue

                etype = classify_key(wfkey)

                m1_src = safe_col(ps, 'mass_1_source', 'mass_1')
                m2_src = safe_col(ps, 'mass_2_source', 'mass_2')
                mt_src = safe_col(ps, 'total_mass_source')

                if mt_src is None and m1_src is not None and m2_src is not None:
                    n = min(len(m1_src), len(m2_src))
                    mt_src = m1_src[:n] + m2_src[:n]

                mc_src = safe_col(ps, 'chirp_mass_source', 'chirp_mass')
                chi    = safe_col(ps, 'chi_eff')
                z      = safe_col(ps, 'redshift')
                mf_src = safe_col(ps,
                    'final_mass_source_non_evolved', 'final_mass_source',
                    'final_mass_non_evolved', 'final_mass')
                e_rad  = safe_col(ps, 'radiated_energy_non_evolved', 'radiated_energy')

                if mt_src is None or len(mt_src) < 50:
                    n_got = len(mt_src) if mt_src is not None else 0
                    print(f"  SKIP {ename}: insufficient mass samples ({n_got})")
                    skipped += 1
                    continue

                M_total = med(mt_src)
                if M_total is None or M_total <= 0:
                    skipped += 1
                    continue

                M_chirp = med(mc_src)
                M_final = med(mf_src)
                E_rad_v = med(e_rad)
                chi_med = med(chi)
                z_med   = med(z)
                n_samp  = len(mt_src)

                f_carrier = f_isco(M_total)
                n_carrier = n_oct(f_carrier)
                gap       = n_carrier - N_BBH if n_carrier else None
                d_alpha   = gap - ALPHA_INV   if gap else None
                mass_corr = math.log2(M_total / 30.0)
                gap_corr  = gap - mass_corr   if gap else None

                n_erad = None
                if E_rad_v and E_rad_v > 0:
                    n_erad = n_oct((E_rad_v * M_SUN * C_SPEED**2) / H_PLANCK)

                rows.append({
                    'event':                ename,
                    'catalog':              catalog,
                    'event_type':           etype,
                    'waveform_key':         wfkey,
                    'n_samples':            n_samp,
                    'total_mass_source':    round(M_total, 4),
                    'chirp_mass_source':    round(M_chirp, 4) if M_chirp else '',
                    'final_mass_source':    round(M_final, 4) if M_final else '',
                    'radiated_energy_Msun': round(E_rad_v, 6) if E_rad_v else '',
                    'chi_eff':              round(chi_med, 4) if chi_med else '',
                    'redshift':             round(z_med,   4) if z_med   else '',
                    'f_isco_hz':            round(f_carrier, 4),
                    'n_carrier':            round(n_carrier, 6) if n_carrier else '',
                    'n_BBH':                round(N_BBH, 6),
                    'gap':                  round(gap, 6) if gap else '',
                    'alpha_inv':            round(ALPHA_INV, 6),
                    'delta_alpha':          round(d_alpha, 6)  if d_alpha else '',
                    'gap_mass_corrected':   round(gap_corr, 6) if gap_corr else '',
                    'n_radiated_energy':    round(n_erad, 6)   if n_erad  else '',
                })

        except Exception as e:
            print(f"  ERROR {ename}: {e}")
            skipped += 1

    print(f"Processed: {len(rows)}  |  Skipped: {skipped}")
    print()

    if not rows:
        sys.exit("No events processed -- check your --dir path.")

    out_events = 'seraphim_alpha_results.csv'
    with open(out_events, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Per-event CSV -> {out_events}")

    print()
    print("=" * 65)
    print("  TEST 1: n_carrier distribution (BBH events)")
    print("=" * 65)

    bbh = [r for r in rows if r['event_type'] == 'BBH'
           and r['n_carrier'] != '' and r['gap'] != '']

    if not bbh:
        print("  No BBH events with valid n_carrier.")
    else:
        gaps   = np.array([float(r['gap'])       for r in bbh])
        ncs    = np.array([float(r['n_carrier'])  for r in bbh])
        deltas = np.array([float(r['delta_alpha']) for r in bbh if r['delta_alpha'] != ''])
        masses = np.array([float(r['total_mass_source']) for r in bbh])

        print(f"  BBH events: {len(bbh)}")
        print()
        print(f"  n_carrier (octave depth of ISCO carrier frequency):")
        print(f"    Median:  {np.median(ncs):.4f}")
        print(f"    Mean:    {np.mean(ncs):.4f}")
        print(f"    Std:     {np.std(ncs):.4f}")
        print(f"    Range:   [{ncs.min():.2f}, {ncs.max():.2f}]")
        print()
        print(f"  Gap = n_carrier - n_BBH:")
        print(f"    Median:  {np.median(gaps):.4f}")
        print(f"    Mean:    {np.mean(gaps):.4f}")
        print(f"    Std:     {np.std(gaps):.4f}")
        print()
        print(f"  1/alpha  = {ALPHA_INV:.4f}")
        d_med = float(np.median(deltas))
        d_pct = d_med / ALPHA_INV * 100
        print(f"  delta    = gap - 1/alpha:")
        print(f"    Median: {d_med:+.4f}  ({d_pct:+.3f}% of 1/alpha)")
        print()

        apct = abs(d_pct)
        print("  VERDICT:")
        if apct < 1.0:
            print(f"  *** STRONG MATCH -- gap within {apct:.2f}% of 1/alpha ***")
        elif apct < 5.0:
            print(f"  ** CANDIDATE -- gap within {apct:.2f}% of 1/alpha **")
        elif apct < 10.0:
            print(f"  * WEAK -- gap within {apct:.2f}% of 1/alpha")
        else:
            print(f"  NO MATCH -- gap is {apct:.1f}% from 1/alpha")
        print()

        f_exact = NU_PLANCK / (2 ** (N_BBH + ALPHA_INV))
        M_exact = (C_SPEED**3 / (6*math.sqrt(6)*math.pi * G * f_exact)) / M_SUN
        print(f"  Mass for exact gap = 1/alpha: {M_exact:.2f} Msun  (f_ISCO = {f_exact:.3f} Hz)")
        print()

        print("-" * 65)
        print("  TEST 2: Mass-gap correlation")
        print("-" * 65)
        if HAS_SCIPY and len(masses) > 10:
            rho, p_s = spearmanr(masses, gaps)
            r_p, p_p = pearsonr(np.log10(masses), gaps)
            print(f"  Spearman r(M_total, gap):   {rho:+.4f}  p={p_s:.2e}")
            print(f"  Pearson  r(log10 M, gap):   {r_p:+.4f}  p={p_p:.2e}")
            print(f"  Expect: positive r, p << 0.05")
        else:
            print("  (scipy required for this test)")
        print()

        print("-" * 65)
        print("  TEST 3: Low-mass convergence  (M_total < 40 Msun)")
        print("-" * 65)
        low = [r for r in bbh if float(r['total_mass_source']) < 40]
        if low:
            lg = np.array([float(r['gap']) for r in low])
            print(f"  N events:   {len(low)}")
            print(f"  Median gap: {np.median(lg):.4f}")
            print(f"  vs 1/alpha: {np.median(lg) - ALPHA_INV:+.4f}")
        else:
            print("  No BBH events below 40 Msun.")
        print()

        print("-" * 65)
        print("  Per-catalog breakdown (BBH)")
        print("-" * 65)
        for cat in ['GWTC-2.1', 'GWTC-3', 'GWTC-4', 'UNKNOWN']:
            cr = [r for r in bbh if r['catalog'] == cat]
            if not cr: continue
            cg = np.array([float(r['gap'])      for r in cr])
            cn = np.array([float(r['n_carrier']) for r in cr])
            print(f"  {cat}  N={len(cr):3d}  "
                  f"median_gap={np.median(cg):.4f}  "
                  f"median_n_carrier={np.median(cn):.4f}  "
                  f"delta={np.median(cg)-ALPHA_INV:+.4f}")
        print()

    for et in ['NSBH', 'BNS_candidate']:
        other = [r for r in rows if r['event_type'] == et and r['n_carrier'] != '']
        if other:
            og = np.array([float(r['gap']) for r in other])
            print(f"  {et} (N={len(other)}):  "
                  f"median_gap={np.median(og):.4f}  "
                  f"delta_1/alpha={np.median(og)-ALPHA_INV:+.4f}")

    sum_rows = []
    for cat in ['GWTC-2.1', 'GWTC-3', 'GWTC-4', 'ALL']:
        for et in ['BBH', 'NSBH', 'BNS_candidate', 'ALL']:
            sub = [r for r in rows
                   if (cat == 'ALL' or r['catalog'] == cat)
                   and (et  == 'ALL' or r['event_type'] == et)
                   and r['n_carrier'] != '' and r['gap'] != '']
            if not sub: continue
            sg = np.array([float(r['gap'])      for r in sub])
            sn = np.array([float(r['n_carrier']) for r in sub])
            sum_rows.append({
                'catalog':            cat,
                'event_type':         et,
                'N':                  len(sub),
                'median_n_carrier':   round(float(np.median(sn)), 4),
                'mean_n_carrier':     round(float(np.mean(sn)),   4),
                'std_n_carrier':      round(float(np.std(sn)),    4),
                'median_gap':         round(float(np.median(sg)), 4),
                'mean_gap':           round(float(np.mean(sg)),   4),
                'std_gap':            round(float(np.std(sg)),    4),
                'alpha_inv':          round(ALPHA_INV, 6),
                'delta_alpha_median': round(float(np.median(sg)) - ALPHA_INV, 4),
                'pct_from_alpha':     round(abs(float(np.median(sg))-ALPHA_INV)/ALPHA_INV*100, 3),
            })

    out_sum = 'seraphim_alpha_summary.csv'
    if sum_rows:
        with open(out_sum, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
            w.writeheader()
            w.writerows(sum_rows)
        print(f"\nSummary CSV -> {out_sum}")

    print()
    print("Done.")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Seraphim alpha gap test')
    p.add_argument('--dir', '-d', default='.',
                   help='Directory containing GWTC HDF5 files (recursive). Default: .')
    args = p.parse_args()
    run(args.dir)
