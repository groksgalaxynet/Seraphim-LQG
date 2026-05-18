"""
GW191219_163120 Posterior Diagnostic
Seraphim LQG Framework — T1 Violation Investigation

Run this against your local GWTC-3 HDF5 file:
  Zenodo 8177023 — GWTC-3 posterior samples

Usage:
  python3 gw191219_diagnose.py /path/to/GWTC3/

The script will search for GW191219_163120 in the directory.
"""

import h5py
import numpy as np
import os
import sys

# ── Framework constants ────────────────────────────────────
alpha      = 1/137.036
gamma_area = 0.2375
K0         = 1.1467e84
nu_P       = 1.855e43
h_planck   = 6.62607015e-34
c          = 2.99792458e8
G          = 6.674e-11
Msun_kg    = 1.989e30
sf         = np.sqrt(3/4)
n_flip     = 3.561
n_BBH      = 5.314
slope      = 3.506

def compute_n(m1_src, m2_src, chi_eff, e_rad_frac=None):
    """
    Compute Seraphim n from posterior samples.
    Uses E_rad from NR fit if not provided.
    """
    M_total = m1_src + m2_src
    q = m2_src / m1_src  # q <= 1
    eta = q / (1 + q)**2

    # E_rad from Healy-Lousto NR fit (same as gap script)
    # E_rad/M = eta*(0.0559745 + 0.580951*eta + ...)
    # Use simple Tichy-Marronetti if no e_rad given
    if e_rad_frac is None:
        p1, p2, p3 = 0.04827, 0.01707, -0.00015
        e_rad_frac = eta * (p1 + p2*eta + p3*chi_eff)
        # fallback: use 4.44% mean if fit gives nonsense
        e_rad_frac = np.clip(e_rad_frac, 0.001, 0.15)

    N_sun = e_rad_frac / alpha
    nu_sq = K0 / (sf * N_sun)
    nu    = np.sqrt(nu_sq)
    n     = np.log2(nu_P / nu)
    return n, N_sun, eta, e_rad_frac

def walk_hdf5(f, prefix=''):
    """Recursively list datasets."""
    for key in f.keys():
        item = f[key]
        path = f'{prefix}/{key}'
        if isinstance(item, h5py.Dataset):
            print(f"  DATASET {path}: shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"  GROUP   {path}/")
            if len(list(item.keys())) < 20:
                walk_hdf5(item, path)

# ── Find the file ──────────────────────────────────────────
search_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
target_file = None

for root, dirs, files in os.walk(search_dir):
    for f in files:
        if '191219' in f and f.endswith('.h5'):
            target_file = os.path.join(root, f)
            print(f"Found: {target_file}")

if target_file is None:
    print(f"ERROR: No GW191219 file found in {search_dir}")
    print("Files found:")
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f.endswith('.h5'):
                print(f"  {os.path.join(root,f)}")
    sys.exit(1)

# ── Open and inspect ──────────────────────────────────────
print(f"\nOpening {target_file}")
with h5py.File(target_file, 'r') as hf:
    print("\n=== FILE STRUCTURE ===")
    walk_hdf5(hf)

# ── Extract posteriors ────────────────────────────────────
print("\n=== EXTRACTING POSTERIORS ===")
with h5py.File(target_file, 'r') as hf:

    # Try GWTC-3 structure: event/waveform/posterior_samples
    # Common keys: mass_1_source, mass_2_source, chi_eff, etc.
    
    def try_get(group, keys):
        for k in keys:
            if k in group:
                return np.array(group[k])
        return None

    # Navigate to posterior samples
    # GWTC-3 structure varies — try common paths
    samples_group = None
    
    # Try direct
    for wf_key in ['C01:IMRPhenomXPHM:HighSpin', 'C01:IMRPhenomXPHM', 
                    'IMRPhenomXPHM', 'mixed']:
        if wf_key in hf:
            samples_group = hf[wf_key]
            print(f"Using waveform key: {wf_key}")
            break
    
    # Try nested
    if samples_group is None:
        for top in hf.keys():
            for sub in hf[top].keys() if isinstance(hf[top], h5py.Group) else []:
                if 'posterior' in sub.lower() or 'samples' in sub.lower():
                    samples_group = hf[top][sub]
                    print(f"Using path: {top}/{sub}")
                    break

    if samples_group is None:
        print("Could not find posterior samples group.")
        print("Top-level keys:", list(hf.keys()))
        sys.exit(1)

    # Get posterior samples group
    ps = samples_group
    if 'posterior_samples' in ps:
        ps = ps['posterior_samples']
    elif 'posterior' in ps:
        ps = ps['posterior']

    print(f"Posterior samples keys: {list(ps.keys())[:20]}")

    # Extract key parameters
    m1 = try_get(ps, ['mass_1_source', 'mass_1'])
    m2 = try_get(ps, ['mass_2_source', 'mass_2'])
    chi = try_get(ps, ['chi_eff', 'chi_effective'])
    q_samp = try_get(ps, ['mass_ratio'])
    chirp = try_get(ps, ['chirp_mass_source', 'chirp_mass'])
    eta_samp = try_get(ps, ['symmetric_mass_ratio'])

    if m1 is None or m2 is None:
        print("ERROR: Cannot find mass parameters")
        print("Available keys:", list(ps.keys()))
        sys.exit(1)

    print(f"\nN samples: {len(m1)}")
    print(f"m1_source: median={np.median(m1):.2f}, mean={np.mean(m1):.2f}, std={np.std(m1):.2f} Msun")
    print(f"m2_source: median={np.median(m2):.2f}, mean={np.mean(m2):.2f}, std={np.std(m2):.2f} Msun")
    
    if chi is not None:
        print(f"chi_eff:   median={np.median(chi):.4f}, mean={np.mean(chi):.4f}, std={np.std(chi):.4f}")
    
    q_arr = m2/m1
    eta_arr = q_arr/(1+q_arr)**2
    print(f"q (m2/m1): median={np.median(q_arr):.4f}, mean={np.mean(q_arr):.4f}")
    print(f"eta:       median={np.median(eta_arr):.4f}, mean={np.mean(eta_arr):.4f}")

    # ── Compute n for each sample ─────────────────────────
    print("\n=== SERAPHIM n COMPUTATION ===")
    chi_use = chi if chi is not None else np.zeros(len(m1))
    
    n_samples, Nsun_samples, efrac_samples = [], [], []
    for i in range(len(m1)):
        n_i, Nsun_i, eta_i, efrac_i = compute_n(m1[i], m2[i], chi_use[i])
        n_samples.append(n_i)
        Nsun_samples.append(Nsun_i)
        efrac_samples.append(efrac_i)

    n_arr_s = np.array(n_samples)
    Nsun_arr = np.array(Nsun_samples)
    efrac_arr = np.array(efrac_samples)

    print(f"n:         median={np.median(n_arr_s):.4f}, mean={np.mean(n_arr_s):.4f}, std={np.std(n_arr_s):.4f}")
    print(f"N_sun:     median={np.median(Nsun_arr):.4f}, mean={np.mean(Nsun_arr):.4f}")
    print(f"E_rad/M:   median={np.median(efrac_arr)*100:.3f}%")
    print()
    print(f"n_flip = {n_flip}")
    print(f"Fraction of samples below n_flip: {(n_arr_s < n_flip).mean()*100:.2f}%")
    print(f"Min n in posterior: {n_arr_s.min():.4f}")
    print(f"Max n in posterior: {n_arr_s.max():.4f}")
    print()

    # ── Why is n so low? Diagnose ─────────────────────────
    print("=== DIAGNOSIS: WHY IS n BELOW n_flip? ===")
    print(f"n_flip = 3.561 corresponds to N_sun = 1 (single LQG face)")
    print(f"n < n_flip means N_sun < 1 -- geometrically forbidden")
    print()
    
    # What N_sun values are below-floor samples producing?
    below_mask = n_arr_s < n_flip
    if below_mask.sum() > 0:
        print(f"Below-floor samples: {below_mask.sum()} of {len(n_arr_s)}")
        print(f"Their N_sun values:  median={np.median(Nsun_arr[below_mask]):.4f}")
        print(f"Their E_rad/M:       median={np.median(efrac_arr[below_mask])*100:.3f}%")
        print(f"Their m1:            median={np.median(m1[below_mask]):.2f} Msun")
        print(f"Their m2:            median={np.median(m2[below_mask]):.2f} Msun")
        print(f"Their chi_eff:       median={np.median(chi_use[below_mask]):.4f}")
        print(f"Their eta:           median={np.median(eta_arr[below_mask]):.4f}")
        print()
        print("Is this event actually a BBH? Check secondary mass:")
        print(f"  m2 range: {m2.min():.2f} to {m2.max():.2f} Msun")
        print(f"  m2 < 3 Msun fraction: {(m2 < 3).mean()*100:.1f}%")
        print(f"  m2 < 5 Msun fraction: {(m2 < 5).mean()*100:.1f}%")
    
    # ── Check if this is really a NSBH masquerading as BBH ─
    print()
    print("=== EVENT CLASSIFICATION CHECK ===")
    print(f"Primary m1:   {np.median(m1):.2f} Msun")
    print(f"Secondary m2: {np.median(m2):.2f} Msun")
    if np.median(m2) < 3.0:
        print("*** SECONDARY MASS < 3 Msun — POSSIBLE NS COMPONENT ***")
        print("    If m2 is a neutron star, C_component = C_NS != 0.5")
        print("    The 4eta*0.5 formula assumes both components are BH")
        print("    C_eff should be 4*eta*C_NS for NSBH, not 4*eta*0.5")
        C_NS_implied = (np.median(n_arr_s) - n_flip) / (slope * 4 * np.median(eta_arr))
        print(f"    Implied C_component from observed n: {C_NS_implied:.4f}")
        print(f"    If NS: this implies R_NS ~ {0.18/C_NS_implied * 10.4:.1f} km (scaled from APR4)")
    elif np.median(m2) < 5.0:
        print("*** SECONDARY IN MASS GAP (3-5 Msun) — AMBIGUOUS ***")
    else:
        print("    Both components appear to be BH mass range")

print("\nScript complete.")
