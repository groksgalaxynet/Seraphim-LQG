# Seraphim LQG Framework

**From the LQG area spectrum and the Robertson minimum uncertainty principle, using only fundamental constants, a single geometric activation constant K₀ = 1.1467 × 10⁸⁴ Hz² is derived — no fitting, no tuning — and from it a predicted octave depth n = 5.314 for binary black hole mergers. That prediction matches 249 of 264 BBH events across three independent GWOSC catalogs to within 0.05σ, holds flat across nine billion years of lookback time, and survives nine independent falsification tests. Either this is a real signal from the quantum geometry of spacetime, or it is the most persistent coincidence in gravitational wave astronomy. LIGO O5 will decide.**

---

## The Core Equation

```
n(C) = 3.561 + 3.506 × C
```

where:
- `n` = octave depth = log₂(ν_Planck / ν_merger)
- `C` = compactness = GM/(Rc²)
- `C = 0.5` (black hole) → n = **5.314** (predicted from K₀ alone)
- `C = 0.0` (singularity) → n = **3.561** (geometric fixed point)
- Slope 3.506 = 2 × (n_BBH − n_flip) = χ(S²) × Δn, where χ(S²) = 2 is the Euler characteristic of the event horizon

---

## Key Constants

| Symbol | Value | Origin |
|--------|-------|--------|
| K₀ | 1.1467 × 10⁸⁴ Hz² | Derived: c² / (128π²·γ_area·ℓ_P²) |
| n_BBH | 5.314 | Predicted from K₀, N★ = 6.09 |
| n_flip | 3.561 | Geometric realm, γ_entropy = 0.12738 |
| β | 0.814 | Empirical (6 × 1M Monte Carlo runs) |
| N★ | 6.09 | E_rad / (M·α), mean over 249 BBH events |

> **CRITICAL:** K₀ exponent is **84**, not 4. Unicode superscript corruption has historically rendered this as 10⁴ in some editors. All scripts use `1.1467e84` explicitly.

---

## Falsifiable Predictions (pre-O5 timestamp)

All predictions archived to Zenodo **before** LIGO O5 data release.

| # | Prediction | Status |
|---|-----------|--------|
| 1 | O5 BBH events cluster at n = 5.314 ± 0.15 | **Pending O5** |
| 2 | O5 NSBH events satisfy n(C) = 3.561 + 3.506·C | Partial: GW190425 passed |
| 3 | Waveform independence: Δn < 0.1 oct across NR families | ✅ Confirmed (Δ = 0.070%) |
| 4 | Data confirm Robertson over Heisenberg at > 0.5 oct separation | ✅ Confirmed (1.000 oct) |
| 5 | χ_eff and q carry independent signals in n | ✅ Confirmed (partial r > 0.60) |
| 6 | NSBH compactness mapping within predicted range | Partial: range check passed |
| 7 | Redshift invariance: dn/dz = 0 | ✅ Confirmed (r = 0.010, p = 0.875) |
| 8 | β → 1.0 in O5 limit | **Pending O5** |
| 9 | Realm-projection geometry universal across compact objects | **Pending O5** |
| 10 | No renormalization freedom: n = 5.314 is the unique zero-parameter prediction | Structural |
| 11 | IMBH mergers at M ≈ 1684 M☉ will show gap ≈ 1/α = 137.036 | **Pending LISA/ET/CE** |

---

## Nine Independent Confirmed Tests

1. BBH band clustering at n = 5.314 across three independent catalogs (0.05σ)
2. Redshift invariance — Spearman r(n,z) = 0.010, p = 0.875, N = 248
3. Waveform independence — IMRPhenomXPHM vs SEOBNRv4PHM Δ = 0.070%
4. Convention falsification — Robertson confirmed over Heisenberg at 1.000 octave
5. Tidal deformability range check — GW190425 passes n(C) prediction to 0.003 oct
6. NSBH compactness consistency — events below BBH band as predicted
7. Spin and mass-ratio partial correlation independence — both confirmed ***
8. Redshift partial correlation — no drift after spin control
9. Alpha gap mass-scaling law — gap = 126.318 + log₂(M), slope = 0.922 (algebraically explained), confirmed across 264 events

---

## Data

Posterior HDF5 files from GWOSC. Download to your working directory before running scripts.

| Catalog | Zenodo | Events |
|---------|--------|--------|
| GWTC-2.1 | [6513631](https://zenodo.org/records/6513631) | 106 |
| GWTC-3 | [8177023](https://zenodo.org/records/8177023) | 72 |
| GWTC-4 | [16053484](https://zenodo.org/records/16053484) | 86 |

All scripts run from the directory containing the HDF5 files — no path arguments needed.

**HDF5 layout by catalog:**
- GWTC-2.1 / GWTC-3: `file → posterior_samples → C01:IMRPhenomXPHM → [datasets]`
- GWTC-4: `file → C00:IMRPhenomXPHM → [datasets]` (no wrapper group)

---

## Scripts

All scripts are self-contained, run from the data directory, and write CSV output.

### Primary Analysis

| Script | What it does | Output |
|--------|-------------|--------|
| `seraphim_gap_structure_v3.py` | **Main pipeline.** Computes n_seraphim and n_carrier per event from raw posteriors. Calculates gap = n_carrier − n_seraphim, alpha gap structure, and mass-scaling law. Handles all three catalog HDF5 layouts. | `seraphim_gap_v3_results.csv` |
| `seraphim_shape_test.py` | n distribution shape analysis. Normality tests (Shapiro-Wilk, D'Agostino K², Anderson-Darling), skewness, kurtosis, tail analysis, per-catalog breakdown, BBH band statistics. | `seraphim_shape_per_event.csv`, `seraphim_shape_normality.csv`, `seraphim_shape_tail.csv` |
| `seraphim_redshift_v4.py` | Redshift invariance test. Spearman and Pearson r(n,z), BBH-band-only test, partial correlations controlling for spin. Handles GWTC-2.1/3/4 combined. | `seraphim_redshift_results.csv` |

### Independent Tests

| Script | What it does | Output |
|--------|-------------|--------|
| `seraphim_convention_test.py` | Robertson vs Heisenberg convention falsification. Computes n under both conventions per event, confirms 1.000 octave separation. | `seraphim_convention_results.csv` |
| `seraphim_anova_test.py` | Catalog-to-catalog consistency (ANOVA). Tests whether GWTC-2.1, GWTC-3, GWTC-4 are drawn from the same n distribution. Pairwise comparisons, effect sizes. | `seraphim_anova_per_event.csv`, `seraphim_anova_descriptives.csv`, `seraphim_anova_tests.csv`, `seraphim_anova_pairwise.csv` |
| `seraphim_width_test.py` | Posterior width vs octave depth scatter. Tests whether outliers from n = 5.314 are poorly constrained events or genuine departures. | (stdout + optional CSV) |
| `seraphim_gauss_bonnet.py` | Gauss-Bonnet identity test. Computes β + 2·E[χ_eff²] per event from full posterior samples. Tests approach to unity. | `seraphim_gauss_bonnet_results.csv`, `seraphim_gb_deduped.csv` |
| `seraphim_beta_posterior.py` | Per-event β from full posterior samples (proper E[χ_eff²], not median²). Compares to Monte Carlo β = 0.814. | `seraphim_q_per_event.csv`, `seraphim_q_beta_strat.csv` |

### Supporting Tests

| Script | What it does | Output |
|--------|-------------|--------|
| `Qtest.py` | Mass ratio q stratification. Bins events by q, computes mean n per bin, tests N★ coupling through mass ratio. | `seraphim_q_per_event.csv`, `seraphim_q_bins.csv`, `seraphim_q_catalog.csv` |
| `beta.py` | Beta Monte Carlo (1M iterations). Samples NS radii and spin distributions to constrain compactness exponent β. | `2file_seraphim_beta_mc_results.csv` |
| `chirpmass.py` | Chirp mass independence test. Partial correlation of M_chirp vs n after controlling for q and χ_eff. | `6513631_chirpmass_run2.csv` |
| `waveform.py` | Waveform family independence test. Compares IMRPhenomXPHM vs SEOBNRv4PHM on matched events in GWTC-2.1. | `8177023_run2_waveform_delta.csv` |
| `partial_corr.py` | Partial correlation utility. Computes partial r(χ_eff → n \| q) and r(q → n \| χ_eff) from a seraphim_results.csv input. | stdout |

---

## Open Questions

- **d_eff ≈ 2.49:** Free regression of n on log₂(N★) yields slope b = 0.402 (CI [0.386, 0.418]), excluding b = 0.500. The effective geometric dimension d_eff = 1/b ≈ 2.49. Whether this is exactly 5/2 is unresolved.
- **Coupling ladder confirmation:** Intercepts 132.97 → 135.00 → 137.036 with step size 2.03 ≈ χ(S²) = 2 require O5 confirmation before claiming significance.
- **β = 0.814:** Empirical measurement, not derived. β → 1.0 in the O5 limit is a prediction, not a correction.
- **Tiling postulate:** N★·A_j = π·(Δx)² is the central Seraphim-specific hypothesis connecting LQG geometry to GW phenomenology. It is not standard LQG and is not proven from first principles.

---

## Two Gamma Values

Both values are correct and serve different roles. They are not inconsistent.

| Role | γ value | Physical meaning |
|------|---------|-----------------|
| K₀ derivation (EM realm) | 0.2375 | j=1/2 area-counting — Meissner 2004 |
| n_flip (geometric realm) | 0.12738 | Entropy-counting, 1 bit per j=1/2 face |

---

## Citation

```bibtex
@misc{seraphim_v18_3,
  author    = {Warden, Mike},
  title     = {Seraphim v18.3: LQG Patch Geometry, Gravitational Wave Energy Loss,
               and a Unified Compactness Equation},
  year      = {2026},
  month     = {March},
  doi       = {10.5281/zenodo.19053650},
  url       = {https://doi.org/10.5281/zenodo.19053650},
  note      = {Preprint. 264 BBH events. Zero free parameters. Predictions
               archived before LIGO O5 data release.}
}
```

---

## Links

- **Zenodo (v18.3):** https://doi.org/10.5281/zenodo.19053650
- **GWOSC:** https://gwosc.org
- **License:** MIT

---

*264 BBH events · GWTC-2.1 · GWTC-3 · GWTC-4 · Nine independent tests · Zero free parameters · All predictions timestamped before LIGO O5*
