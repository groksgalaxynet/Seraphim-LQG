# Seraphim LQG Framework

**LQG Patch Geometry, Gravitational Wave Energy Loss, and a Unified Compactness Equation**

*lis10inc · Independent Researcher · April 2026*

---

> **Working Paper — Priority Claim — Not Yet Submitted**
>
> Timestamped on Zenodo. arXiv gr-qc submission pending endorsement.

---

## Core Result

```
n(C) = 3.561 + 3.506 × C
```

The octave depth of a compact binary merger is determined entirely by the compactness of its components. Derived from the LQG area spectrum and the Robertson minimum uncertainty principle with **zero free parameters**.

| Quantity | Value | Status |
|---|---|---|
| n_BBH (equal-mass BBH) | **5.314** | Confirmed: 264 events, 0.05σ |
| n_flip (singularity / realm boundary) | **3.561** | Confirmed: universal lower bound |
| Compactness slope | **3.506 = 2Δn = χ(S²)·Δn** | Structural identity |
| K₀ (geometric activation constant) | **1.1467 × 10^84 Hz²** | Derived from CODATA constants |
| γ_area (Immirzi, area-counting) | **0.2375** | Meissner 2004 |
| γ_entropy (Immirzi, entropy-counting) | **0.12738 = ln(2)/(π√3)** | BH thermodynamics |
| H₀_local (predicted) | **73.96 km/s/Mpc** | 0.18σ from SH0ES+JWST 2025 |
| Helix winding number | **W = 1/α = 137.036** | Exact, zero free parameters |

> ⚠️ **K₀ exponent is 84, not 4.** Unicode superscript corruption has previously rendered 10^84 as 10^4 in some renderers. The correct value is K₀ = 1.1467 × 10^84 Hz².

---

## Papers

### Main Paper — v18.4 (April 2026)
**DOI: [10.5281/zenodo.18852768](https://doi.org/10.5281/zenodo.18852768)**

The first integrated edition of the Seraphim LQG framework. Incorporates the main paper (v18.3) with all four companion derivations as new sections. No numerical results changed from v18.3.

New in v18.4:
- **Section 10 — The Tiling Identity:** The framework's one previously unproven bridge (N☉·Aⱼ = π(Δx)²) is proven to be an algebraic identity. γ and ℓ_P² cancel exactly. Verified to 12 significant figures.
- **Section 11 — Spin Foam Addendum:** EPRL amplitude analysis. j = 1/2 is the minimum-action configuration on the Robertson constraint surface. S_total = γK₀√(j(j+1))/ν² is monotone increasing in j; minimum at j = 1/2.
- **Section 12 — Immirzi Bridge:** Closed-form derivation of n_flip = 3.561 from γ_entropy = ln(2)/(π√3) plus hadronic matter correction. Helix winding W = 1/α = 137.036, exact to 8 decimal places, zero free parameters.
- **Section 13 — Hubble Tension:** H₀_local = 73.96 km/s/Mpc derived from Immirzi channel mismatch (γ_area ≠ γ_entropy). Prediction confirmed at 0.18σ against SH0ES+JWST 2025, zero free parameters.

Earlier versions established: n = 5.314 from first principles (v7); 264-event confirmation across three GWOSC catalogs (v8); coupling ladder with 1/α terminus (v9); unified compactness equation (v10); redshift invariance (v11); Robertson/Heisenberg convention falsification (v12/v13); realm-projection geometry (v14); spin and mass-ratio independence at partial correlation level (v15); √(j(j+1)) placement correction (v16); Physical Picture section (v17); Alpha Dual-Role, Topological Factor of 2, Kerr Remnant Spin (v18); gap residual analysis and χₚ null result (v18.3).

---

### Companion Papers — Batch DOI: [10.5281/zenodo.19266568](https://doi.org/10.5281/zenodo.19266568)

#### The Seraphim Octave Hierarchy — Paper 2 v16 (March 2026)
*A Universal Frequency Map from Planck Scale to Hubble Horizon*

Extends the confirmed framework across the full octave spectrum. The compactness inversion C_eff(n) = (n − 3.561) / 3.506 applied across all physical scales from the Planck wall (n = 0) to the Hubble horizon (n = 201.87).

**Confirmed extensions:**
- Periodic table sorted by LQG compactness: all 118 elements classified by condensation state at the n_H = 92.19 ceiling, zero violations
- Coupling ladder terminus at 1/α = 137.036 octaves, structurally confirmed
- Robertson-derived C = ±0.5 symmetry (BBH event horizon and anti-horizon)

**Derived results (priority claims):**
- Two-phase grids (Grid 1 phase 0.055, Grid 2 phase 0.275) derived from n_flip and n_BBH
- CMB temperature identity T_CMB ~ √(ν_Hub) with LQG factor fixed by K₀, γ_area, γ_entropy (predicted 2.740 K vs FIRAS 2.725 K)
- GUT desert walls placed by the χ(S²) ladder to within 0.019 octaves
- Desert center C_eff = π identity (0.13% precision)
- QCD axion prediction: chi k = 56, E = 1.25 μeV (ADMX search window)
- PTA/NANOGrav band: QCD confinement ring antipode at n = 169.735, 14.89 nHz
- AION/AEDGE band: EW symmetry-breaking antipode at n = 160.345, 9.99 μHz
- M31 HELGA dust three-zone structure aligned to condensation ceiling

**Speculative (labeled explicitly):**
- BBH mirror / anti-horizon at n = 1.808 (C = −0.5)
- Cosmological ladder extensions beyond confirmed band

#### Compact Binary Degeneracy — Degen v3/v4 (March 2026)

Derives and corrects the 4η effective compactness formula for all compact binary classes:

```
C_eff = 4η × C_component
n = 3.561 + 3.506 × (4η × C_component)
```

- Equal-mass limit (η = 0.25, C = 0.5) recovers n = 5.314 exactly
- EMRI limit: n → n_flip = 3.561 as q → 0 (LISA prediction)
- GW190814 identified as the unique degenerate event: η = 0.090 encodes C_NS = 0.180 (APR4 EOS, R ≈ 10.4 km) at 0.075σ
- BNS O5 band: n ≈ 4.05–4.30 depending on equation of state
- NSBH formula corrected (v4): mass-weighted two-component formula replacing falsified single-component formula
- Helix v4 appended: octave ring is the 2D projection of a helix with W = 1/α = 137.036, pitch p = N_Hub·α = 1.4731 oct; α–QCD inscribed rectangle derived; open prediction at n ≈ 36.87 (C = 9.5 coupling rung)

#### The Tiling Identity (Incorporated in v18.4 Section 10)
Proof that N☉·Aⱼ = π(Δx)² is an algebraic identity — not a postulate. γ and ℓ_P² cancel exactly, leaving c²/(16π). The framework's foundational status upgraded from "one unproven bridge" to fully derived.

#### The Immirzi Bridge (Incorporated in v18.4 Section 12)
Five-equation derivation of n_flip = 3.561 from γ_entropy. Proton–QCD gap = 2.650 octaves. Helix winding W = 1/α exact.

#### Zero-Point Derivation and Observer and the Ring
Companion derivations of K₀ construction and the observer frame in realm-projection geometry.

#### Hubble Tension Paper (Incorporated in v18.4 Section 13)
Master identity: N_Hub × α = k_seesaw × [½·log₂(γ_area/γ_entropy)] × 2^(n_flip − n_BBH), k_seesaw = 11 derived by two independent paths. H₀_local = 73.96 km/s/Mpc, zero free parameters. Every dark energy modification tested against DESI DR2 (2025) leaves the tension intact — confirming it is not a dark energy problem.

---

## Observational Results Summary

Nine independent tests across 264 events (GWTC-2.1, GWTC-3, GWTC-4):

| Test | Result |
|---|---|
| BBH band clustering | 249/264 BBH events within band; combined mean 5.319, offset 0.005 oct (0.05σ) |
| Robertson vs. Heisenberg convention | 1.000-octave separation confirmed; Robertson selected at full-octave precision |
| Redshift invariance | Partial r(n, z) = 0.111, p = 0.071 (non-significant at partial correlation level) |
| Mass-ratio independence | Partial r(n, η) = −0.042, p = 0.58 (non-significant) |
| Spin independence (χ_eff) | Partial r(n, χ_eff) = −0.050, p = 0.51 (non-significant) |
| χₚ null result | Partial r = 0.20; drops to r ≈ 0, p = 0.99 at q > 0.70 |
| Kerr remnant spin | Spin operates through LQG face activation; partial r(a_f \| log₂M) = −0.972 |
| β exponent (d_eff) | OLS b = 0.720; BEC universality class b = 3/4 inside 95% CI; PN b = 0.5 excluded at 7.5σ |
| Gauss–Bonnet topology | GB-derived β confirmed through topology |
| Catalog consistency (ANOVA) | No significant inter-catalog difference; framework is catalog-invariant |

**Compact object classifier:** n and η alone achieve 100% accuracy on all 264 labeled GWTC events.

---

## Predictions and Falsification

| ID | Prediction | Test |
|---|---|---|
| P1 | BBH events cluster at n = 5.314 in O5 | O5 data release |
| P2 | IMBH mergers at M ≈ 1684 M☉ show gap ≈ 1/α | ET / LISA |
| P3 | β → 0.5 at M ~ 10⁵–10⁷ M☉ (quantum-to-classical transition) | O5+O6 cumulative (~1505 events to discriminate) |
| P4 | H₀_local = 73.96 km/s/Mpc | Euclid 2026, SKAO (±0.3 km/s/Mpc) |
| P5 | BNS mergers at n ≈ 4.05–4.30 band in O5 | O5 BNS events |
| P6 | EMRI limit n → 3.561 as q → 0 | LISA |
| P7 | ADMX axion at chi k = 56, E = 1.25 μeV | ADMX ongoing search |
| P8 | Euclid Hubble tension not resolved by dark energy | DESI DR2+ |

Primary upcoming falsification event: **O5 LIGO data release.**

---

## Interactive Visualization Suite

**[groksgalaxynet.github.io/seraphim-html](https://groksgalaxynet.github.io/seraphim-html/index.html)**

16 interactive HTML visualizations:

| VIZ | Title | Description |
|---|---|---|
| 01 | [Dot Chart](https://groksgalaxynet.github.io/seraphim-html/dot_chart.html) | Event-level scatter distribution across the GWTC catalog |
| 02 | [GWTC Rubin Sky Map](https://groksgalaxynet.github.io/seraphim-html/gwtc_rubin_skymap.html) | Full-sky localization of gravitational wave events |
| 03 | [Line Charts](https://groksgalaxynet.github.io/seraphim-html/line_charts.html) | Spin, mass ratio, and energy loss trends |
| 04 | [Seraphim 4D](https://groksgalaxynet.github.io/seraphim-html/seraphim_4d.html) | n(C) spin-coupling surface across the full BBH population |
| 05 | [Distance Map](https://groksgalaxynet.github.io/seraphim-html/seraphim_distance_map.html) | Luminosity distance distribution and redshift invariance |
| 06 | [Periodic Structure](https://groksgalaxynet.github.io/seraphim-html/seraphim_periodic.html) | Coupling ladder periodicity and 1/α terminus at 137.036 |
| 07 | [Seraphim Ring](https://groksgalaxynet.github.io/seraphim-html/seraphim_ring.html) | LQG area spectrum ring geometry |
| 08 | [Seraphim Ring 2](https://groksgalaxynet.github.io/seraphim-html/seraphim_ring2.html) | Enhanced ring topology and coupling ladder resonance |
| 09 | [Complete Ring](https://groksgalaxynet.github.io/seraphim-html/seraphim_complete_ring.html) | Full-spectrum ring with all octave zones |
| 10 | [Metallic Ring](https://groksgalaxynet.github.io/seraphim-html/metallic_ring.html) | High-fidelity Immirzi parameter phase space |
| 11 | [M31 Ring Mapping](https://groksgalaxynet.github.io/seraphim-html/seraphim_ring_m31.html) | Ring geometry overlaid on M31 dust distribution |
| 12 | [M31 Dust Compactness](https://groksgalaxynet.github.io/seraphim-html/m31_dust_compactness.html) | M31 dust translated to compactness coordinates |
| 13 | [HELGA Dust Viewer](https://groksgalaxynet.github.io/seraphim-html/m31_helga_viewer.html) | HELGA cold dust spatial distribution |
| 14 | [Matter Timeline](https://groksgalaxynet.github.io/seraphim-html/matter_timeline.html) | Octave mapping across cosmic time |
| 15 | [Thermodynamic Phase Space](https://groksgalaxynet.github.io/seraphim-html/seraphim_thermo.html) | Entropy, energy, and phase transition mapping |
| 16 | [Seraphim Study](https://groksgalaxynet.github.io/seraphim-html/seraphim_study.html) | Five-view ring study: full ring, GUT desert, α-bracket, α–QCD rectangle, helix |

---

## Repository Contents

### Analysis Scripts (Python / HDF5)

| File | Description |
|---|---|
| `Qtest.py` | Mass ratio q test across GWTC posterior samples |
| `beta.py` | β exponent Monte Carlo (6×1M runs) |
| `partial_corr.py` | Partial correlation analysis (n vs z, η, χ_eff) |
| `waveform.py` | Waveform key selection and delta analysis |
| `chirpmass.py` | Chirp mass analysis |
| `seraphim_redshift_v4.py` | Redshift invariance test v4 |
| `seraphim_convention_test.py` | Robertson vs. Heisenberg convention falsification test |
| `seraphim_gap_structure_v3.py` | Gap residual and alpha-ladder analysis |
| `seraphim_gauss_bonnet.py` | Gauss–Bonnet topology test |
| `seraphim_beta_posterior.py` | β posterior distribution analysis |
| `seraphim_shape_test.py` | n distribution shape and normality tests |
| `seraphim_width_test.py` | BBH band width characterization |
| `seraphim_anova_test.py` | Inter-catalog ANOVA |
| `seraphim_snr_inclination.py` | SNR cut robustness and inclination analysis |
| `seraphim_redshift_v4.py` | Per-event redshift correlation |
| `data_dwnldr.py` | GWOSC HDF5 data downloader |

**HDF5 catalog access notes:**
- GWTC-2.1 (Zenodo 6513631): compound datasets directly under waveform key, access via `np.array(ds['column'])`
- GWTC-3 (Zenodo 8177023): Group under waveform key containing `posterior_samples`
- GWTC-4 (Zenodo 16053484): Group structure, must target `posterior_samples` explicitly, skip `meta_data`
- Waveform key priority: `C01:IMRPhenomXPHM:HighSpin` → `C01:IMRPhenomXPHM` → `C01:SEOBNRv4PHM`

### Core Result CSVs

| File | Description |
|---|---|
| `seraphim_shape_per_event.csv` | Per-event n, band classification, χ_eff (264 events) |
| `seraphim_gap_v3_results.csv` | Gap residuals and alpha-ladder per event (264 events) |
| `seraphim_q_per_event.csv` | Per-event mass ratio, β posterior, q-bin classification |
| `seraphim_orbital_v2_per_event.csv` | Orbital spin parameters per event |
| `seraphim_gb_deduped.csv` | Gauss–Bonnet results, deduplicated |
| `seraphim_convention_results.csv` | Robertson vs. Heisenberg n comparison per event |
| `seraphim_redshift_results.csv` | Per-event redshift, luminosity distance, band flag |
| `seraphim_snr_per_event.csv` | SNR, inclination, θ_JN per event |
| `seraphim_snr_cuts.csv` | SNR cut robustness summary |
| `seraphim_inclination_bins.csv` | n vs. inclination angle bins |
| `seraphim_inclination_corr.csv` | Inclination correlation tests |
| `seraphim_anova_per_event.csv` | Per-event data for inter-catalog ANOVA |
| `seraphim_anova_descriptives.csv` | Catalog-level descriptive statistics |
| `seraphim_anova_tests.csv` | ANOVA and Kruskal–Wallis test results |
| `seraphim_anova_pairwise.csv` | Pairwise catalog comparisons |
| `seraphim_shape_normality.csv` | Normality tests on n distribution |
| `seraphim_shape_tail.csv` | Band tail statistics by event type |
| `seraphim_q_bins.csv` | n statistics stratified by mass ratio bin |
| `seraphim_q_beta_strat.csv` | β stratified by q |
| `seraphim_q_catalog.csv` | Mass ratio statistics by catalog |
| `seraphim_orbital_v2_correlations.csv` | Orbital parameter correlation matrix |
| `seraphim_gauss_bonnet_results.csv` | Full Gauss–Bonnet dataset (264 events) |
| `planck_seraphim_v3_dn_table.csv` | Planck chain H₀ comparison table |
| `seraphim_q_summary.json` | Summary statistics for q analysis |
| Catalog-specific CSVs (`6513631_*`, `8177023_*`, `16053484_*`) | Per-catalog run2 results |

### Per-Catalog Run2 Data

| File | Description |
|---|---|
| `6513631_beta_mc_run2.csv` | GWTC-2.1 β Monte Carlo run 2 |
| `6513631_chirpmass_run2.csv` | GWTC-2.1 chirp mass run 2 |
| `6513631_qtest_run2.csv` | GWTC-2.1 q test run 2 |
| `6513631_waveform_delta_run2.csv` | GWTC-2.1 waveform delta run 2 |
| `8177023_beta_mc_run2.csv` | GWTC-3 β Monte Carlo run 2 |
| `8177023_run2_qtest.csv` | GWTC-3 q test run 2 |
| `8177023_run2_waveform_delta.csv` | GWTC-3 waveform delta run 2 |
| `16053484_beta_mc_run2.csv` | GWTC-4 β Monte Carlo run 2 |
| `16053484_qtest_run2.csv` | GWTC-4 q test run 2 |

---

## Framework Constants

| Symbol | Value | Meaning |
|---|---|---|
| K₀ | 1.1467 × 10^84 Hz² | Geometric activation constant = c²/(128π²γ_area·ℓ_P²) |
| γ_area | 0.2375 | Area-counting Immirzi parameter (Meissner 2004) |
| γ_entropy | 0.12738 = ln(2)/(π√3) | Entropy-counting Immirzi parameter |
| n_flip | 3.561 | Realm boundary (Robertson flip point) |
| n_BBH | 5.314 | Equal-mass BBH ground state |
| Δn | 1.753 | n_BBH − n_flip |
| slope | 3.506 = χ(S²)·Δn | Compactness coupling gradient |
| N☉ | 6.09 | Activated LQG face count (BBH) |
| χ(S²) | 2 | Euler characteristic of S² |
| W | 137.036 = 1/α | Helix winding number (exact, 8 d.p.) |
| N_Hub | 201.87 octaves | Planck–Hubble ring span |
| k_seesaw | 11 | Hubble tension seesaw scale |
| H₀_local | 73.96 km/s/Mpc | Predicted local Hubble constant |

---

## Error Documentation

Three documented errors corrected in companion paper reprints — none changed framework constants:

1. **K₀ exponent Unicode corruption** — 10^84 rendered as 10^4 via superscript character in some documents. Corrected in all reprints. The literal value is 1.1467 × 10^84 Hz².
2. **Double-squaring error in t_flip** — identified and corrected in companion derivations. No effect on n_flip or any observational result.
3. **φ₁ description error** — labeling error in observer paper. Corrected. No numerical change.

Errors are documented in change logs rather than silently corrected. This is deliberate.

---

## Zenodo DOIs

| Paper | DOI |
|---|---|
| Main paper v18.4 | [10.5281/zenodo.19418111](https://doi.org/10.5281/zenodo.19418111) |
| Companion papers batch | [10.5281/zenodo.19266568](https://doi.org/10.5281/zenodo.19266568) |
| Original deposit | [10.5281/zenodo.18852768](https://doi.org/10.5281/zenodo.18852768) |

---

## Data Sources

- **GWTC-2.1:** Zenodo 6513631
- **GWTC-3:** Zenodo 8177023
- **GWTC-4:** Zenodo 16053484
- All posteriors accessed via GWOSC public release

---

## Citation

```
lis10inc (2026). LQG Patch Geometry, Gravitational Wave Energy Loss,
and a Unified Compactness Equation (v18.4).
Zenodo. https://doi.org/10.5281/zenodo.19418111
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Seraphim LQG Framework · 264 BBH events · Nine independent tests · Zero free parameters · April 2026*
