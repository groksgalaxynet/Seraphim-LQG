# Seraphim LQG Framework

**LQG Patch Geometry, Gravitational Wave Energy Loss, and a Unified Compactness Equation**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20271290.svg)](https://doi.org/10.5281/zenodo.20271290)

> **264 BBH events · GWTC-2.1 · GWTC-3 · GWTC-4 · Eight independent tests + one internal consistency check · Zero free parameters**
>
> WORKING PAPER — PRIORITY CLAIM — NOT YET PEER REVIEWED

---

## What This Is

An independent theoretical physics framework connecting Loop Quantum Gravity (LQG) patch geometry to observed gravitational wave energy loss in binary black hole mergers. From two inputs — the LQG area spectrum and the Robertson minimum uncertainty principle — the framework derives a geometric activation constant and predicts the octave depth of BBH merger frequencies with zero free parameters.

**The master equation:**

```
n(C) = 3.561 + 3.506 × C
```

**The core prediction:** octave depth n = 5.314 for equal-mass BBH mergers (C = 0.5), confirmed against 264 events across three independent GWOSC catalogs at 0.05σ, representing 19,901,988 posterior samples.

**The activation constant** (critical — exponent is 84, not 4):

```
K₀ = 1.1467 × 10^84 Hz²     derived from c²/(128π²·γ_area·ℓ_P²)
```

This project was built by an independent researcher using physical intuition and AI-assisted implementation. The math and pipelines need independent verification. If you find errors, that is useful. Nothing is hidden — every test prints raw output to CSV.

---

## Quickstart (Independent Verification)

Run the pipeline against public GWOSC data and confirm the clustering result. The entire verification takes one command and a few hours of download time.

**Step 1 — Install dependencies:**

```bash
pip install numpy scipy h5py pandas
```

**Step 2 — Download the GWOSC HDF5 posterior files:**

| Catalog  | Zenodo Record                                                      | Size    |
|----------|--------------------------------------------------------------------|---------|
| GWTC-2.1 | [zenodo.org/records/6513631](https://zenodo.org/records/6513631)   | ~22.7 GB |
| GWTC-3   | [zenodo.org/records/8177023](https://zenodo.org/records/8177023)   | ~23.6 GB |
| GWTC-4   | [zenodo.org/records/16053484](https://zenodo.org/records/16053484) | ~14.6 GB |

Place all `.h5` / `.hdf5` files in one directory (subdirectories are fine — scripts walk recursively).

**Step 3 — Run the primary clustering test:**

```bash
cd /path/to/your/hdf5/folder
python seraphim_gap_structure_v3.py
```

**Step 4 — Check these numbers:**

| What to check           | Expected value        |
|-------------------------|-----------------------|
| Combined median n       | ~5.329                |
| Events in BBH band      | 257 / 264 (97.3%)     |
| Sigma from n = 5.314    | < 0.10                |
| Robertson vs Heisenberg | Robertson 19× closer  |

Results write to `seraphim_gap_v3_results.csv` (per-event) and print to terminal.

**If your numbers match: the core empirical result is independently confirmed.**
**If they don't: open an issue with your output. That is exactly what this repo needs.**

> **Note on disk space:** Full three-catalog download is ~60 GB. If space is limited, GWTC-4 (Zenodo 16053484, ~14.6 GB) is the most recent and gives 110 events. Run the same script — catalog is auto-detected.

---

## Key Constants

| Symbol        | Value                 | Meaning                                                  |
|---------------|-----------------------|----------------------------------------------------------|
| K₀            | 1.1467 × 10^84 Hz²    | Geometric activation constant — exponent is 84           |
| γ_area        | 0.2375                | Area-counting Immirzi parameter (Meissner 2004)          |
| γ_entropy     | 0.12738 = ln(2)/(π√3) | Entropy-counting Immirzi parameter                       |
| n_flip        | 3.561                 | Realm boundary (Robertson minimum flip)                  |
| n_BBH         | 5.314                 | Equal-mass BBH ground state                              |
| slope         | 3.506                 | Compactness coupling = χ(S²) × (n_BBH − n_flip)         |
| N_Hub         | 201.87 octaves        | Planck–Hubble ring span                                  |
| H₀_predicted  | 73.96 km/s/Mpc        | Hubble constant prediction (0.92σ from SH0ES 2024)       |

---

## Repository Structure

### Python Pipelines — Tests on HDF5 Posterior Samples

All scripts run against GWOSC HDF5 posterior files and write results to CSV. No results are hardcoded.

| Script                         | What It Tests                                                                         |
|--------------------------------|---------------------------------------------------------------------------------------|
| `seraphim_gap_structure_v3.py` | **START HERE** — core clustering, n per event, band membership                        |
| `seraphim_shape_test.py`       | Population shape, normality, j = ½ vs j = 1 (27× chi-squared), band membership       |
| `seraphim_convention_test.py`  | Robertson vs Heisenberg — per-event n under both conventions, mean comparison         |
| `seraphim_redshift_v4.py`      | Redshift invariance — n stable to z = 2.15                                            |
| `seraphim_gauss_bonnet.py`     | Gauss–Bonnet topology identity β + 2E[χ²_eff] = 1.000                                |
| `seraphim_anova_test.py`       | Cross-catalog ANOVA — no significant between-catalog drift                            |
| `seraphim_snr_inclination.py`  | SNR and inclination robustness — partial correlations null                            |
| `waveform.py`                  | Waveform family independence — 0.070% across IMR/SEOBNR/NR                           |
| `Qtest.py`                     | Spin and mass-ratio independence — multivariate OLS, partial r(χ_eff\|q) and r(q\|χ_eff) |
| `seraphim_k0_sweep.py`         | K₀ physical selection — sweeps 4 decades, data minimum at predicted A* vs 20 null trials on 19M samples |

### CSV Outputs — Accompanying Data

| File                                   | Contents                                                              |
|----------------------------------------|-----------------------------------------------------------------------|
| `seraphim_gap_v3_results.csv`          | Per-event: n, M_total, chi_eff, redshift, lum_dist, gap, catalog      |
| `seraphim_shape_per_event.csv`         | Per-event: median_n, std_n, in_bbh_band, catalog                      |
| `seraphim_shape_tail.csv`              | Band membership and tail summary                                      |
| `seraphim_shape_normality.csv`         | Normality test results                                                |
| `seraphim_convention_results.csv`      | Robertson vs Heisenberg n per event                                   |
| `seraphim_redshift_results.csv`        | Redshift, luminosity distance, n per event                            |
| `seraphim_gauss_bonnet_results.csv`    | Gauss–Bonnet per event (264 events)                                   |
| `seraphim_gb_deduped.csv`              | Gauss–Bonnet deduplicated (175 unique events)                         |
| `seraphim_anova_per_event.csv`         | ANOVA inputs per event                                                |
| `seraphim_anova_descriptives.csv`      | Catalog-level descriptive statistics                                  |
| `seraphim_anova_tests.csv`             | ANOVA test statistics and p-values                                    |
| `seraphim_anova_pairwise.csv`          | Pairwise catalog comparisons                                          |
| `seraphim_snr_per_event.csv`           | SNR, inclination, band membership (264 events)                        |
| `seraphim_snr_cuts.csv`                | SNR cut robustness summary                                            |
| `seraphim_inclination_bins.csv`        | Inclination bin statistics                                            |
| `seraphim_inclination_corr.csv`        | Inclination correlation results                                       |

### Papers

All papers are working preprints, priority timestamped on Zenodo. Companion papers are theoretical extensions built on the confirmed framework — they are not independently confirmed and should be read as priority claims.

| File                                    | Description                                          | Status         |
|-----------------------------------------|------------------------------------------------------|----------------|
| `seraphim_v19_6.docx`                   | Main paper v19.6 — current version                   | Empirical core |
| `immirzi_bridge_v6.docx`                | Two Immirzi values, N☉ first-principles derivation   | Companion      |
| `tiling_identity_addendum_v7.docx`      | Spin foam amplitudes, photon sphere boundary         | Companion      |
| `tiling_identity.docx`                  | Algebraic proof: tiling postulate is an identity     | Companion      |
| `hubble_tension_v3.docx`                | H₀ = 73.96 km/s/Mpc from Immirzi mismatch            | Companion      |
| `seesaw_scale_v4.docx`                  | k_seesaw = 11 by four independent routes             | Companion      |
| `degen_v3_v4_corrected.docx`            | Compact binary degeneracy, BNS/NSBH/EMRI spectrum    | Companion      |
| `seraphim_nsun_subthreshold.docx`       | GW191219 as sub-threshold activation event           | Companion      |
| `seraphim_unified_compendium_v3.docx`   | All 16 parts embedded                                | Reference      |
| `seraphim_claims_audit.docx`            | 63-item claims audit, 5 confidence tiers             | Audit          |
| `seraphim_companion_audit.docx`         | 30-item companion papers audit                       | Audit          |

**Extensions folder:** Speculative extensions (octave hierarchy, helix geometry, GUT desert mapping) are in `/extensions`. These are timestamped priority claims only — not results.

---

## Confirmed Test Results

| Test                              | Result                                                              | Script                         |
|-----------------------------------|---------------------------------------------------------------------|----------------------------|
| BBH n prediction                  | 0.05σ from n = 5.314, 257/264 in band                               | `seraphim_gap_structure_v3.py` |
| Robertson vs Heisenberg           | Robertson mean 5.329 vs Heisenberg mean 4.3 — convention falsified  | `seraphim_convention_test.py`  |
| K₀ physical selection             | Data minimum at predicted A* vs null floor, 19.9M samples           | `seraphim_k0_sweep.py`         |
| Redshift invariance               | r = 0.010, p = 0.875 to z = 2.15                                    | `seraphim_redshift_v4.py`      |
| Waveform independence             | Mean n differs 0.070% across families                               | `waveform.py`                  |
| j = ½ selection                   | 27× chi-squared over j = 1                                          | `seraphim_shape_test.py`       |
| Spin and mass-ratio independence  | partial r(χ_eff\|q) = 0.60–0.81, partial r(q\|χ_eff) = 0.85–0.88   | `Qtest.py`                     |
| Gauss–Bonnet identity             | β + 2E[χ²_eff] = 1.000, residual < 10⁻⁵                            | `seraphim_gauss_bonnet.py`     |
| SNR robustness                    | r(n, SNR \| C_eff, q) = −0.001, p = 0.986                           | `seraphim_snr_inclination.py`  |
| Inclination                       | r collapses 0.025 → 0.015 after χ_eff, q control                   | `seraphim_snr_inclination.py`  |
| Cross-catalog ANOVA               | No significant between-catalog drift                                | `seraphim_anova_test.py`       |

---

## Data Sources

All tests run against GWOSC HDF5 posterior sample files. All data is publicly available.

- **GWTC-2.1** — [Zenodo 6513631](https://zenodo.org/records/6513631)
- **GWTC-3** — [Zenodo 8177023](https://zenodo.org/records/8177023)
- **GWTC-4** — [Zenodo 16053484](https://zenodo.org/records/16053484)

**HDF5 structure note:** The three catalogs use different internal layouts. GWTC-2.1 uses compound datasets under waveform keys. GWTC-3 uses `C01+XPHM` with a `posterior_samples` subgroup. GWTC-4 uses `XPHM` keys without a Tidal/NSBH wrapper. All scripts handle this automatically, you do not need to know the internal structure.

---

## Open Questions (Honest)

- Exact baryon loading suppression path connecting Stage 2 hadronic term to η_b
- Formal semiclassical EPRL derivation of the tiling identity (algebraic proof is complete; dynamical origin is open)
- Whether the Hubble tension prediction H₀ = 73.96 km/s/Mpc survives Euclid 2026
- arXiv submission pending gr-qc endorsement

---

## Falsification

**Primary event: LIGO O5 data release.** The framework predicts n = 5.314 for all equal-mass BBH events regardless of redshift, mass, or detector sensitivity. This either holds on fresh data or it does not. No adjustment is possible after the fact.

**Secondary events:** Roman Space Telescope (~late 2026, MAST) will test H₀ and IMBH microlensing predictions. Euclid 2026 will test the Hubble tension prediction independently.

---

## Visualizations

Interactive visualization suite (separate repository):

**[groksgalaxynet.github.io/seraphim-html](https://groksgalaxynet.github.io/seraphim-html)**

All visualizations are built from the CSV outputs in this repository.

---

## Zenodo DOIs (Priority Timestamps)

| Record                    | Contents                                                  | DOI                                                                |
|---------------------------|-----------------------------------------------------------|--------------------------------------------------------------------|
| **Main framework (v19.6)**| Core paper + Immirzi Bridge v6 + Tiling Addendum v7       | [10.5281/zenodo.20271290](https://doi.org/10.5281/zenodo.20271290) |
| Companion papers          | Tiling Identity, Hubble Tension, Degen, Zero Point, Ring  | [10.5281/zenodo.19266568](https://doi.org/10.5281/zenodo.19266568) |
| v18.4 integrated edition  | Main paper with companion derivations embedded            | [10.5281/zenodo.19418111](https://doi.org/10.5281/zenodo.19418111) |
| Corrected expansions      | Corrected companion papers + Unified Compendium v3        | [10.5281/zenodo.19574318](https://doi.org/10.5281/zenodo.19574318) |

---

## Contributing / Verification

**What we need:**

1. Run `seraphim_gap_structure_v3.py` against the GWOSC HDF5 data and confirm the CSV output matches what is in this repo
2. Run any other test script and report whether results reproduce
3. Audit the derivation chain in the main paper — especially the K₀ derivation (Appendix B) and the tiling identity cancellation (Section 3)
4. Flag any inconsistency between claimed results and CSV data

**Credit:** Anyone who contributes meaningful verification work will be credited by name on the Zenodo preprint record. That is a citable, timestamped scientific contribution — real resume credit for a physics student.

No formal background required. The CSV outputs are plain data. The Python scripts are self-contained. If something does not reproduce, open a GitHub issue with your output.

---

## License

All code: MIT License.
All papers: © lis10inc, all rights reserved.
Zenodo records establish priority timestamp independent of peer review status.

---

*Seraphim LQG Framework · lis10inc · Independent Researcher · Houston, TX*
*Zero free parameters · Priority timestamped · Not yet peer reviewed*
