# Seraphim LQG Framework

**LQG Patch Geometry, Gravitational Wave Energy Loss, and a Unified Compactness Equation**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18852768.svg)](https://doi.org/10.5281/zenodo.18852768)

> **264 BBH events · GWTC-2.1 · GWTC-3 · GWTC-4 · Nine independent tests · Zero free parameters**
> 
> WORKING PAPER — PRIORITY CLAIM — NOT YET PEER REVIEWED

-----

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

-----

## Quickstart (Independent Verification)

This is what we need from an independent verifier: run the pipeline against the public GWOSC data and confirm the clustering result. The entire verification takes one command and a few hours of download time.

**Step 1 — Install dependencies:**

```bash
pip install numpy scipy h5py pandas
```

**Step 2 — Download the GWOSC HDF5 posterior files:**

|Catalog |Zenodo Record                                                     |Size  |
|--------|------------------------------------------------------------------|------|
|GWTC-2.1|[zenodo.org/records/6513631](https://zenodo.org/records/6513631)  |~90 GB|
|GWTC-3  |[zenodo.org/records/8177023](https://zenodo.org/records/8177023)  |~75 GB|
|GWTC-4  |[zenodo.org/records/16053484](https://zenodo.org/records/16053484)|~30 GB|

Place all `.h5` / `.hdf5` files in one directory (subdirectories are fine — the script walks recursively).

**Step 3 — Run the primary clustering test:**

```bash
cd /path/to/your/hdf5/folder
python seraphim_gap_structure_v3.py
```

Or point to the folder explicitly:

```bash
python seraphim_gap_structure_v3.py --dir /path/to/hdf5/folder
```

**Step 4 — Check these numbers:**

|What to check          |Expected value      |
|-----------------------|--------------------|
|Combined median n      |~5.329              |
|Events in BBH band     |257 / 264 (97.3%)   |
|Sigma from n=5.314     |< 0.10              |
|Robertson vs Heisenberg|Robertson 19× closer|

Results are written to `seraphim_gap_v3_results.csv` (per-event) and printed to terminal.

**If your numbers match: the core empirical result is independently confirmed.**
**If they don’t: open an issue. That is exactly what this repo needs.**

> Note on disk space: the full three-catalog download is ~195 GB. If you only have space for one catalog, GWTC-4 (Zenodo 16053484, ~30 GB) is the most recent and gives 86 events. Run the same script — it detects the catalog automatically.

-----

## Key Constants

|Symbol      |Value                |Meaning                                                 |
|------------|---------------------|--------------------------------------------------------|
|K₀          |1.1467 × 10^84 Hz²   |Geometric activation constant                           |
|γ_area      |0.2375               |Area-counting Immirzi parameter (Meissner 2004)         |
|γ_entropy   |0.12738 = ln(2)/(π√3)|Entropy-counting Immirzi parameter                      |
|n_flip      |3.561                |Realm boundary (Robertson flip)                         |
|n_BBH       |5.314                |Equal-mass BBH ground state                             |
|slope       |3.506                |Compactness coupling gradient = χ(S²) × (n_BBH − n_flip)|
|N_Hub       |201.87 octaves       |Planck–Hubble ring span                                 |
|H₀_predicted|73.96 km/s/Mpc       |Hubble constant (0.92σ from SH0ES 2024)                 |

-----

## Repository Structure

### Python Pipelines (Tests on HDF5 Posterior Samples)

All scripts run against GWOSC HDF5 posterior files and write results to CSV. No results are hardcoded.

|Script                        |What It Tests                                                      |
|------------------------------|-------------------------------------------------------------------|
|`seraphim_gap_structure_v3.py`|**START HERE** — core clustering test, n per event, band membership|
|`seraphim_shape_test.py`      |Population shape, normality, band membership across catalogs       |
|`seraphim_convention_test.py` |Robertson vs Heisenberg — Robertson confirmed 19× over Heisenberg  |
|`seraphim_redshift_v4.py`     |Redshift invariance — n stable to z = 2.15                         |
|`seraphim_gauss_bonnet.py`    |Gauss-Bonnet topology identity β + 2E[χ²_eff] = 1.000              |
|`seraphim_snr_inclination.py` |SNR and inclination robustness — partial correlations              |
|`seraphim_anova_test.py`      |ANOVA across three catalogs                                        |
|`seraphim_beta_posterior.py`  |Beta power-law exponent from posterior samples                     |
|`seraphim_width_test.py`      |Band width and distribution tails                                  |
|`partial_corr.py`             |Partial correlation control tests                                  |
|`Qtest.py`                    |Mass ratio (q) stratification                                      |
|`beta.py`                     |Beta exponent Monte Carlo                                          |
|`waveform.py`                 |Waveform family independence test                                  |
|`chirpmass.py`                |Chirp mass analysis                                                |

### Roman Space Telescope Pipelines (Pre-built, Awaiting Data ~Late 2026)

|Script                  |What It Tests                                                     |
|------------------------|------------------------------------------------------------------|
|`roman_h0_pipeline.py`  |H₀ direct measurement via microlensing                            |
|`roman_imbh_pipeline.py`|IMBH detection — six test categories against framework predictions|

### CSV Outputs (All Test Results from Our Runs)

Pre-computed results from our pipeline runs are included for comparison. Run the scripts yourself to reproduce them independently.

|File                                  |Contents                                                                 |
|--------------------------------------|-------------------------------------------------------------------------|
|`seraphim_gap_v3_results.csv`         |Per-event: n_seraphim, M_total, chi_eff, redshift, lum_dist, gap, catalog|
|`seraphim_shape_per_event.csv`        |Per-event: median_n, std_n, in_bbh_band, catalog                         |
|`seraphim_convention_results.csv`     |Robertson vs Heisenberg per event                                        |
|`seraphim_redshift_results.csv`       |Redshift/distance per event                                              |
|`seraphim_snr_per_event.csv`          |SNR, inclination, band membership (264 events)                           |
|`seraphim_gauss_bonnet_results.csv`   |Gauss-Bonnet per event (264 events)                                      |
|`seraphim_gb_deduped.csv`             |Gauss-Bonnet deduplicated                                                |
|`seraphim_q_per_event.csv`            |Mass ratio, beta_posterior, q_bin per event                              |
|`seraphim_orbital_v2_per_event.csv`   |Spin observables, Kerr ratio, chi_orb                                    |
|`seraphim_anova_per_event.csv`        |ANOVA inputs per event                                                   |
|`seraphim_anova_descriptives.csv`     |Catalog-level descriptive statistics                                     |
|`seraphim_anova_tests.csv`            |ANOVA test statistics and p-values                                       |
|`seraphim_shape_tail.csv`             |Tail and band membership summary                                         |
|`seraphim_shape_normality.csv`        |Normality test results                                                   |
|`seraphim_q_bins.csv`                 |Mass ratio bin statistics                                                |
|`seraphim_q_beta_strat.csv`           |Beta stratified by mass ratio                                            |
|`planck_seraphim_v3_dn_table.csv`     |Planck chain comparison, Hubble tension test                             |
|`seraphim_snr_cuts.csv`               |SNR cut robustness summary                                               |
|`seraphim_inclination_bins.csv`       |Inclination bin statistics                                               |
|`seraphim_inclination_corr.csv`       |Inclination correlation results                                          |
|`seraphim_orbital_v2_correlations.csv`|Orbital observable correlations                                          |
|`2file_seraphim_beta_mc_results.csv`  |Beta Monte Carlo summary                                                 |
|`seraphim_q_summary.json`             |Mass ratio summary (JSON)                                                |

### Papers

All papers are working preprints. Priority timestamped on Zenodo. **The companion papers are theoretical extensions built on the confirmed framework outputs — they are not independently confirmed and should be read as priority claims, not results.**

|File                                       |Description                                      |Status        |
|-------------------------------------------|-------------------------------------------------|--------------|
|`seraphim_v18_8.docx`                      |Main paper v18.8 — current version               |Empirical core|
|`seraphim_unified_compendium_v3.docx`      |All 16 parts embedded                            |Reference     |
|`tiling_identity.docx`                     |Algebraic proof: tiling postulate is an identity |Companion     |
|`tiling_identity_addendum_spinfoam_v5.docx`|Spin foam amplitudes, j=1/2 minimum-action       |Companion     |
|`immirzi_bridge_v3.docx`                   |Two Immirzi values, hadronic origin of n_flip    |Companion     |
|`hubble_tension_v3.docx`                   |H₀ = 73.96 km/s/Mpc from Immirzi mismatch        |Companion     |
|`seesaw_scale_v4.docx`                     |k_seesaw = 11 by four independent routes         |Companion     |
|`degen_v3_v4_corrected.docx`               |Compact binary degeneracy, BNS/NSBH/EMRI spectrum|Companion     |
|`seraphim_nsun_subthreshold.docx`          |GW191219 as sub-threshold activation event       |Companion     |
|`seraphim_claims_audit.docx`               |63-item claims audit, 5 confidence tiers         |Audit         |
|`seraphim_companion_audit.docx`            |30-item companion papers audit                   |Audit         |

**Extensions folder:** All highly speculative extensions (octave hierarchy, helix geometry, GUT desert mapping, M31 dust analysis) are in `/extensions`. These are timestamped priority claims only — not results.

-----

## Data Sources

Tests run against GWOSC HDF5 posterior sample files. All data is publicly available.

- **GWTC-2.1** — [Zenodo 6513631](https://zenodo.org/records/6513631)
- **GWTC-3** — [Zenodo 8177023](https://zenodo.org/records/8177023)
- **GWTC-4** — [Zenodo 16053484](https://zenodo.org/records/16053484)

**HDF5 structure note:** GWTC-2.1 and GWTC-3 use `file → posterior_samples → C01:IMRPhenomXPHM → datasets`. GWTC-4 uses `file → C00:IMRPhenomXPHM → datasets` (no wrapper group). All scripts handle this automatically via recursive `find_waveform_group` — you do not need to know the internal structure.

-----

## Confirmed Test Results

|Test                   |Result                                         |Script                        |
|-----------------------|-----------------------------------------------|------------------------------|
|BBH n prediction       |0.05σ from n=5.314, 257/264 in band            |`seraphim_gap_structure_v3.py`|
|Robertson vs Heisenberg|Robertson 19× closer on 19.9M samples          |`seraphim_convention_test.py` |
|Redshift invariance    |Spearman r = 0.010, p = 0.875 to z = 2.15      |`seraphim_redshift_v4.py`     |
|Waveform independence  |Mean n differs by 0.070% across families       |`waveform.py`                 |
|Gauss-Bonnet identity  |β + 2E[χ²_eff] = 1.000, residual < 1e-5        |`seraphim_gauss_bonnet.py`    |
|SNR partial correlation|r(n, SNR | C_eff, q) = 0.000                   |`seraphim_snr_inclination.py` |
|Inclination            |r = 0.221 → 0.015 after controlling χ_eff and q|`seraphim_snr_inclination.py` |
|j = 1/2 selection      |27× chi² improvement over j = 1                |`seraphim_shape_test.py`      |
|ANOVA across catalogs  |No significant between-catalog drift           |`seraphim_anova_test.py`      |

-----

## Open Questions (Honest)

- Exact baryon loading suppression path connecting Stage2 to η_b
- Formal derivation of why k_seesaw = 11 from LQG spin foam geometry
- Whether the companion paper Hubble tension prediction (H₀ = 73.96) survives Euclid 2026
- arXiv submission pending endorsement (gr-qc section)

-----

## Falsification

**Primary event: LIGO O5 data release.** The framework predicts n = 5.314 for all equal-mass BBH events regardless of redshift, mass, or detector sensitivity. This either holds on fresh data or it doesn’t. No adjustment is possible after the fact.

**Secondary events:** Roman Space Telescope (~late 2026, MAST) will test H₀ and IMBH predictions. Euclid 2026 will test the Hubble tension prediction independently.

-----

## Visualizations

Interactive visualization suite (separate repository):

**[groksgalaxynet.github.io/seraphim-html](https://groksgalaxynet.github.io/seraphim-html)**

All visualizations built from the CSV outputs in this repository.

-----

## Zenodo DOIs (Priority Timestamps)

|Record                  |Contents                                                                                     |DOI                                                               |
|------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------|
|Main framework (v19)  |Core paper                                                                     |[10.5281/zenodo.20127464](https://doi.org/10.5281/zenodo.20127464)|
|Companion papers        |Tiling Identity, Immirzi Bridge, Hubble Tension, Degen, BB/Hubble, Zero Point, Observer, Ring|[10.5281/zenodo.19266568](https://doi.org/10.5281/zenodo.19266568)|
|v18.4 integrated edition|Main paper with companion derivations embedded                                               |[10.5281/zenodo.19418111](https://doi.org/10.5281/zenodo.19418111)|
|Corrected expansions    |All corrected companion papers + Unified Compendium v3                                       |[10.5281/zenodo.19574318](https://doi.org/10.5281/zenodo.19574318)|

-----

## Contributing / Verification

**What we need:**

1. Run `seraphim_gap_structure_v3.py` against the GWOSC HDF5 data and confirm the CSV output matches what is in this repo
1. Run any of the other test scripts and report whether results reproduce
1. Audit the derivation chain in the main paper — especially the K₀ derivation (Appendix B) and the tiling identity cancellation (Section 10)
1. Flag any inconsistency between claimed results and CSV data

**Credit:** Anyone who contributes meaningful verification work will be credited by name on the Zenodo preprint record. That is a citable, timestamped scientific contribution — real resume credit for a physics student.

No formal background required to start. The CSV outputs are plain data. The Python scripts are self-contained. If something doesn’t reproduce, open a GitHub issue with your output and we will look at it.

-----

## License

All code: MIT License.
All papers: © lis10inc, all rights reserved.
Zenodo records establish priority timestamp independent of peer review status.

-----

*Seraphim LQG Framework · lis10inc · Independent Researcher · Houston, TX*
*Zero free parameters · Priority timestamped · Not yet peer reviewed*
