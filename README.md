# Seraphim: LQG Patch Geometry and Gravitational Wave Energy Loss
# https://doi.org/10.5281/zenodo.18852768
**Preprint v16 — February 2026**

> *LQG Patch Geometry, Gravitational Wave Energy Loss, and a Unified Compactness Equation*

A parameter-free geometric framework connecting Loop Quantum Gravity (LQG)
patch structure to observed gravitational wave energy loss in binary mergers.
Validated against 264 events across three independent GWOSC catalogs
(GWTC-2.1, GWTC-3, GWTC-4). Eight independent confirmed tests. Zero free parameters.

**Core result:**

```
n(C) = 3.561 + 3.506 × C
```

Where `n` is octave depth and `C` is compactness. Predicts `n = 5.314` for
black hole mergers (C = 0.5) from first principles alone.

-----

## The Framework in One Paragraph

From the LQG area spectrum (Rovelli & Smolin 1995; Meissner 2004) and the
Robertson minimum uncertainty principle, the geometric activation constant
K₀ = 1.1467 × 10⁸⁴ Hz² is derived from fundamental constants with no
free parameters. This predicts that BBH mergers should cluster at octave
depth n = 5.314. Testing against 264 real GWOSC posterior samples across
three independent catalogs confirms the prediction to within 0.05σ. The
compactness equation n(C) = 3.561 + 3.506·C unifies the prediction across
all compact objects from the quantum geometric singularity (C = 0) to the
black hole event horizon (C = 0.5).

-----

## Repository Structure

```
seraphim_v16.pdf          — The paper (preprint v16)
Qtest.py                  — Main octave depth analysis (runs on GWOSC HDF5 files)
waveform.py               — Waveform independence test (IMRPhenomXPHM vs SEOBNRv4PHM)
beta.py                   — Compactness exponent Monte Carlo (1M iterations)
chirpmass.py              — Chirp mass independence test
partial_corr.py           — Partial correlation: spin vs mass ratio independence
seraphim_convention_test.py  — Robertson vs Heisenberg convention falsification
seraphim_redshift_v4.py   — Redshift invariance test
results/
 seraphim_convention_results.csv   — Convention falsification output (106 events)
 seraphim_redshift_results.csv     — Redshift invariance output (106 events)
 6513631_qtest_run2.csv            — GWTC-2.1 octave results
 6513631_beta_mc_run2.csv          — GWTC-2.1 beta Monte Carlo
 6513631_waveform_delta_run2.csv   — GWTC-2.1 waveform delta
 6513631_chirpmass_run2.csv        — GWTC-2.1 chirp mass test
 8177023_run2_qtest.csv            — GWTC-3 octave results
 8177023_beta_mc_run2.csv          — GWTC-3 beta Monte Carlo
 8177023_run2_waveform_delta.csv   — GWTC-3 waveform delta
 16053484_qtest_run2.csv           — GWTC-4 octave results
 16053484_beta_mc_run2.csv         — GWTC-4 beta Monte Carlo
 2file_seraphim_beta_mc_results.csv — Combined beta summary
```

-----

## Reproducing the Results

### Requirements

```bash
pip install numpy scipy h5py pandas
```

### Data

Download the GWOSC posterior sample HDF5 files from Zenodo.
Place all `.h5` files in a directory before running any script.

|Catalog |Zenodo ID|URL                               |
|--------|---------|----------------------------------|
|GWTC-2.1|6513631  |https://zenodo.org/record/6513631 |
|GWTC-3  |8177023  |https://zenodo.org/record/8177023 |
|GWTC-4  |16053484 |https://zenodo.org/record/16053484|


> **Note:** These are large datasets (~75/100 GB total). Each catalog can be
> run independently. The scripts auto-detect all `.h5` files in the
> working directory.

### Run the main octave depth test

```bash
# Place GWOSC .h5 files in current directory, then:
python Qtest.py
# Output: seraphim_results.csv, seraphim_results.json
```

### Run the waveform independence test

```bash
python waveform.py
# Output: seraphim_waveform_delta_results.csv
# Checks IMRPhenomXPHM vs SEOBNRv4PHM delta
```

### Run the beta Monte Carlo (takes ~1 min)

```bash
python beta.py
# Output: seraphim_beta_mc_results.csv
# 1,000,000 iterations, EoS radius sampling 10-14 km
```

### Run the chirp mass independence test

```bash
python chirpmass.py
# Output: seraphim_chirpmass_results.csv
```

### Run the partial correlation test

```bash
# Requires seraphim_results.csv from Qtest.py first
python partial_corr.py
# Tests spin vs mass ratio independence
```

### Run the convention falsification test

```bash
python seraphim_convention_test.py
# Output: seraphim_convention_results.csv
# Robertson vs Heisenberg, 72-106 events
```

### Run the redshift invariance test

```bash
python seraphim_redshift_v4.py
# Output: seraphim_redshift_results.csv
# Spearman and partial correlation vs redshift
```

-----

## Key Results Summary

|Test                          |Result                                  |Status      |
|------------------------------|----------------------------------------|------------|
|BBH octave depth n = 5.314    |249/264 events in band, mean 5.3197     |✅ CONFIRMED |
|Redshift invariance (Spearman)|BBH r(n,z) = 0.010, p = 0.875           |✅ CONFIRMED |
|Waveform independence         |GWTC-2.1 delta = 0.070%                 |✅ CONFIRMED |
|Robertson vs Heisenberg       |1.0 octave separation, Robertson wins   |✅ CONFIRMED |
|Tidal deformability (GW190425)|Range check 0.003 octave residual       |✅ CONFIRMED |
|NSBH compactness mapping      |GW190814 n=4.189, predicted 4.192       |✅ CONFIRMED |
|Spin/mass ratio independence  |Partial r ≥ 0.60 both, p < 10⁻⁷         |✅ CONFIRMED |
|Redshift partial correlation  |Partial r(z|χ_eff) = 0.111, p = 0.071   |✅ CONFIRMED |
|Compactness exponent β        |β = 0.814 ± CI across 6 independent runs|🔄 Pending O5|

-----

## Fundamental Constants Used

|Symbol           |Value              |Units        |
|-----------------|-------------------|-------------|
|ħ                |1.054571817 × 10⁻³⁴|J s          |
|c                |2.997924580 × 10⁸  |m/s          |
|h                |6.626070150 × 10⁻³⁴|J s          |
|l_P              |1.616255 × 10⁻³⁵   |m            |
|γ (Immirzi, area)|0.2375             |dimensionless|
|K₀               |1.1467 × 10⁸⁴      |Hz²          |

-----

## Falsifiable Predictions

Three predictions remain open and testable with O5 data:

- **Prediction 8:** β → 1.0 with high-precision O5 tidal deformability measurements
- **Prediction 9:** n(C) linear for all compact objects with independent C from tidal Λ
- **Prediction 10:** Slope 3.506 = 2Δn has no renormalization freedom

-----

## Citation

```
[Author]. "LQG Patch Geometry, Gravitational Wave Energy Loss,
and a Unified Compactness Equation." Preprint v16, February 2026.
```

-----

## References

- Rovelli & Smolin, Nucl. Phys. B 442, 593 (1995)
- Meissner, Class. Quant. Grav. 21, 5245 (2004)
- Abbott et al. (LVK), GWTC-3, arXiv:2111.03606 (2021)
- Abbott et al. (LVK), GWTC-2.1, arXiv:2108.01045 (2021)
- Robertson, H.P., Phys. Rev. 34, 163 (1929)
- GWOSC: https://gwosc.org

-----

## License

This repository contains original research. Scripts are released for
reproducibility. Please cite the preprint if you use this work.
