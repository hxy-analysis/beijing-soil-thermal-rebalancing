# Depth-Dependent Soil Thermal Response in Beijing

This repository contains the analysis code and analysis-ready derived data
associated with the manuscript:

**Depth-Dependent Reorganization of Conditional Contributions from
Near-Surface Air-Temperature Time-Series Information and Antecedent Soil
Thermal State**

The study investigates the depth dependence of soil-temperature responses to
near-surface air-temperature variability in Beijing using ERA5-Land
reanalysis, with complementary analyses based on FLDAS, terrain, soil
moisture, seasonal stratification, persistence benchmarks, and lake-cover
sensitivity tests.

## Repository structure

    .
    ├── README.md
    ├── RUN_ORDER.md
    ├── requirements.txt
    ├── .gitignore
    │
    ├── data/
    │   ├── README.md
    │   ├── derived/
    │   │   ├── beijing_analysis_grid.csv
    │   │   ├── era5land_beijing_monthly_1979_2024.csv
    │   │   └── fldas_beijing_monthly_1982_2024.csv
    │   │
    │   └── raw/
    │       └── README.md
    │
    ├── results/
    │   └── tables/
    │
    └── scripts/
        └── analysis/
            ├── 00_prepare_era5land_input.py
            ├── 00b_prepare_fldas_series.py
            ├── 00c_prepare_analysis_grid.py
            ├── 01_prepare_era5land_series.py
            ├── 02_lag_persistence_diagnostics.py
            ├── 03_conditional_models_robustness.py
            ├── 04_k12_window_sensitivity.py
            ├── 05_persistence_benchmark.py
            ├── 06_soil_moisture_sensitivity.py
            ├── 07_seasonal_sensitivity.py
            ├── 08_cross_product_era5land_fldas.py
            ├── 09_terrain_sensitivity.py
            └── 10_lake_cover_sensitivity.py

## Data included in this repository

Three analysis-ready derived datasets are distributed with the repository.

### ERA5-Land monthly series

    data/derived/era5land_beijing_monthly_1979_2024.csv

This file contains the Beijing-wide monthly ERA5-Land series for 1979–2024,
including raw and processed variables used in the main analyses.

### FLDAS monthly series

    data/derived/fldas_beijing_monthly_1982_2024.csv

This file contains the Beijing-wide monthly FLDAS regional means for
1982–2024 used in the cross-product robustness analysis.

### Beijing analysis grid

    data/derived/beijing_analysis_grid.csv

This file contains the 169 ERA5-Land analysis grid cells used in the spatial
analyses, with grid ID, longitude, latitude, and grid-cell mean elevation.

Raw third-party datasets are not redistributed. Their requirements and
expected local paths are described in:

    data/raw/README.md

## Reproducing the analyses

### Minimal reproduction from the distributed derived data

The distributed analysis-ready files allow the main statistical analyses to
be reproduced without downloading and preprocessing the complete original
ERA5-Land and FLDAS archives.

The core scripts can be run from the repository root in numerical order.

For example:

    python scripts/analysis/02_lag_persistence_diagnostics.py
    python scripts/analysis/03_conditional_models_robustness.py
    python scripts/analysis/04_k12_window_sensitivity.py
    python scripts/analysis/05_persistence_benchmark.py
    python scripts/analysis/06_soil_moisture_sensitivity.py
    python scripts/analysis/07_seasonal_sensitivity.py
    python scripts/analysis/08_cross_product_era5land_fldas.py

The terrain and lake-cover sensitivity analyses additionally require the
corresponding spatial/raw inputs documented in `data/raw/README.md`.

### Full preprocessing

To reproduce the analysis-ready datasets from the original source data, run:

    python scripts/analysis/00_prepare_era5land_input.py
    python scripts/analysis/00b_prepare_fldas_series.py
    python scripts/analysis/00c_prepare_analysis_grid.py
    python scripts/analysis/01_prepare_era5land_series.py

The preprocessing scripts accept command-line arguments for local input and
output paths.

Detailed execution order and script dependencies are provided in
`RUN_ORDER.md`.

## Analysis overview

The scripts cover the following analyses:

- lag-correlation and soil-temperature persistence diagnostics;
- conditional regression models separating the information added by recent
  air-temperature time series and antecedent soil thermal state;
- sensitivity to alternative air-temperature history windows;
- time-blocked predictive evaluation and persistence benchmarks;
- soil-moisture sensitivity;
- seasonal sensitivity;
- ERA5-Land and FLDAS cross-product comparison;
- terrain sensitivity;
- lake-cover sensitivity.

The primary conditional model uses a 3-month air-temperature history window
and an AR(3) representation of antecedent soil thermal state. Robustness
analyses examine alternative model specifications and resampling procedures.

## Important note on conditional partial R²

The conditional partial R² for the air-temperature time-series block and the
conditional partial R² for the antecedent soil thermal-state block are
calculated relative to different reduced models.

They therefore quantify the incremental contribution of each predictor block
conditional on the other block and should not be interpreted as additive
variance shares or directly subtracted from one another.

## Cross-product comparison

The FLDAS analysis is used as a cross-product robustness assessment of the
vertical response structure.

Because both ERA5-Land and FLDAS are model-based land-surface products, this
comparison should not be interpreted as independent observational validation.

## Software

The analysis was validated using Python 3.12.3

Required Python packages are listed in:

    requirements.txt

Install the dependencies with:

    pip install -r requirements.txt

## Outputs

Analysis scripts write generated tables and diagnostic outputs to the
`results/` directory.

Intermediate outputs, raw third-party datasets, Python cache files, IDE
metadata, and temporary validation files are excluded from version control.

## Reproducibility status

The preprocessing and analysis workflow was locally validated by regenerating
the distributed derived datasets and comparing them with the files used in
the manuscript analyses.

The validated workflow includes:

    ERA5-Land source data
        -> ERA5-Land Beijing preprocessing
        -> analysis-ready ERA5-Land monthly series

    FLDAS source data
        -> Beijing spatial screening and weighted averaging
        -> analysis-ready FLDAS monthly series

    ERA5-Land grid + Beijing boundary + DEM
        -> 169-cell Beijing analysis grid

The analysis scripts were subsequently tested against these inputs.

## Citation

If you use this repository, please cite the associated manuscript.

A formal manuscript citation and repository DOI will be added after they
become available.

## License

The analysis code in this repository is released under the MIT License.

The derived datasets and result tables are provided for research
reproducibility. Third-party raw datasets are not redistributed and remain
subject to the terms and licences of their original providers.
