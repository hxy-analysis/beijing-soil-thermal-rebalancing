# Run order

This document describes the recommended execution order for reproducing the
preprocessing and analyses associated with the manuscript.

All commands below should be run from the repository root.

## 1. Two reproduction modes

The repository supports two levels of reproduction.

### A. Minimal reproduction

If the analysis-ready files distributed in `data/derived/` are used, the main
statistical analyses can be run without downloading and preprocessing the full
ERA5-Land and FLDAS source archives.

Required distributed files:

    data/derived/
    ├── beijing_analysis_grid.csv
    ├── era5land_beijing_monthly_1979_2024.csv
    └── fldas_beijing_monthly_1982_2024.csv

The main analysis scripts can then be run as described in Section 3.

### B. Full preprocessing

To regenerate the analysis-ready datasets from the original third-party
source data, first prepare the raw inputs described in:

    data/raw/README.md

Then run the preprocessing scripts described in Section 2.

---

## 2. Preprocessing

### Step 00 — ERA5-Land preprocessing

Script:

    scripts/analysis/00_prepare_era5land_input.py

Purpose:

Subsets the ERA5-Land monthly source data to the Beijing study domain,
standardizes the required variables and units, applies basic range screening,
and creates the intermediate Beijing ERA5-Land NetCDF file.

Output:

    data/raw/ERA5_Land_Beijing_1979_2024.nc

Example:

    python scripts/analysis/00_prepare_era5land_input.py \
        --era5-file "<path-to-ERA5-Land-monthly-file>" \
        --beijing-shp "<path-to-beijing.shp>" \
        --output "data/raw/ERA5_Land_Beijing_1979_2024.nc"

This intermediate NetCDF file is not distributed in the public repository.

### Step 00b — FLDAS regional series

Script:

    scripts/analysis/00b_prepare_fldas_series.py

Purpose:

Identifies FLDAS grid-cell centers located within the Beijing administrative
boundary and calculates cosine-of-latitude weighted regional monthly means.

Output:

    data/derived/fldas_beijing_monthly_1982_2024.csv

Example:

    python scripts/analysis/00b_prepare_fldas_series.py \
        --fldas-dir "<directory-containing-FLDAS-files>" \
        --beijing-shp "<path-to-beijing.shp>" \
        --output "data/derived/fldas_beijing_monthly_1982_2024.csv"

This step is independent of the ERA5-Land preprocessing and only needs to be
completed before the cross-product analysis in Step 08.

### Step 00c — Beijing analysis grid and DEM

Script:

    scripts/analysis/00c_prepare_analysis_grid.py

Purpose:

Identifies the 169 ERA5-Land grid-cell centers within the Beijing
administrative boundary and aggregates the DEM to the ERA5-Land 0.1° grid
using average resampling.

Output:

    data/derived/beijing_analysis_grid.csv

Example:

    python scripts/analysis/00c_prepare_analysis_grid.py \
        --era5-file "data/raw/ERA5_Land_Beijing_1979_2024.nc" \
        --dem-file "<path-to-beijing-dem.tif>" \
        --beijing-shp "<path-to-beijing.shp>" \
        --output "data/derived/beijing_analysis_grid.csv"

This step requires the output from Step 00.

### Step 01 — ERA5-Land analysis-ready monthly series

Script:

    scripts/analysis/01_prepare_era5land_series.py

Purpose:

Generates the Beijing-wide monthly ERA5-Land series used by the main
statistical analyses.

Inputs:

    data/raw/ERA5_Land_Beijing_1979_2024.nc
    data/derived/beijing_analysis_grid.csv

Output:

    data/derived/era5land_beijing_monthly_1979_2024.csv

Run:

    python scripts/analysis/01_prepare_era5land_series.py

---

## 3. Main analyses

Once the analysis-ready derived datasets are available, run the analysis
scripts as follows.

### Step 02 — Lag and persistence diagnostics

    python scripts/analysis/02_lag_persistence_diagnostics.py

Main purpose:

Calculates lag correlations between near-surface air temperature and soil
temperature, together with soil-temperature persistence diagnostics.

Primary input:

    data/derived/era5land_beijing_monthly_1979_2024.csv

### Step 03 — Primary conditional models and robustness analyses

    python scripts/analysis/03_conditional_models_robustness.py

Main purpose:

Fits the primary conditional models and performs the main robustness
analyses, including resampling and time-blocked predictive evaluation.

Primary input:

    data/derived/era5land_beijing_monthly_1979_2024.csv

### Step 04 — Air-temperature history-window sensitivity

    python scripts/analysis/04_k12_window_sensitivity.py

Main purpose:

Evaluates alternative air-temperature history windows, including the primary
3-month window and longer sensitivity windows.

Primary input:

    data/derived/era5land_beijing_monthly_1979_2024.csv

### Step 05 — Persistence benchmark

    python scripts/analysis/05_persistence_benchmark.py

Main purpose:

Compares model predictive performance with a persistence benchmark using
time-blocked evaluation.

Primary input:

    data/derived/era5land_beijing_monthly_1979_2024.csv

### Step 06 — Soil-moisture sensitivity

    python scripts/analysis/06_soil_moisture_sensitivity.py

Main purpose:

Tests the sensitivity of the conditional-model results to additional control
for antecedent same-layer soil moisture.

Primary input:

    data/derived/era5land_beijing_monthly_1979_2024.csv

### Step 07 — Seasonal sensitivity

    python scripts/analysis/07_seasonal_sensitivity.py

Main purpose:

Repeats the conditional analysis separately for spring, summer, autumn, and
winter.

Primary input:

    data/derived/era5land_beijing_monthly_1979_2024.csv

### Step 08 — ERA5-Land and FLDAS cross-product comparison

    python scripts/analysis/08_cross_product_era5land_fldas.py

Main purpose:

Assesses whether the main vertical response structure is reproduced across
ERA5-Land and FLDAS.

Inputs:

    data/derived/era5land_beijing_monthly_1979_2024.csv
    data/derived/fldas_beijing_monthly_1982_2024.csv

### Step 09 — Terrain sensitivity

    python scripts/analysis/09_terrain_sensitivity.py

Main purpose:

Evaluates the sensitivity of the conditional-model results to terrain
background and alternative elevation thresholds.

Inputs include:

    data/raw/ERA5_Land_Beijing_1979_2024.nc
    data/derived/beijing_analysis_grid.csv

Because this analysis uses spatially resolved ERA5-Land fields, the
intermediate ERA5-Land NetCDF file from Step 00 is required.

### Step 10 — Lake-cover sensitivity

    python scripts/analysis/10_lake_cover_sensitivity.py

Main purpose:

Tests whether exclusion of ERA5-Land grid cells with larger lake-cover
fractions affects the primary conditional-model results.

Inputs include:

    data/raw/ERA5_Land_Beijing_1979_2024.nc
    data/derived/beijing_analysis_grid.csv
    data/raw/clake.nc

The lake-cover source file is not redistributed and should be obtained from
the original data provider as described in `data/raw/README.md`.

---

## 4. Recommended complete sequence

For full preprocessing and analysis, the recommended order is:

    00_prepare_era5land_input.py
        ↓
    00c_prepare_analysis_grid.py
        ↓
    01_prepare_era5land_series.py
        ↓
    02_lag_persistence_diagnostics.py
    03_conditional_models_robustness.py
    04_k12_window_sensitivity.py
    05_persistence_benchmark.py
    06_soil_moisture_sensitivity.py
    07_seasonal_sensitivity.py
        ↓
    08_cross_product_era5land_fldas.py
    09_terrain_sensitivity.py
    10_lake_cover_sensitivity.py

FLDAS preprocessing can be run independently:

    00b_prepare_fldas_series.py
        ↓
    08_cross_product_era5land_fldas.py

Thus, Step 00b only needs to be completed before Step 08.

---

## 5. Notes

The scripts were validated locally using Python 3.12.

Generated result tables are written to the `results/` directory.

Raw third-party source datasets and intermediate raw-data products are not
tracked by version control. Analysis-ready derived datasets required for the
main statistical analyses are distributed in `data/derived/`.

The conditional partial R² values for the air-temperature time-series block
and the antecedent soil thermal-state block are calculated relative to
different reduced models. They should therefore not be interpreted as
additive variance shares or directly subtracted from one another.