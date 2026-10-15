# Data

This directory contains the data used by the reproducibility workflow.

## Derived data

The `derived/` directory contains analysis-ready datasets distributed with
this repository:

    beijing_analysis_grid.csv
    era5land_beijing_monthly_1979_2024.csv
    fldas_beijing_monthly_1982_2024.csv

These files are sufficient for reproducing most of the main statistical
analyses without repeating the full upstream preprocessing.

## Raw data

The `raw/` directory is reserved for original third-party datasets and
intermediate preprocessing products.

Raw source datasets are not redistributed in this repository.

Instructions for obtaining and organizing the required raw inputs are provided
in:

    raw/README.md

Some spatial sensitivity analyses require additional raw inputs, including the
ERA5-Land intermediate NetCDF file and the lake-cover field.