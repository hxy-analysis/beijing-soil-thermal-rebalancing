# Raw data

Raw and third-party source datasets are not redistributed in this repository.
Users who wish to reproduce the preprocessing steps should obtain the
corresponding datasets from their original providers and place them in the
expected local directories described below.

The analysis-ready derived series required for the main statistical analyses
are provided in `data/derived/`.

## Expected raw-data structure

A convenient local directory structure is:

    data/raw/
    ├── ERA5_Land_Beijing_1979_2024.nc
    ├── clake.nc
    └── source/
        ├── era5land/
        │   └── <ERA5-Land monthly source file>
        ├── fldas/
        │   ├── FLDAS_NOAH01_C_GL_M*.nc4
        │   └── ...
        ├── beijing_boundary/
        │   ├── beijing.shp
        │   ├── beijing.shx
        │   ├── beijing.dbf
        │   ├── beijing.prj
        │   └── ...
        └── beijing_dem.tif

Paths can also be supplied explicitly through the command-line arguments of
the preprocessing scripts.

## 1. ERA5-Land

Dataset:

ERA5-Land monthly averaged data from 1950 to present, distributed by the
Copernicus Climate Data Store (CDS).

DOI:

10.24381/cds.68d2bb30

The study uses monthly data for 1979–2024 on the regular 0.1° latitude–
longitude grid.

Variables required by the preprocessing and subsequent analyses include
2-m air temperature, soil temperature in four layers, soil water content in
four layers, surface latent and sensible heat fluxes, surface solar radiation
downwards, and total precipitation.

The ERA5-Land soil layers are:

- Layer 1: 0–7 cm
- Layer 2: 7–28 cm
- Layer 3: 28–100 cm
- Layer 4: 100–289 cm

The script

    scripts/analysis/00_prepare_era5land_input.py

subsets the source data to the Beijing study domain, standardizes the required
variables and units, applies basic range screening, and generates:

    data/raw/ERA5_Land_Beijing_1979_2024.nc

This intermediate NetCDF file is intentionally excluded from version control.

The script

    scripts/analysis/01_prepare_era5land_series.py

then produces the analysis-ready monthly regional series distributed in:

    data/derived/era5land_beijing_monthly_1979_2024.csv

## 2. FLDAS

Dataset:

FLDAS Noah Land Surface Model L4 Global Monthly 0.1° × 0.1°
(MERRA-2 and CHIRPS), Version 1.

Product short name:

    FLDAS_NOAH01_C_GL_M

Provider:

NASA Goddard Earth Sciences Data and Information Services Center
(GES DISC).

The study uses monthly data for 1982–2024.

The preprocessing script expects the monthly NetCDF files to follow the
original product naming convention:

    FLDAS_NOAH01_C_GL_M*.nc4

By default, these files can be placed under:

    data/raw/source/fldas/

The script

    scripts/analysis/00b_prepare_fldas_series.py

identifies FLDAS grid-cell centers within the Beijing administrative boundary,
calculates cosine-of-latitude weighted regional means, and generates:

    data/derived/fldas_beijing_monthly_1982_2024.csv

The derived file contains monthly raw regional means of near-surface air
temperature and soil temperature in the four FLDAS soil layers.

## 3. Beijing administrative boundary

Administrative-division data were ultimately derived from the National
Platform for Common Geospatial Information Services (Tianditu):

https://cloudcenter.tianditu.gov.cn/administrativeDivision

The Beijing municipal boundary used in this study was extracted from a
processed version of the national administrative-division dataset and was
used for spatial subsetting, grid-cell screening, and visualization.

The expected local location is:

    data/raw/source/beijing_boundary/beijing.shp

All associated shapefile components should be stored in the same directory,
for example:

    beijing.shp
    beijing.shx
    beijing.dbf
    beijing.prj

The administrative-boundary data are third-party source data and are not
redistributed in this repository.

## 4. Digital elevation model

The 90-m digital elevation model used in this study was obtained from the
Resource and Environmental Science Data Platform:

https://www.resdc.cn

The DEM product was compiled from SRTM V4.1 data.

The expected local filename is:

    data/raw/source/beijing_dem.tif

The script

    scripts/analysis/00c_prepare_analysis_grid.py

clips the DEM to the Beijing municipal boundary and aggregates elevation to
the ERA5-Land 0.1° grid using average resampling. It then generates:

    data/derived/beijing_analysis_grid.csv

The distributed grid table contains:

    grid_id
    longitude
    latitude
    DEM_mean

The original DEM is third-party source data and is not redistributed in this
repository.

## 5. Lake-cover data

Lake-cover sensitivity analysis requires the ERA5-Land static lake-cover
field.

Variable:

    cl

The field represents the fraction of each grid cell covered by lakes, ranging
from 0 to 1.

The local file used by the analysis is:

    data/raw/clake.nc

This file is used only by:

    scripts/analysis/10_lake_cover_sensitivity.py

The original lake-cover file is not redistributed in this repository.

## Data redistribution

The raw datasets listed above remain subject to the terms and licences of
their original providers and are therefore not included in this repository.

For reproduction of the main statistical analyses without repeating the
upstream preprocessing, use the analysis-ready files provided in
`data/derived/`.