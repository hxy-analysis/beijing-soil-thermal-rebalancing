"""
Prepare the Beijing ERA5-Land analysis grid and grid-cell mean elevation.
"""

from pathlib import Path
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ERA5_FILE = (
    ROOT
    / "data"
    / "raw"
    / "ERA5_Land_Beijing_1979_2024.nc"
)

DEFAULT_DEM_FILE = (
    ROOT
    / "data"
    / "raw"
    / "source"
    / "beijing_dem.tif"
)

DEFAULT_BEIJING_SHP = (
    ROOT
    / "data"
    / "raw"
    / "source"
    / "beijing_boundary"
    / "beijing.shp"
)

DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "derived"
    / "beijing_analysis_grid.csv"
)

EXPECTED_GRID_COUNT = 169


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the Beijing ERA5-Land analysis grid."
    )

    parser.add_argument(
        "--era5-file",
        type=Path,
        default=DEFAULT_ERA5_FILE,
    )

    parser.add_argument(
        "--dem-file",
        type=Path,
        default=DEFAULT_DEM_FILE,
    )

    parser.add_argument(
        "--beijing-shp",
        type=Path,
        default=DEFAULT_BEIJING_SHP,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    era5_file = args.era5_file.expanduser().resolve()
    dem_file = args.dem_file.expanduser().resolve()
    shp_file = args.beijing_shp.expanduser().resolve()
    output_file = args.output.expanduser().resolve()

    for path in [
        era5_file,
        dem_file,
        shp_file,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    ds = xr.open_dataset(
        era5_file
    )

    lats = ds["latitude"].values
    lons = ds["longitude"].values

    boundary = (
        gpd.read_file(shp_file)
        .to_crs("EPSG:4326")
    )

    geometry = (
        boundary
        .geometry
        .union_all()
    )

    rows = []

    for lat in lats:
        for lon in lons:
            point = Point(
                float(lon),
                float(lat),
            )

            if geometry.covers(point):
                rows.append(
                    {
                        "longitude": float(lon),
                        "latitude": float(lat),
                    }
                )

    grid = pd.DataFrame(
        rows
    )

    grid = (
        grid
        .sort_values(
            ["longitude", "latitude"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    if len(grid) != EXPECTED_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRID_COUNT} "
            f"Beijing grid cells, "
            f"found {len(grid)}."
        )

    dem = (
        rxr.open_rasterio(
            dem_file,
            masked=True,
        )
        .squeeze(drop=True)
        .astype("float32")
    )

    if dem.rio.crs is None:
        raise ValueError(
            "DEM has no CRS."
        )

    boundary_dem = boundary.to_crs(
        dem.rio.crs
    )

    dem_clip = dem.rio.clip(
        boundary_dem.geometry,
        boundary_dem.crs,
        drop=True,
    )

    dem_clip = dem_clip.where(
        dem_clip >= 0
    )

    template = xr.DataArray(
        np.zeros(
            (
                len(lats),
                len(lons),
            ),
            dtype=np.float32,
        ),
        coords={
            "latitude": lats,
            "longitude": lons,
        },
        dims=[
            "latitude",
            "longitude",
        ],
    )

    template = (
        template
        .rio.write_crs(
            "EPSG:4326"
        )
        .rio.set_spatial_dims(
            x_dim="longitude",
            y_dim="latitude",
        )
    )

    dem_clip = (
        dem_clip
        .rio.write_nodata(
            np.nan,
            inplace=False,
        )
    )

    dem_era = (
        dem_clip
        .rio.reproject_match(
            template,
            resampling=Resampling.average,
        )
    )

    if (
        "x" in dem_era.dims
        and "longitude" not in dem_era.dims
    ):
        dem_era = dem_era.rename(
            {
                "x": "longitude"
            }
        )

    if (
        "y" in dem_era.dims
        and "latitude" not in dem_era.dims
    ):
        dem_era = dem_era.rename(
            {
                "y": "latitude"
            }
        )

    dem_era = dem_era.assign_coords(
        longitude=lons,
        latitude=lats,
    )

    dem_values = []

    for _, row in grid.iterrows():
        value = dem_era.sel(
            longitude=row["longitude"],
            latitude=row["latitude"],
        ).item()

        dem_values.append(
            float(value)
        )

    grid["DEM_mean"] = dem_values

    if grid["DEM_mean"].isna().any():
        raise ValueError(
            "Missing DEM_mean values found."
        )

    grid.insert(
        0,
        "grid_id",
        np.arange(
            1,
            len(grid) + 1,
        ),
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    ds.close()

    print("=" * 72)
    print("ANALYSIS GRID PREPARATION COMPLETE")
    print("=" * 72)
    print(f"Grid cells : {len(grid)}")
    print(
        "DEM range  : "
        f"{grid['DEM_mean'].min():.3f} "
        f"to {grid['DEM_mean'].max():.3f} m"
    )
    print(f"Output     : {output_file}")


if __name__ == "__main__":
    main()