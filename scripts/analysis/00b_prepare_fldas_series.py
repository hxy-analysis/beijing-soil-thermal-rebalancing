"""
Prepare Beijing-wide monthly FLDAS series for 1982–2024.
"""

from pathlib import Path
import argparse
import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FLDAS_DIR = (
    ROOT / "data" / "raw" / "source" / "fldas"
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
    / "fldas_beijing_monthly_1982_2024.csv"
)

FILE_PATTERN = "FLDAS_NOAH01_C_GL_M*.nc4"

START_DATE = "1982-01-01"
END_DATE = "2024-12-01"
EXPECTED_MONTHS = 516
EXPECTED_GRID_COUNT = 174

VARIABLES = [
    "Tair_f_tavg",
    "SoilTemp00_10cm_tavg",
    "SoilTemp10_40cm_tavg",
    "SoilTemp40_100cm_tavg",
    "SoilTemp100_200cm_tavg",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Beijing-wide FLDAS monthly series."
    )

    parser.add_argument(
        "--fldas-dir",
        type=Path,
        default=DEFAULT_FLDAS_DIR,
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


def resolve_coordinates(ds):
    if "lat" in ds.coords:
        lat_name = "lat"
    elif "latitude" in ds.coords:
        lat_name = "latitude"
    elif "Y" in ds.coords:
        lat_name = "Y"
    else:
        raise ValueError(
            "Latitude coordinate not found."
        )

    if "lon" in ds.coords:
        lon_name = "lon"
    elif "longitude" in ds.coords:
        lon_name = "longitude"
    elif "X" in ds.coords:
        lon_name = "X"
    else:
        raise ValueError(
            "Longitude coordinate not found."
        )

    return lat_name, lon_name


def build_beijing_mask(
    ds,
    shapefile,
    lat_name,
    lon_name,
):
    boundary = (
        gpd.read_file(shapefile)
        .to_crs("EPSG:4326")
    )

    geometry = boundary.geometry.union_all()

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    mask = np.zeros(
        (len(lats), len(lons)),
        dtype=bool,
    )

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            mask[i, j] = geometry.covers(
                Point(
                    float(lon),
                    float(lat),
                )
            )

    return xr.DataArray(
        mask,
        coords={
            lat_name: ds[lat_name],
            lon_name: ds[lon_name],
        },
        dims=(
            lat_name,
            lon_name,
        ),
    )


def weighted_mean(
    da,
    lat_name,
    lon_name,
):
    lat_weights = xr.DataArray(
        np.cos(
            np.deg2rad(
                da[lat_name].values
            )
        ),
        coords={
            lat_name: da[lat_name]
        },
        dims=[lat_name],
    )

    return (
        da
        .weighted(lat_weights)
        .mean(
            dim=[
                lat_name,
                lon_name,
            ],
            skipna=True,
        )
    )


def main():
    args = parse_args()

    fldas_dir = (
        args.fldas_dir
        .expanduser()
        .resolve()
    )

    shapefile = (
        args.beijing_shp
        .expanduser()
        .resolve()
    )

    output_file = (
        args.output
        .expanduser()
        .resolve()
    )

    files = sorted(
        glob.glob(
            str(
                fldas_dir
                / FILE_PATTERN
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No FLDAS files found in {fldas_dir}"
        )

    if not shapefile.exists():
        raise FileNotFoundError(
            shapefile
        )

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
    )

    if "time_bnds" in ds:
        ds = ds.drop_vars(
            "time_bnds"
        )

    ds = ds.sortby(
        "time"
    )

    ds = ds.sel(
        time=slice(
            START_DATE,
            END_DATE,
        )
    )

    time_index = pd.DatetimeIndex(
        ds["time"].values
    )

    month_index = (
        time_index
        .to_period("M")
    )

    expected = pd.period_range(
        "1982-01",
        "2024-12",
        freq="M",
    )

    if month_index.duplicated().any():
        raise ValueError(
            "Duplicate FLDAS months found."
        )

    if not month_index.equals(expected):
        raise ValueError(
            "FLDAS monthly sequence is incomplete."
        )

    if len(time_index) != EXPECTED_MONTHS:
        raise ValueError(
            f"Expected {EXPECTED_MONTHS} months, "
            f"found {len(time_index)}."
        )

    missing_variables = [
        var
        for var in VARIABLES
        if var not in ds.data_vars
    ]

    if missing_variables:
        raise ValueError(
            "Missing FLDAS variables: "
            f"{missing_variables}"
        )

    lat_name, lon_name = (
        resolve_coordinates(ds)
    )

    mask = build_beijing_mask(
        ds,
        shapefile,
        lat_name,
        lon_name,
    )

    grid_count = int(
        mask.sum().item()
    )

    if grid_count != EXPECTED_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRID_COUNT} "
            f"Beijing FLDAS grid cells, "
            f"found {grid_count}."
        )

    ds_beijing = ds.where(
        mask
    )

    output = pd.DataFrame(
        {
            "date": pd.date_range(
                START_DATE,
                END_DATE,
                freq="MS",
            )
        }
    )

    for var in VARIABLES:
        mean_series = weighted_mean(
            ds_beijing[var],
            lat_name,
            lon_name,
        )

        values = np.asarray(
            mean_series.values,
            dtype=float,
        ).squeeze()

        if len(values) != EXPECTED_MONTHS:
            raise ValueError(
                f"Unexpected length for {var}."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{var} contains NaN or Inf."
            )

        output[
            f"{var}_raw"
        ] = values

    output["date"] = (
        output["date"]
        .dt.strftime("%Y-%m")
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    ds.close()

    print("=" * 72)
    print("FLDAS PREPROCESSING COMPLETE")
    print("=" * 72)
    print(f"Files      : {len(files)}")
    print(f"Months     : {len(output)}")
    print(f"Beijing grid cells: {grid_count}")
    print(f"Output     : {output_file}")


if __name__ == "__main__":
    main()