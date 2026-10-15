"""
Preprocess ERA5-Land monthly data for Beijing, 1979–2024,
and generate the NetCDF input used in subsequent analyses.
"""

from pathlib import Path
import argparse

import geopandas as gpd
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ERA5_FILE = ROOT / "data" / "raw" / "source" / "era5land_monthly_1979_2024.nc"
DEFAULT_BEIJING_SHP = ROOT / "data" / "raw" / "source" / "beijing_boundary" / "beijing.shp"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "ERA5_Land_Beijing_1979_2024.nc"

START_DATE = "1979-01-01"
END_DATE = "2024-12-31"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the Beijing bounding-box ERA5-Land monthly NetCDF."
    )
    parser.add_argument("--era5-file", type=Path, default=DEFAULT_ERA5_FILE)
    parser.add_argument("--beijing-shp", type=Path, default=DEFAULT_BEIJING_SHP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def get_time_dim(ds):
    if "valid_time" in ds.dims:
        return "valid_time"
    if "time" in ds.dims:
        return "time"

    candidates = [d for d in ds.dims if "time" in d.lower() or "valid" in d.lower()]
    if not candidates:
        raise ValueError("No time dimension found.")
    return candidates[0]


def normalize_longitude(ds):
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(
            longitude=(((ds.longitude + 180) % 360) - 180)
        ).sortby("longitude")
    return ds


def convert_units(ds, time_dim):
    for var in ["stl1", "stl2", "stl3", "stl4", "t2m", "mx2t", "mn2t"]:
        if var in ds:
            ds[var] = ds[var] - 273.15
            ds[var].attrs["units"] = "°C"
            ds[var].attrs["conversion_note"] = "Converted from K to °C"

    if "tp" in ds:
        days = ds[time_dim].dt.days_in_month
        ds["tp"] = ds["tp"] * 1000.0 * days
        ds["tp"].attrs["units"] = "mm/month"
        ds["tp"].attrs["long_name"] = "Monthly total precipitation"
        ds["tp"].attrs["conversion_note"] = (
            "Converted to monthly total precipitation: tp * 1000 * days_in_month."
        )

    if "ssrd" in ds:
        ds["ssrd"] = ds["ssrd"] / 86400.0
        ds["ssrd"].attrs["units"] = "W m-2"
        ds["ssrd"].attrs["long_name"] = "Surface solar radiation downward flux"
        ds["ssrd"].attrs["conversion_note"] = (
            "Converted to mean flux: ssrd / 86400."
        )

    return ds


def screen_ranges(ds):
    for var in ds.data_vars:
        da = ds[var]
        units = da.attrs.get("units", "")

        if "°C" in units:
            ds[var] = da.where((da > -40) & (da < 60))
        elif units == "mm/month":
            ds[var] = da.where((da >= 0) & (da < 1000))
        elif units in {"W m-2", "W/m²", "W m**-2"}:
            ds[var] = da.where((da >= 0) & (da < 500))

    return ds


def main():
    args = parse_args()

    era5_file = args.era5_file.resolve()
    shp_file = args.beijing_shp.resolve()
    output_file = args.output.resolve()

    if not era5_file.exists():
        raise FileNotFoundError(era5_file)
    if not shp_file.exists():
        raise FileNotFoundError(shp_file)

    ds = xr.open_dataset(era5_file)
    ds = normalize_longitude(ds)

    time_dim = get_time_dim(ds)
    ds = ds.sel({time_dim: slice(START_DATE, END_DATE)})

    gdf = gpd.read_file(shp_file).to_crs("EPSG:4326")
    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds

    ds = ds.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
    )

    ds = convert_units(ds, time_dim)
    ds = screen_ranges(ds)

    ds.attrs["Description"] = (
        "ERA5-Land monthly averaged reanalysis data for the Beijing bounding box, 1979–2024."
    )
    ds.attrs["Note"] = (
        "Bounding-box extraction rather than exact administrative clipping; "
        "temperatures converted to °C; tp to mm/month; ssrd to W m-2; "
        "no LSM threshold mask applied."
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_file)

    print("=" * 72)
    print("ERA5-LAND PREPROCESSING COMPLETE")
    print("=" * 72)
    print(f"Output     : {output_file}")
    print(f"Months     : {ds.sizes.get(time_dim, 0)}")
    print(f"Grid       : {ds.sizes.get('latitude', 0)} x {ds.sizes.get('longitude', 0)}")
    print(f"Variables  : {len(ds.data_vars)}")


if __name__ == "__main__":
    main()