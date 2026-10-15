"""
Prepare Beijing-wide monthly ERA5-Land T2m, soil-temperature,
and soil-moisture series for 1979-2024.

The script calculates cosine-latitude-weighted regional means and
deseasonalized, linearly detrended anomalies for subsequent analyses.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import detrend


# Repository paths
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ERA5_FILE = (
    REPO_ROOT
    / "data"
    / "raw"
    / "ERA5_Land_Beijing_1979_2024.nc"
)

DEFAULT_GRID_FILE = (
    REPO_ROOT
    / "data"
    / "derived"
    / "beijing_analysis_grid.csv"
)

DEFAULT_OUTPUT_FILE = (
    REPO_ROOT
    / "data"
    / "derived"
    / "era5land_beijing_monthly_1979_2024.csv"
)

# Study configuration
TEMPERATURE_VARIABLES = [
    "t2m",
    "stl1",
    "stl2",
    "stl3",
    "stl4",
]

SOIL_MOISTURE_VARIABLES = [
    "swvl1",
    "swvl2",
    "swvl3",
    "swvl4",
]

REQUIRED_VARIABLES = (
    TEMPERATURE_VARIABLES
    + SOIL_MOISTURE_VARIABLES
)

EXPECTED_GRID_COUNT = 169
EXPECTED_MONTH_COUNT = 552

EXPECTED_START = pd.Timestamp("1979-01-01")
EXPECTED_END = pd.Timestamp("2024-12-01")

COORDINATE_TOLERANCE = 0.051

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the Beijing-wide monthly ERA5-Land "
            "T2m/ST/SM derived series for 1979-2024."
        )
    )

    parser.add_argument(
        "--era5",
        type=Path,
        default=DEFAULT_ERA5_FILE,
        help=(
            "Path to the ERA5-Land NetCDF file containing "
            "t2m, stl1-stl4, and swvl1-swvl4."
        ),
    )

    parser.add_argument(
        "--grid",
        type=Path,
        default=DEFAULT_GRID_FILE,
        help=(
            "CSV containing the 169 Beijing grid cells. "
            "Required columns: grid_id, longitude, latitude."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output CSV for the derived Beijing-wide monthly series.",
    )

    return parser.parse_args()

def remove_monthly_climatology(
    series: pd.Series,
) -> pd.Series:

    climatology = series.groupby(
        series.index.month
    ).transform("mean")

    return series - climatology


def linear_detrend(
    series: pd.Series,
) -> pd.Series:
    values = series.to_numpy(dtype=float)
    valid = np.isfinite(values)

    result = np.full(
        values.shape,
        np.nan,
        dtype=float,
    )

    if valid.sum() > 2:
        result[valid] = detrend(
            values[valid],
            type="linear",
        )

    return pd.Series(
        result,
        index=series.index,
        name=series.name,
    )


def prepare_anomaly_series(
    values: np.ndarray,
    time_index: pd.DatetimeIndex,
    name: str,
) -> pd.Series:
    series = pd.Series(
        np.asarray(values, dtype=float),
        index=time_index,
        name=name,
    )

    anomaly = remove_monthly_climatology(
        series
    )

    return linear_detrend(
        anomaly
    )


def nearest_coordinate(
    target: float,
    coordinates: np.ndarray,
    coordinate_name: str,
) -> tuple[float, float]:
    coordinates = np.asarray(
        coordinates,
        dtype=float,
    )

    index = int(
        np.argmin(
            np.abs(coordinates - target)
        )
    )

    matched = float(
        coordinates[index]
    )

    difference = abs(
        matched - target
    )

    if difference > COORDINATE_TOLERANCE:
        raise ValueError(
            f"{coordinate_name} matching distance is too large: "
            f"target={target}, matched={matched}, "
            f"difference={difference:.6f}."
        )

    return matched, difference


def maybe_convert_temperature_to_celsius(
    values: np.ndarray,
) -> tuple[np.ndarray, bool]:
    array = np.asarray(
        values,
        dtype=float,
    )

    finite = array[
        np.isfinite(array)
    ]

    if finite.size == 0:
        return array, False

    if np.nanmedian(finite) > 150.0:
        return array - 273.15, True

    return array, False


def extract_weighted_regional_mean(
    dataset: xr.Dataset,
    grid: pd.DataFrame,
    variable: str,
    latitude_name: str,
    longitude_name: str,
    time_name: str,
) -> np.ndarray:
    latitudes = np.asarray(
        dataset[latitude_name].values,
        dtype=float,
    )

    longitudes = np.asarray(
        dataset[longitude_name].values,
        dtype=float,
    )

    n_time = dataset.sizes[
        time_name
    ]

    grid_series = []
    weights = []
    matched_pairs = []

    for row in grid.itertuples(
        index=False
    ):

        target_longitude = float(
            row.longitude
        )

        target_latitude = float(
            row.latitude
        )

        matched_longitude, lon_difference = (
            nearest_coordinate(
                target_longitude,
                longitudes,
                "longitude",
            )
        )

        matched_latitude, lat_difference = (
            nearest_coordinate(
                target_latitude,
                latitudes,
                "latitude",
            )
        )

        values = dataset[
            variable
        ].sel(
            {
                latitude_name:
                    matched_latitude,
                longitude_name:
                    matched_longitude,
            }
        ).values

        values = np.asarray(
            values,
            dtype=float,
        ).squeeze()

        if (
            values.ndim != 1
            or len(values) != n_time
        ):
            raise ValueError(
                f"{variable} produced an unexpected shape "
                f"at ({matched_longitude}, "
                f"{matched_latitude}): {values.shape}"
            )

        grid_series.append(
            values
        )

        weights.append(
            np.cos(
                np.deg2rad(
                    matched_latitude
                )
            )
        )

        matched_pairs.append(
            (
                matched_longitude,
                matched_latitude,
            )
        )

        if max(
            lon_difference,
            lat_difference,
        ) > 1e-8:
            warnings.warn(
                f"{variable}: requested grid "
                f"({target_longitude}, {target_latitude}) "
                f"was matched to "
                f"({matched_longitude}, {matched_latitude})."
            )

    if (
        len(set(matched_pairs))
        != len(grid)
    ):
        raise ValueError(
            "Nearest-coordinate matching produced duplicate "
            "ERA5-Land grid cells."
        )

    grid_array = np.vstack(
        grid_series
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    regional_mean = np.full(
        n_time,
        np.nan,
        dtype=float,
    )

    for t in range(
        n_time
    ):
        values_t = grid_array[
            :,
            t,
        ]

        valid = (
            np.isfinite(values_t)
            & np.isfinite(weights)
        )

        if valid.sum() == 0:
            continue

        regional_mean[t] = (
            np.sum(
                values_t[valid]
                * weights[valid]
            )
            / np.sum(
                weights[valid]
            )
        )

    return regional_mean

def resolve_coordinate_names(
    dataset: xr.Dataset,
) -> tuple[str, str, str]:
    if "valid_time" in dataset.coords:
        time_name = "valid_time"
    elif "time" in dataset.coords:
        time_name = "time"
    else:
        raise ValueError(
            "Could not identify a time coordinate "
            "('valid_time' or 'time')."
        )

    if "latitude" in dataset.coords:
        latitude_name = "latitude"
    elif "lat" in dataset.coords:
        latitude_name = "lat"
    else:
        raise ValueError(
            "Could not identify a latitude coordinate."
        )

    if "longitude" in dataset.coords:
        longitude_name = "longitude"
    elif "lon" in dataset.coords:
        longitude_name = "lon"
    else:
        raise ValueError(
            "Could not identify a longitude coordinate."
        )

    return (
        time_name,
        latitude_name,
        longitude_name,
    )


def validate_monthly_time_index(
    time_index: pd.DatetimeIndex,
) -> None:

    normalized = (
        time_index
        .to_period("M")
        .to_timestamp()
    )

    if normalized.duplicated().any():
        raise ValueError(
            "Duplicate months were found "
            "in the ERA5-Land time coordinate."
        )

    expected_index = pd.date_range(
        EXPECTED_START,
        EXPECTED_END,
        freq="MS",
    )

    if len(normalized) != EXPECTED_MONTH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MONTH_COUNT} months, "
            f"but found {len(normalized)}."
        )

    if not normalized.equals(
        expected_index
    ):
        missing = expected_index.difference(
            normalized
        )

        extra = normalized.difference(
            expected_index
        )

        raise ValueError(
            "ERA5-Land does not contain the expected "
            "continuous monthly sequence from "
            "1979-01 through 2024-12. "
            f"Missing={list(missing[:10])}; "
            f"extra={list(extra[:10])}."
        )


def load_grid_definition(
    grid_file: Path,
) -> pd.DataFrame:
    if not grid_file.exists():
        raise FileNotFoundError(
            f"Grid CSV not found: {grid_file}"
        )

    grid = pd.read_csv(
        grid_file
    )

    required_columns = [
        "grid_id",
        "longitude",
        "latitude",
    ]

    missing = [
        column
        for column in required_columns
        if column not in grid.columns
    ]

    if missing:
        raise ValueError(
            "Grid CSV is missing required columns: "
            f"{missing}"
        )

    grid = grid[
        required_columns
    ].copy()

    if len(grid) != EXPECTED_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRID_COUNT} "
            f"Beijing grid cells, "
            f"but found {len(grid)}."
        )

    if grid[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate grid_id values were found."
        )

    if grid[
        [
            "longitude",
            "latitude",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate longitude/latitude "
            "pairs were found."
        )

    return grid

# Main preparation workflow
def main() -> None:

    args = parse_args()

    era5_file = args.era5.expanduser().resolve()
    grid_file = args.grid.expanduser().resolve()
    output_file = args.output.expanduser().resolve()

    if not era5_file.exists():
        raise FileNotFoundError(
            f"ERA5-Land NetCDF not found: {era5_file}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid = load_grid_definition(
        grid_file
    )

    print("=" * 72)
    print(
        "PREPARING BEIJING-WIDE "
        "ERA5-LAND MONTHLY DERIVED SERIES"
    )
    print("=" * 72)

    print(
        f"ERA5-Land file : {era5_file}"
    )

    print(
        f"Grid file      : {grid_file}"
    )

    print(
        f"Grid cells     : {len(grid)}"
    )

    with xr.open_dataset(
        era5_file
    ) as dataset_raw:

        (
            time_name,
            latitude_name,
            longitude_name,
        ) = resolve_coordinate_names(
            dataset_raw
        )

        missing_variables = [
            variable
            for variable in REQUIRED_VARIABLES
            if variable
            not in dataset_raw.data_vars
        ]

        if missing_variables:
            raise ValueError(
                "ERA5-Land file is missing "
                "required variables: "
                f"{missing_variables}"
            )

        dataset = (
            dataset_raw[
                REQUIRED_VARIABLES
            ]
            .sortby(
                time_name
            )
            .load()
        )

    time_index = pd.DatetimeIndex(
        pd.to_datetime(
            dataset[
                time_name
            ].values
        )
    )

    time_index = (
        time_index
        .to_period("M")
        .to_timestamp()
    )

    validate_monthly_time_index(
        time_index
    )

    print(
        "Time period     : "
        f"{time_index.min():%Y-%m} "
        "to "
        f"{time_index.max():%Y-%m}"
    )

    print(
        f"Months          : {len(time_index)}"
    )

    regional_raw = {}

    # Temperature variables
    for variable in TEMPERATURE_VARIABLES:

        print(
            f"Extracting {variable} ..."
        )

        values = extract_weighted_regional_mean(
            dataset=dataset,
            grid=grid,
            variable=variable,
            latitude_name=latitude_name,
            longitude_name=longitude_name,
            time_name=time_name,
        )

        values, converted = (
            maybe_convert_temperature_to_celsius(
                values
            )
        )

        regional_raw[
            variable
        ] = values

        unit_message = (
            "Kelvin -> degrees Celsius"
            if converted
            else "no temperature conversion"
        )

        print(
            f"  {unit_message}"
        )

    for variable in SOIL_MOISTURE_VARIABLES:

        print(
            f"Extracting {variable} ..."
        )

        values = extract_weighted_regional_mean(
            dataset=dataset,
            grid=grid,
            variable=variable,
            latitude_name=latitude_name,
            longitude_name=longitude_name,
            time_name=time_name,
        )

        finite = np.asarray(
            values,
            dtype=float,
        )

        finite = finite[
            np.isfinite(finite)
        ]

        if finite.size == 0:
            raise ValueError(
                f"{variable} contains no finite "
                "regional-mean values."
            )

        if (
            np.nanmin(finite) < -0.01
            or np.nanmax(finite) > 1.2
        ):
            warnings.warn(
                f"{variable} is outside the expected "
                "volumetric soil-water-content range. "
                "Check the input variable and units."
            )

        regional_raw[
            variable
        ] = values

    output = pd.DataFrame(
        {
            "date": time_index,
        }
    )

    for variable in TEMPERATURE_VARIABLES:

        output[
            f"{variable}_raw_C"
        ] = regional_raw[
            variable
        ]

        anomaly = prepare_anomaly_series(
            regional_raw[
                variable
            ],
            time_index,
            variable,
        )

        output[
            f"{variable}_anom_deseason_detrend_C"
        ] = anomaly.to_numpy(
            dtype=float
        )

    for variable in SOIL_MOISTURE_VARIABLES:

        output[
            f"{variable}_raw_m3m3"
        ] = regional_raw[
            variable
        ]

        anomaly = prepare_anomaly_series(
            regional_raw[
                variable
            ],
            time_index,
            variable,
        )

        output[
            f"{variable}_anom_deseason_detrend"
        ] = anomaly.to_numpy(
            dtype=float
        )

    if len(output) != EXPECTED_MONTH_COUNT:
        raise RuntimeError(
            "Unexpected number of rows "
            "in the derived output."
        )

    value_columns = [
        column
        for column in output.columns
        if column != "date"
    ]

    missing_counts = output[
        value_columns
    ].isna().sum()

    if (
        missing_counts > 0
    ).any():
        problematic = (
            missing_counts[
                missing_counts > 0
            ]
            .to_dict()
        )

        raise ValueError(
            "The derived regional series "
            "contains missing values: "
            f"{problematic}"
        )

    output_to_save = output.copy()

    output_to_save[
        "date"
    ] = output_to_save[
        "date"
    ].dt.strftime(
        "%Y-%m"
    )

    output_to_save.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("DATA PREPARATION COMPLETE")
    print("=" * 72)

    print(
        f"Output          : {output_file}"
    )

    print(
        f"Rows            : {len(output_to_save)}"
    )

    print(
        f"Variables       : {len(output_to_save.columns) - 1}"
    )

    print(
        "Period          : "
        f"{output_to_save['date'].iloc[0]} "
        "to "
        f"{output_to_save['date'].iloc[-1]}"
    )

    print()
    print(
        "The output contains Beijing-wide raw regional means "
        "and deseasonalized, linearly detrended anomalies "
        "for T2m, four soil-temperature layers, and four "
        "soil-moisture layers."
    )


if __name__ == "__main__":
    main()