"""
Lake-cover sensitivity analysis for the primary K3-AR3 model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]

ERA5_FILE = (
    REPO_ROOT
    / "data"
    / "raw"
    / "ERA5_Land_Beijing_1979_2024.nc"
)

LAKE_FILE = (
    REPO_ROOT
    / "data"
    / "raw"
    / "clake.nc"
)

GRID_FILE = (
    REPO_ROOT
    / "data"
    / "derived"
    / "beijing_analysis_grid.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "tables"
)

LAYERS = [
    "stl1",
    "stl2",
    "stl3",
    "stl4",
]

LAYER_LABELS = {
    "stl1": "Layer 1",
    "stl2": "Layer 2",
    "stl3": "Layer 3",
    "stl4": "Layer 4",
}

K = 3
AR_ORDER = 3
COMMON_LAG = 6

EXPECTED_GRID_COUNT = 169
EXPECTED_MONTHS = 552

COORDINATE_TOLERANCE = 0.051

SCENARIOS = {
    "Original_169": None,
    "LakeCover_le_5pct": 0.05,
    "LakeCover_le_10pct": 0.10,
}


def resolve_coord_name(
    dataset,
    candidates,
    label,
):
    for name in candidates:
        if (
            name in dataset.coords
            or name in dataset.dims
        ):
            return name

    raise ValueError(
        f"Could not identify {label} coordinate."
    )


def nearest_coordinate(
    target,
    coordinates,
):
    coordinates = np.asarray(
        coordinates,
        dtype=float,
    )

    index = int(
        np.argmin(
            np.abs(
                coordinates - target
            )
        )
    )

    matched = float(
        coordinates[index]
    )

    difference = abs(
        matched - target
    )

    if (
        difference
        > COORDINATE_TOLERANCE
    ):
        raise ValueError(
            "Coordinate matching distance "
            "is too large: "
            f"target={target}, "
            f"matched={matched}."
        )

    return matched


def load_grid():
    grid = pd.read_csv(
        GRID_FILE
    )

    required = [
        "grid_id",
        "longitude",
        "latitude",
    ]

    missing = [
        column
        for column in required
        if column not in grid.columns
    ]

    if missing:
        raise ValueError(
            f"Grid file missing columns: "
            f"{missing}"
        )

    grid = grid[
        required
    ].copy()

    if (
        len(grid)
        != EXPECTED_GRID_COUNT
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_GRID_COUNT} grids, "
            f"found {len(grid)}."
        )

    if (
        grid[
            [
                "longitude",
                "latitude",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate longitude/latitude "
            "pairs found."
        )

    return grid


def add_lake_cover(
    grid,
):
    with xr.open_dataset(
        LAKE_FILE
    ) as lake_dataset:

        if "cl" not in lake_dataset.data_vars:
            raise ValueError(
                "Lake-cover file "
                "has no 'cl' variable."
            )

        latitude_name = (
            resolve_coord_name(
                lake_dataset,
                [
                    "latitude",
                    "lat",
                ],
                "latitude",
            )
        )

        longitude_name = (
            resolve_coord_name(
                lake_dataset,
                [
                    "longitude",
                    "lon",
                ],
                "longitude",
            )
        )

        latitudes = np.asarray(
            lake_dataset[
                latitude_name
            ].values,
            dtype=float,
        )

        longitudes = np.asarray(
            lake_dataset[
                longitude_name
            ].values,
            dtype=float,
        )

        lake_cover = (
            lake_dataset["cl"]
            .squeeze(
                drop=True
            )
        )

        values = []

        for row in grid.itertuples(
            index=False
        ):
            latitude = (
                nearest_coordinate(
                    float(
                        row.latitude
                    ),
                    latitudes,
                )
            )

            longitude = (
                nearest_coordinate(
                    float(
                        row.longitude
                    ),
                    longitudes,
                )
            )

            value = lake_cover.sel(
                {
                    latitude_name:
                        latitude,
                    longitude_name:
                        longitude,
                }
            ).values

            value = float(
                np.asarray(
                    value
                ).squeeze()
            )

            values.append(
                value
            )

    output = grid.copy()

    output[
        "lake_cover"
    ] = values

    if (
        output[
            "lake_cover"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Missing lake-cover values "
            "were found."
        )

    return output


def prepare_series(
    values,
    dates,
):
    series = pd.Series(
        np.asarray(
            values,
            dtype=float,
        ),
        index=dates,
    )

    climatology = (
        series
        .groupby(
            series.index.month
        )
        .transform("mean")
    )

    anomaly = (
        series
        - climatology
    )

    x = np.arange(
        len(anomaly),
        dtype=float,
    )

    y = anomaly.to_numpy(
        dtype=float
    )

    valid = np.isfinite(
        y
    )

    slope, intercept = (
        np.polyfit(
            x[valid],
            y[valid],
            1,
        )
    )

    output = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    output[valid] = (
        y[valid]
        - (
            intercept
            + slope * x[valid]
        )
    )

    return pd.Series(
        output,
        index=dates,
    )


def regional_mean(
    dataset,
    grid,
    variable,
    latitude_name,
    longitude_name,
):
    latitudes = np.asarray(
        dataset[
            latitude_name
        ].values,
        dtype=float,
    )

    longitudes = np.asarray(
        dataset[
            longitude_name
        ].values,
        dtype=float,
    )

    series_list = []
    weights = []
    matched_pairs = []

    for row in grid.itertuples(
        index=False
    ):
        latitude = (
            nearest_coordinate(
                float(
                    row.latitude
                ),
                latitudes,
            )
        )

        longitude = (
            nearest_coordinate(
                float(
                    row.longitude
                ),
                longitudes,
            )
        )

        values = (
            dataset[
                variable
            ]
            .sel(
                {
                    latitude_name:
                        latitude,
                    longitude_name:
                        longitude,
                }
            )
            .values
        )

        values = np.asarray(
            values,
            dtype=float,
        ).squeeze()

        series_list.append(
            values
        )

        weights.append(
            np.cos(
                np.deg2rad(
                    latitude
                )
            )
        )

        matched_pairs.append(
            (
                longitude,
                latitude,
            )
        )

    if (
        len(
            set(
                matched_pairs
            )
        )
        != len(grid)
    ):
        raise ValueError(
            "Duplicate ERA5-Land "
            "grid matches."
        )

    matrix = np.vstack(
        series_list
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = np.isfinite(
        matrix
    )

    numerator = np.nansum(
        matrix
        * weights[:, None],
        axis=0,
    )

    denominator = np.sum(
        valid
        * weights[:, None],
        axis=0,
    )

    if (
        denominator <= 0
    ).any():
        raise ValueError(
            "Invalid spatial weights."
        )

    return (
        numerator
        / denominator
    )


def fit_k3_ar3(
    air,
    soil,
):
    df = pd.DataFrame(
        {
            "air": air,
            "soil": soil,
        }
    )

    air_cols = []
    state_cols = []

    for lag in range(
        K + 1
    ):
        column = (
            f"air_lag{lag}"
        )

        df[column] = (
            df["air"]
            .shift(lag)
        )

        air_cols.append(
            column
        )

    for lag in range(
        1,
        AR_ORDER + 1,
    ):
        column = (
            f"soil_lag{lag}"
        )

        df[column] = (
            df["soil"]
            .shift(lag)
        )

        state_cols.append(
            column
        )

    df = (
        df.iloc[
            COMMON_LAG:
        ]
        .dropna(
            subset=[
                "soil",
                *air_cols,
                *state_cols,
            ]
        )
        .copy()
    )

    y = (
        df["soil"]
        .to_numpy(
            dtype=float
        )
    )

    X_air = sm.add_constant(
        df[
            air_cols
        ].to_numpy(
            dtype=float
        ),
        has_constant="add",
    )

    X_state = sm.add_constant(
        df[
            state_cols
        ].to_numpy(
            dtype=float
        ),
        has_constant="add",
    )

    X_full = sm.add_constant(
        df[
            air_cols
            + state_cols
        ].to_numpy(
            dtype=float
        ),
        has_constant="add",
    )

    air_model = sm.OLS(
        y,
        X_air,
    ).fit()

    state_model = sm.OLS(
        y,
        X_state,
    ).fit()

    full_model = sm.OLS(
        y,
        X_full,
    ).fit()

    sse_air = float(
        np.sum(
            air_model.resid ** 2
        )
    )

    sse_state = float(
        np.sum(
            state_model.resid ** 2
        )
    )

    sse_full = float(
        np.sum(
            full_model.resid ** 2
        )
    )

    return {
        "n":
            len(df),

        "R2_air_only":
            float(
                air_model.rsquared
            ),

        "R2_state_only":
            float(
                state_model.rsquared
            ),

        "R2_full":
            float(
                full_model.rsquared
            ),

        "Air_partial_R2_given_state":
            float(
                (
                    sse_state
                    - sse_full
                )
                / sse_state
            ),

        "State_partial_R2_given_air":
            float(
                (
                    sse_air
                    - sse_full
                )
                / sse_air
            ),
    }


def run_scenario(
    dataset,
    grid,
    dates,
    latitude_name,
    longitude_name,
    scenario,
):
    air_raw = regional_mean(
        dataset,
        grid,
        "t2m",
        latitude_name,
        longitude_name,
    )

    if (
        np.nanmedian(
            air_raw
        )
        > 150
    ):
        air_raw = (
            air_raw
            - 273.15
        )

    air = prepare_series(
        air_raw,
        dates,
    )

    rows = []

    for layer in LAYERS:
        soil_raw = regional_mean(
            dataset,
            grid,
            layer,
            latitude_name,
            longitude_name,
        )

        if (
            np.nanmedian(
                soil_raw
            )
            > 150
        ):
            soil_raw = (
                soil_raw
                - 273.15
            )

        soil = prepare_series(
            soil_raw,
            dates,
        )

        result = fit_k3_ar3(
            air,
            soil,
        )

        rows.append(
            {
                "scenario":
                    scenario,

                "n_grids":
                    len(grid),

                "soil_layer":
                    layer,

                "layer_label":
                    LAYER_LABELS[
                        layer
                    ],

                **result,
            }
        )

    return rows


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid = load_grid()

    grid = add_lake_cover(
        grid
    )

    print(
        "\nLake-cover grid counts"
    )

    print(
        "Original:",
        len(grid),
    )

    print(
        "Lake cover <=5%:",
        int(
            (
                grid[
                    "lake_cover"
                ]
                <= 0.05
            ).sum()
        ),
    )

    print(
        "Lake cover <=10%:",
        int(
            (
                grid[
                    "lake_cover"
                ]
                <= 0.10
            ).sum()
        ),
    )

    with xr.open_dataset(
        ERA5_FILE
    ) as dataset_raw:

        time_name = (
            resolve_coord_name(
                dataset_raw,
                [
                    "valid_time",
                    "time",
                ],
                "time",
            )
        )

        latitude_name = (
            resolve_coord_name(
                dataset_raw,
                [
                    "latitude",
                    "lat",
                ],
                "latitude",
            )
        )

        longitude_name = (
            resolve_coord_name(
                dataset_raw,
                [
                    "longitude",
                    "lon",
                ],
                "longitude",
            )
        )

        required = [
            "t2m",
            *LAYERS,
        ]

        missing = [
            variable
            for variable in required
            if variable
            not in dataset_raw.data_vars
        ]

        if missing:
            raise ValueError(
                "ERA5-Land missing "
                f"variables: {missing}"
            )

        dataset = (
            dataset_raw[
                required
            ]
            .sortby(
                time_name
            )
            .load()
        )

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            dataset[
                time_name
            ].values
        )
    ).to_period(
        "M"
    ).to_timestamp()

    if (
        len(dates)
        != EXPECTED_MONTHS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_MONTHS} months, "
            f"found {len(dates)}."
        )

    all_rows = []

    for (
        scenario,
        threshold,
    ) in SCENARIOS.items():

        if threshold is None:
            selected = (
                grid.copy()
            )

        else:
            selected = (
                grid.loc[
                    grid[
                        "lake_cover"
                    ]
                    <= threshold
                ]
                .copy()
            )

        all_rows.extend(
            run_scenario(
                dataset,
                selected,
                dates,
                latitude_name,
                longitude_name,
                scenario,
            )
        )

    results = pd.DataFrame(
        all_rows
    )

    baseline = (
        results.loc[
            results[
                "scenario"
            ]
            == "Original_169",
            [
                "soil_layer",
                "Air_partial_R2_given_state",
                "State_partial_R2_given_air",
                "R2_full",
            ],
        ]
        .rename(
            columns={
                "Air_partial_R2_given_state":
                    "Air_baseline",

                "State_partial_R2_given_air":
                    "State_baseline",

                "R2_full":
                    "R2_full_baseline",
            }
        )
    )

    comparison = (
        results.merge(
            baseline,
            on="soil_layer",
            how="left",
        )
    )

    comparison[
        "Air_diff_from_original"
    ] = (
        comparison[
            "Air_partial_R2_given_state"
        ]
        - comparison[
            "Air_baseline"
        ]
    )

    comparison[
        "State_diff_from_original"
    ] = (
        comparison[
            "State_partial_R2_given_air"
        ]
        - comparison[
            "State_baseline"
        ]
    )

    comparison[
        "R2_full_diff_from_original"
    ] = (
        comparison[
            "R2_full"
        ]
        - comparison[
            "R2_full_baseline"
        ]
    )

    results.to_csv(
        OUTPUT_DIR
        / "lake_cover_sensitivity.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "lake_cover_sensitivity_comparison.csv",
        index=False,
    )

    print(
        "\nLake-cover sensitivity"
    )

    print(
        comparison[
            [
                "scenario",
                "n_grids",
                "layer_label",
                "Air_partial_R2_given_state",
                "Air_diff_from_original",
                "State_partial_R2_given_air",
                "State_diff_from_original",
                "R2_full",
                "R2_full_diff_from_original",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()