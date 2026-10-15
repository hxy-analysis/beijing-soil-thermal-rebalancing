"""
Topographic modulation and DEM-threshold sensitivity analysis.
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

GRID_FILE = (
    REPO_ROOT
    / "data"
    / "derived"
    / "beijing_analysis_grid.csv"
)

OUTPUT_DIR = REPO_ROOT / "results" / "tables"

LAYERS = ["stl1", "stl2", "stl3", "stl4"]

LAYER_LABELS = {
    "stl1": "Layer 1",
    "stl2": "Layer 2",
    "stl3": "Layer 3",
    "stl4": "Layer 4",
}

LAYER_DEPTHS = {
    "stl1": "0-7 cm",
    "stl2": "7-28 cm",
    "stl3": "28-100 cm",
    "stl4": "100-289 cm",
}

DEM_THRESHOLDS = [200, 300, 400]
K_VALUES = [3, 6]

AR_ORDER = 3
COMMON_LAG = 6

BOOTSTRAP_REPS = 2000
BLOCK_LENGTH = 36
RANDOM_SEED = 20260814

EXPECTED_GRID_COUNT = 169
EXPECTED_MONTHS = 552

COORDINATE_TOLERANCE = 0.051


def prepare_series(
    values,
    time_index,
):
    series = pd.Series(
        np.asarray(values, dtype=float),
        index=time_index,
    )

    climatology = (
        series
        .groupby(series.index.month)
        .transform("mean")
    )

    deseason = (
        series - climatology
    )

    x = np.arange(
        len(series),
        dtype=float,
    )

    y = deseason.to_numpy(
        dtype=float
    )

    valid = np.isfinite(y)

    slope, intercept = np.polyfit(
        x[valid],
        y[valid],
        1,
    )

    result = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    result[valid] = (
        y[valid]
        - (
            intercept
            + slope * x[valid]
        )
    )

    return pd.Series(
        result,
        index=time_index,
    )


def weighted_mean(
    values,
    weights,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    valid = np.isfinite(
        values
    )

    numerator = np.where(
        valid,
        values * weights[None, :],
        0.0,
    ).sum(axis=1)

    denominator = np.where(
        valid,
        weights[None, :],
        0.0,
    ).sum(axis=1)

    result = np.full(
        values.shape[0],
        np.nan,
    )

    good = denominator > 0

    result[good] = (
        numerator[good]
        / denominator[good]
    )

    return result


def nearest_index(
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

    if (
        abs(matched - target)
        > COORDINATE_TOLERANCE
    ):
        raise ValueError(
            f"Coordinate matching failed: "
            f"target={target}, matched={matched}"
        )

    return index, matched


def extract_grid_matrix(
    dataset,
    variable,
    latitude_name,
    longitude_name,
    lat_indices,
    lon_indices,
):
    data = dataset[
        variable
    ].isel(
        {
            latitude_name:
                xr.DataArray(
                    lat_indices,
                    dims="grid",
                ),
            longitude_name:
                xr.DataArray(
                    lon_indices,
                    dims="grid",
                ),
        }
    ).squeeze(
        drop=True
    )

    time_dims = [
        dim
        for dim in data.dims
        if dim != "grid"
    ]

    if len(time_dims) != 1:
        raise ValueError(
            f"Unexpected dimensions for "
            f"{variable}: {data.dims}"
        )

    data = data.transpose(
        time_dims[0],
        "grid",
    )

    return np.asarray(
        data.values,
        dtype=float,
    )


def build_dynamic_arrays(
    air,
    soil,
    k,
):
    air = np.asarray(
        air,
        dtype=float,
    )

    soil = np.asarray(
        soil,
        dtype=float,
    )

    t = np.arange(
        COMMON_LAG,
        len(air),
    )

    y = soil[t]

    air_block = np.column_stack(
        [
            air[t - lag]
            for lag in range(
                k + 1
            )
        ]
    )

    state_block = np.column_stack(
        [
            soil[t - lag]
            for lag in range(
                1,
                AR_ORDER + 1
            )
        ]
    )

    valid = (
        np.isfinite(y)
        & np.all(
            np.isfinite(
                air_block
            ),
            axis=1,
        )
        & np.all(
            np.isfinite(
                state_block
            ),
            axis=1,
        )
    )

    return (
        y[valid],
        air_block[valid],
        state_block[valid],
    )


def fit_metrics(
    y,
    air_block,
    state_block,
):
    X_air = sm.add_constant(
        air_block,
        has_constant="add",
    )

    X_state = sm.add_constant(
        state_block,
        has_constant="add",
    )

    X_full = sm.add_constant(
        np.column_stack(
            [
                air_block,
                state_block,
            ]
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

    air_partial = (
        (
            sse_state
            - sse_full
        )
        / sse_state
    )

    state_partial = (
        (
            sse_air
            - sse_full
        )
        / sse_air
    )

    return {
        "n":
            len(y),

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
                air_partial
            ),

        "State_partial_R2_given_air":
            float(
                state_partial
            ),

        "BIC_full":
            float(
                full_model.bic
            ),
    }


def fast_partial_r2(
    air,
    soil,
    k,
):
    (
        y,
        air_block,
        state_block,
    ) = build_dynamic_arrays(
        air,
        soil,
        k,
    )

    ones = np.ones(
        (
            len(y),
            1,
        )
    )

    X_air = np.column_stack(
        [
            ones,
            air_block,
        ]
    )

    X_state = np.column_stack(
        [
            ones,
            state_block,
        ]
    )

    X_full = np.column_stack(
        [
            ones,
            air_block,
            state_block,
        ]
    )

    beta_air = np.linalg.lstsq(
        X_air,
        y,
        rcond=None,
    )[0]

    beta_state = np.linalg.lstsq(
        X_state,
        y,
        rcond=None,
    )[0]

    beta_full = np.linalg.lstsq(
        X_full,
        y,
        rcond=None,
    )[0]

    sse_air = np.sum(
        (
            y
            - X_air @ beta_air
        ) ** 2
    )

    sse_state = np.sum(
        (
            y
            - X_state @ beta_state
        ) ** 2
    )

    sse_full = np.sum(
        (
            y
            - X_full @ beta_full
        ) ** 2
    )

    air_partial = (
        (
            sse_state
            - sse_full
        )
        / sse_state
    )

    state_partial = (
        (
            sse_air
            - sse_full
        )
        / sse_air
    )

    return (
        float(air_partial),
        float(state_partial),
    )


def circular_block_indices(
    n,
    block_length,
    rng,
):
    blocks = []

    while sum(
        len(block)
        for block in blocks
    ) < n:

        start = int(
            rng.integers(
                0,
                n,
            )
        )

        blocks.append(
            (
                start
                + np.arange(
                    block_length
                )
            ) % n
        )

    return np.concatenate(
        blocks
    )[:n]


def percentile_ci(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    return (
        float(
            np.mean(values)
        ),
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


def load_inputs():
    grid = pd.read_csv(
        GRID_FILE
    )

    required = [
        "grid_id",
        "longitude",
        "latitude",
        "DEM_mean",
    ]

    missing = [
        col
        for col in required
        if col not in grid.columns
    ]

    if missing:
        raise ValueError(
            f"Grid file missing columns: "
            f"{missing}"
        )

    grid = grid[
        required
    ].copy()

    if len(grid) != EXPECTED_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRID_COUNT} grids, "
            f"found {len(grid)}."
        )

    dataset = xr.open_dataset(
        ERA5_FILE
    )

    time_name = (
        "valid_time"
        if "valid_time"
        in dataset.coords
        else "time"
    )

    latitude_name = (
        "latitude"
        if "latitude"
        in dataset.coords
        else "lat"
    )

    longitude_name = (
        "longitude"
        if "longitude"
        in dataset.coords
        else "lon"
    )

    variables = [
        "t2m",
        *LAYERS,
    ]

    missing_variables = [
        variable
        for variable in variables
        if variable
        not in dataset.data_vars
    ]

    if missing_variables:
        raise ValueError(
            f"ERA5-Land missing variables: "
            f"{missing_variables}"
        )

    dataset = (
        dataset[
            variables
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
    ).to_period(
        "M"
    ).to_timestamp()

    if len(time_index) != EXPECTED_MONTHS:
        raise ValueError(
            f"Expected {EXPECTED_MONTHS} months, "
            f"found {len(time_index)}."
        )

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

    lat_indices = []
    lon_indices = []
    matched_lats = []
    matched_lons = []

    for row in grid.itertuples(
        index=False
    ):
        lat_index, lat = (
            nearest_index(
                float(
                    row.latitude
                ),
                latitudes,
            )
        )

        lon_index, lon = (
            nearest_index(
                float(
                    row.longitude
                ),
                longitudes,
            )
        )

        lat_indices.append(
            lat_index
        )

        lon_indices.append(
            lon_index
        )

        matched_lats.append(
            lat
        )

        matched_lons.append(
            lon
        )

    matched = pd.DataFrame(
        {
            "latitude":
                matched_lats,
            "longitude":
                matched_lons,
        }
    )

    if matched.duplicated().any():
        raise ValueError(
            "Duplicate ERA5-Land grid matches."
        )

    matrices = {}

    for variable in variables:
        matrices[
            variable
        ] = extract_grid_matrix(
            dataset,
            variable,
            latitude_name,
            longitude_name,
            np.asarray(
                lat_indices
            ),
            np.asarray(
                lon_indices
            ),
        )

    dataset.close()

    grid[
        "matched_latitude"
    ] = matched_lats

    return (
        grid,
        matrices,
        time_index,
    )


def build_region_series(
    grid,
    matrices,
    time_index,
):
    output = {}

    weights_all = np.cos(
        np.deg2rad(
            grid[
                "matched_latitude"
            ].to_numpy(
                dtype=float
            )
        )
    )

    for threshold in DEM_THRESHOLDS:

        mountain = (
            grid["DEM_mean"]
            .to_numpy(
                dtype=float
            )
            > threshold
        )

        plain = ~mountain

        if (
            mountain.sum() == 0
            or plain.sum() == 0
        ):
            raise ValueError(
                f"Invalid DEM threshold: "
                f"{threshold}"
            )

        print(
            f"DEM {threshold} m: "
            f"Mountain/Hilly={mountain.sum()}, "
            f"Plain={plain.sum()}"
        )

        for region, mask in [
            (
                "Mountain",
                mountain,
            ),
            (
                "Plain",
                plain,
            ),
        ]:
            for variable, matrix in (
                matrices.items()
            ):
                raw_mean = weighted_mean(
                    matrix[
                        :,
                        mask,
                    ],
                    weights_all[
                        mask
                    ],
                )

                output[
                    (
                        threshold,
                        region,
                        variable,
                    )
                ] = prepare_series(
                    raw_mean,
                    time_index,
                )

    return output


def point_estimates(
    region_series,
    grid,
):
    rows = []

    dem = grid[
        "DEM_mean"
    ].to_numpy(
        dtype=float
    )

    for threshold in DEM_THRESHOLDS:

        region_counts = {
            "Mountain":
                int(
                    np.sum(
                        dem > threshold
                    )
                ),

            "Plain":
                int(
                    np.sum(
                        dem <= threshold
                    )
                ),
        }

        for region in [
            "Mountain",
            "Plain",
        ]:

            air = region_series[
                (
                    threshold,
                    region,
                    "t2m",
                )
            ].to_numpy(
                dtype=float
            )

            for layer in LAYERS:

                soil = region_series[
                    (
                        threshold,
                        region,
                        layer,
                    )
                ].to_numpy(
                    dtype=float
                )

                for k in K_VALUES:

                    (
                        y,
                        air_block,
                        state_block,
                    ) = build_dynamic_arrays(
                        air,
                        soil,
                        k,
                    )

                    metrics = fit_metrics(
                        y,
                        air_block,
                        state_block,
                    )

                    rows.append(
                        {
                            "DEM_threshold_m":
                                threshold,

                            "region":
                                region,

                            "n_grids":
                                region_counts[
                                    region
                                ],

                            "soil_layer":
                                layer,

                            "layer_label":
                                LAYER_LABELS[
                                    layer
                                ],

                            "depth":
                                LAYER_DEPTHS[
                                    layer
                                ],

                            "K":
                                k,

                            "AR_order":
                                AR_ORDER,

                            **metrics,
                        }
                    )

    return pd.DataFrame(
        rows
    )


def bootstrap_analysis(
    region_series,
    point_table,
):
    region_boot = {}
    contrast_boot = {}

    for threshold in DEM_THRESHOLDS:
        for k in K_VALUES:
            for region in [
                "Mountain",
                "Plain",
            ]:
                for layer in LAYERS:

                    region_boot[
                        (
                            threshold,
                            k,
                            region,
                            layer,
                            "Air",
                        )
                    ] = []

                    region_boot[
                        (
                            threshold,
                            k,
                            region,
                            layer,
                            "State",
                        )
                    ] = []

            for layer in LAYERS:

                contrast_boot[
                    (
                        threshold,
                        k,
                        layer,
                        "Air_Mountain_minus_Plain",
                    )
                ] = []

                contrast_boot[
                    (
                        threshold,
                        k,
                        layer,
                        "State_Mountain_minus_Plain",
                    )
                ] = []

    n_months = EXPECTED_MONTHS

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    for _ in range(
        BOOTSTRAP_REPS
    ):
        idx = circular_block_indices(
            n_months,
            BLOCK_LENGTH,
            rng,
        )

        for threshold in DEM_THRESHOLDS:
            for k in K_VALUES:

                current = {}

                for region in [
                    "Mountain",
                    "Plain",
                ]:

                    air = region_series[
                        (
                            threshold,
                            region,
                            "t2m",
                        )
                    ].to_numpy(
                        dtype=float
                    )[idx]

                    for layer in LAYERS:

                        soil = region_series[
                            (
                                threshold,
                                region,
                                layer,
                            )
                        ].to_numpy(
                            dtype=float
                        )[idx]

                        (
                            air_partial,
                            state_partial,
                        ) = fast_partial_r2(
                            air,
                            soil,
                            k,
                        )

                        current[
                            (
                                region,
                                layer,
                            )
                        ] = (
                            air_partial,
                            state_partial,
                        )

                        region_boot[
                            (
                                threshold,
                                k,
                                region,
                                layer,
                                "Air",
                            )
                        ].append(
                            air_partial
                        )

                        region_boot[
                            (
                                threshold,
                                k,
                                region,
                                layer,
                                "State",
                            )
                        ].append(
                            state_partial
                        )

                for layer in LAYERS:

                    air_m, state_m = (
                        current[
                            (
                                "Mountain",
                                layer,
                            )
                        ]
                    )

                    air_p, state_p = (
                        current[
                            (
                                "Plain",
                                layer,
                            )
                        ]
                    )

                    contrast_boot[
                        (
                            threshold,
                            k,
                            layer,
                            "Air_Mountain_minus_Plain",
                        )
                    ].append(
                        air_m - air_p
                    )

                    contrast_boot[
                        (
                            threshold,
                            k,
                            layer,
                            "State_Mountain_minus_Plain",
                        )
                    ].append(
                        state_m - state_p
                    )

    region_rows = []

    for key, values in (
        region_boot.items()
    ):
        (
            threshold,
            k,
            region,
            layer,
            metric,
        ) = key

        mean, low, high = (
            percentile_ci(
                values
            )
        )

        point = point_table.loc[
            (
                point_table[
                    "DEM_threshold_m"
                ]
                == threshold
            )
            & (
                point_table["K"]
                == k
            )
            & (
                point_table[
                    "region"
                ]
                == region
            )
            & (
                point_table[
                    "soil_layer"
                ]
                == layer
            )
        ].iloc[0]

        column = (
            "Air_partial_R2_given_state"
            if metric == "Air"
            else
            "State_partial_R2_given_air"
        )

        region_rows.append(
            {
                "DEM_threshold_m":
                    threshold,

                "K":
                    k,

                "AR_order":
                    AR_ORDER,

                "region":
                    region,

                "soil_layer":
                    layer,

                "layer_label":
                    LAYER_LABELS[
                        layer
                    ],

                "metric":
                    column,

                "point_estimate":
                    float(
                        point[
                            column
                        ]
                    ),

                "bootstrap_mean":
                    mean,

                "CI95_lower":
                    low,

                "CI95_upper":
                    high,

                "bootstrap_reps":
                    BOOTSTRAP_REPS,

                "block_length_months":
                    BLOCK_LENGTH,
            }
        )

    contrast_rows = []

    for key, values in (
        contrast_boot.items()
    ):

        (
            threshold,
            k,
            layer,
            contrast,
        ) = key

        mean, low, high = (
            percentile_ci(
                values
            )
        )

        mountain = point_table.loc[
            (
                point_table[
                    "DEM_threshold_m"
                ]
                == threshold
            )
            & (
                point_table["K"]
                == k
            )
            & (
                point_table[
                    "region"
                ]
                == "Mountain"
            )
            & (
                point_table[
                    "soil_layer"
                ]
                == layer
            )
        ].iloc[0]

        plain = point_table.loc[
            (
                point_table[
                    "DEM_threshold_m"
                ]
                == threshold
            )
            & (
                point_table["K"]
                == k
            )
            & (
                point_table[
                    "region"
                ]
                == "Plain"
            )
            & (
                point_table[
                    "soil_layer"
                ]
                == layer
            )
        ].iloc[0]

        if contrast.startswith(
            "Air"
        ):
            column = (
                "Air_partial_R2_given_state"
            )

        else:
            column = (
                "State_partial_R2_given_air"
            )

        point_estimate = (
            float(
                mountain[column]
            )
            - float(
                plain[column]
            )
        )

        values_array = np.asarray(
            values,
            dtype=float,
        )

        contrast_rows.append(
            {
                "DEM_threshold_m":
                    threshold,

                "K":
                    k,

                "AR_order":
                    AR_ORDER,

                "soil_layer":
                    layer,

                "layer_label":
                    LAYER_LABELS[
                        layer
                    ],

                "depth":
                    LAYER_DEPTHS[
                        layer
                    ],

                "contrast":
                    contrast,

                "point_estimate":
                    point_estimate,

                "bootstrap_mean":
                    mean,

                "CI95_lower":
                    low,

                "CI95_upper":
                    high,

                "Probability_gt_0":
                    float(
                        np.mean(
                            values_array > 0
                        )
                    ),

                "Probability_lt_0":
                    float(
                        np.mean(
                            values_array < 0
                        )
                    ),

                "bootstrap_reps":
                    BOOTSTRAP_REPS,

                "block_length_months":
                    BLOCK_LENGTH,
            }
        )

    return (
        pd.DataFrame(
            region_rows
        ),
        pd.DataFrame(
            contrast_rows
        ),
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        grid,
        matrices,
        time_index,
    ) = load_inputs()

    region_series = (
        build_region_series(
            grid,
            matrices,
            time_index,
        )
    )

    point_table = (
        point_estimates(
            region_series,
            grid,
        )
    )

    (
        region_bootstrap,
        contrasts,
    ) = bootstrap_analysis(
        region_series,
        point_table,
    )

    point_table.to_csv(
        OUTPUT_DIR
        / "terrain_region_metrics.csv",
        index=False,
    )

    region_bootstrap.to_csv(
        OUTPUT_DIR
        / "terrain_region_bootstrap.csv",
        index=False,
    )

    contrasts.to_csv(
        OUTPUT_DIR
        / "terrain_mountain_plain_contrasts.csv",
        index=False,
    )

    primary = point_table.loc[
        (
            point_table[
                "DEM_threshold_m"
            ]
            == 300
        )
        & (
            point_table["K"]
            == 3
        )
    ]

    print(
        "\nDEM 300 m, K3-AR3"
    )

    print(
        primary[
            [
                "region",
                "layer_label",
                "Air_partial_R2_given_state",
                "State_partial_R2_given_air",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    primary_contrasts = (
        contrasts.loc[
            (
                contrasts[
                    "DEM_threshold_m"
                ]
                == 300
            )
            & (
                contrasts["K"]
                == 3
            )
        ]
    )

    print(
        "\nMountain - Plain contrasts"
    )

    print(
        primary_contrasts[
            [
                "layer_label",
                "contrast",
                "point_estimate",
                "CI95_lower",
                "CI95_upper",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()