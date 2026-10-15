"""
ERA5-Land and FLDAS cross-product reproducibility analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[2]

ERA_FILE = (
    REPO_ROOT
    / "data"
    / "derived"
    / "era5land_beijing_monthly_1979_2024.csv"
)

FLDAS_FILE = (
    REPO_ROOT
    / "data"
    / "derived"
    / "fldas_beijing_monthly_1982_2024.csv"
)

OUTPUT_DIR = REPO_ROOT / "results" / "tables"

DATASETS = ["ERA5-Land", "FLDAS"]
LAYERS = ["L1", "L2", "L3", "L4"]

K_VALUES = [3, 6]
AR_ORDER = 3
COMMON_LAG = 6

COMMON_START = pd.Timestamp("1982-01-01")
COMMON_END = pd.Timestamp("2024-12-01")

BOOTSTRAP_REPS = 2000
BLOCK_LENGTH = 36
RANDOM_SEED = 20260814

LAG_RANGE = range(-6, 7)
ACF_MAX_LAG = 24
EFOLD_THRESHOLD = 1 / np.e

LAYER_DEPTHS = {
    "ERA5-Land": {
        "L1": "0-7 cm",
        "L2": "7-28 cm",
        "L3": "28-100 cm",
        "L4": "100-289 cm",
    },
    "FLDAS": {
        "L1": "0-10 cm",
        "L2": "10-40 cm",
        "L3": "40-100 cm",
        "L4": "100-200 cm",
    },
}

ERA_COLUMNS = {
    "air": "t2m_raw_C",
    "L1": "stl1_raw_C",
    "L2": "stl2_raw_C",
    "L3": "stl3_raw_C",
    "L4": "stl4_raw_C",
}

FLDAS_COLUMNS = {
    "air": "Tair_f_tavg_raw",
    "L1": "SoilTemp00_10cm_tavg_raw",
    "L2": "SoilTemp10_40cm_tavg_raw",
    "L3": "SoilTemp40_100cm_tavg_raw",
    "L4": "SoilTemp100_200cm_tavg_raw",
}


def prepare_common_period(series):
    series = series.loc[
        COMMON_START:COMMON_END
    ].astype(float)

    expected = pd.date_range(
        COMMON_START,
        COMMON_END,
        freq="MS",
    )

    series = series.reindex(
        expected
    )

    if series.isna().any():
        raise ValueError(
            "Missing months in common-period series."
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
        len(deseason),
        dtype=float,
    )

    values = deseason.to_numpy(
        dtype=float
    )

    slope, intercept = np.polyfit(
        x,
        values,
        1,
    )

    detrended = (
        values
        - (
            intercept
            + slope * x
        )
    )

    return pd.Series(
        detrended,
        index=expected,
    )


def calculate_lag_correlation(
    air,
    soil,
):
    rows = []

    for lag in LAG_RANGE:
        aligned = pd.concat(
            [
                air.shift(lag),
                soil,
            ],
            axis=1,
        ).dropna()

        aligned.columns = [
            "air",
            "soil",
        ]

        r = float(
            aligned["air"].corr(
                aligned["soil"]
            )
        )

        rows.append(
            {
                "lag": lag,
                "r": r,
            }
        )

    table = pd.DataFrame(
        rows
    )

    peak = table.loc[
        table["r"].idxmax()
    ]

    lag0 = table.loc[
        table["lag"] == 0,
        "r",
    ].iloc[0]

    return (
        float(peak["r"]),
        int(peak["lag"]),
        float(lag0),
    )


def calculate_acf(
    series,
    max_lag=24,
):
    values = np.asarray(
        series,
        dtype=float,
    )

    acf = [1.0]

    for lag in range(
        1,
        max_lag + 1,
    ):
        x = values[:-lag]
        y = values[lag:]

        acf.append(
            float(
                np.corrcoef(
                    x,
                    y,
                )[0, 1]
            )
        )

    return np.asarray(
        acf,
        dtype=float,
    )


def calculate_efolding(
    acf,
):
    for lag in range(
        1,
        len(acf),
    ):
        previous = acf[
            lag - 1
        ]

        current = acf[
            lag
        ]

        if current <= EFOLD_THRESHOLD:

            denominator = (
                previous - current
            )

            if denominator == 0:
                return float(lag)

            fraction = (
                previous
                - EFOLD_THRESHOLD
            ) / denominator

            return float(
                (lag - 1)
                + fraction
            )

    return np.nan


def build_dataframe(
    air,
    soil,
    k,
):
    df = pd.DataFrame(
        index=soil.index
    )

    df["soil"] = soil

    air_cols = []
    state_cols = []

    for lag in range(
        k + 1
    ):
        col = f"air_lag{lag}"
        df[col] = air.shift(lag)
        air_cols.append(col)

    for lag in range(
        1,
        AR_ORDER + 1,
    ):
        col = f"soil_lag{lag}"
        df[col] = soil.shift(lag)
        state_cols.append(col)

    return (
        df.dropna(),
        air_cols,
        state_cols,
    )


def fit_ols(
    df,
    predictors,
):
    y = df["soil"].astype(float)

    X = sm.add_constant(
        df[predictors].astype(float),
        has_constant="add",
    )

    return sm.OLS(
        y,
        X,
    ).fit()


def model_sse(
    result,
):
    residuals = np.asarray(
        result.resid,
        dtype=float,
    )

    return float(
        np.sum(
            residuals ** 2
        )
    )


def partial_r2(
    reduced_sse,
    full_sse,
):
    if reduced_sse <= 0:
        return np.nan

    return float(
        (
            reduced_sse
            - full_sse
        )
        / reduced_sse
    )


def fit_models(
    df,
    air_cols,
    state_cols,
):
    air_model = fit_ols(
        df,
        air_cols,
    )

    state_model = fit_ols(
        df,
        state_cols,
    )

    full_model = fit_ols(
        df,
        air_cols + state_cols,
    )

    sse_air = model_sse(
        air_model
    )

    sse_state = model_sse(
        state_model
    )

    sse_full = model_sse(
        full_model
    )

    return {
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
            partial_r2(
                sse_state,
                sse_full,
            ),

        "State_partial_R2_given_air":
            partial_r2(
                sse_air,
                sse_full,
            ),
    }


def circular_block_indices(
    n,
    block_length,
    rng,
):
    n_blocks = int(
        np.ceil(
            n / block_length
        )
    )

    blocks = []

    for _ in range(
        n_blocks
    ):
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


def ols_sse_numpy(
    y,
    X,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    X = np.asarray(
        X,
        dtype=float,
    )

    X = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )[0]

    residual = (
        y - X @ beta
    )

    return float(
        np.sum(
            residual ** 2
        )
    )


def percentile_ci(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    return (
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


def load_file(
    path,
    columns,
):
    df = pd.read_csv(
        path
    )

    if "date" not in df.columns:
        raise ValueError(
            f"{path.name} has no 'date' column."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df
        .sort_values("date")
        .set_index("date")
    )

    missing = [
        col
        for col in columns.values()
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} missing columns: "
            f"{missing}"
        )

    return df


def load_datasets():
    era = load_file(
        ERA_FILE,
        ERA_COLUMNS,
    )

    fldas = load_file(
        FLDAS_FILE,
        FLDAS_COLUMNS,
    )

    series = {
        "ERA5-Land": {},
        "FLDAS": {},
    }

    for variable, column in (
        ERA_COLUMNS.items()
    ):
        series[
            "ERA5-Land"
        ][variable] = (
            prepare_common_period(
                era[column]
            )
        )

    for variable, column in (
        FLDAS_COLUMNS.items()
    ):
        series[
            "FLDAS"
        ][variable] = (
            prepare_common_period(
                fldas[column]
            )
        )

    return series


def temporal_characteristics(
    series,
):
    rows = []

    for dataset in DATASETS:
        air = series[
            dataset
        ]["air"]

        for layer in LAYERS:
            soil = series[
                dataset
            ][layer]

            (
                maximum_r,
                peak_lag,
                r0,
            ) = calculate_lag_correlation(
                air,
                soil,
            )

            acf = calculate_acf(
                soil,
                ACF_MAX_LAG,
            )

            rows.append(
                {
                    "dataset":
                        dataset,

                    "layer":
                        layer,

                    "depth":
                        LAYER_DEPTHS[
                            dataset
                        ][layer],

                    "Maximum_r":
                        maximum_r,

                    "Peak_lag":
                        peak_lag,

                    "R_at_lag0":
                        r0,

                    "Lag1_ACF":
                        float(
                            acf[1]
                        ),

                    "Efolding_months":
                        calculate_efolding(
                            acf
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def conditional_analysis(
    series,
):
    common_model_start = (
        pd.date_range(
            COMMON_START,
            COMMON_END,
            freq="MS",
        )[COMMON_LAG]
    )

    rows = []
    cache = {}

    for dataset in DATASETS:

        air = series[
            dataset
        ]["air"]

        for layer in LAYERS:

            soil = series[
                dataset
            ][layer]

            for k in K_VALUES:

                (
                    model_df,
                    air_cols,
                    state_cols,
                ) = build_dataframe(
                    air,
                    soil,
                    k,
                )

                model_df = (
                    model_df
                    .loc[
                        common_model_start:
                    ]
                    .copy()
                )

                if len(model_df) != 510:
                    raise ValueError(
                        f"{dataset}, {layer}, K{k}: "
                        f"expected n=510, "
                        f"found {len(model_df)}."
                    )

                result = fit_models(
                    model_df,
                    air_cols,
                    state_cols,
                )

                rows.append(
                    {
                        "dataset":
                            dataset,

                        "layer":
                            layer,

                        "depth":
                            LAYER_DEPTHS[
                                dataset
                            ][layer],

                        "K":
                            k,

                        "AR_order":
                            AR_ORDER,

                        "n":
                            len(model_df),

                        **result,
                    }
                )

                cache[
                    (
                        dataset,
                        layer,
                        k,
                    )
                ] = {
                    "df":
                        model_df,
                    "air_cols":
                        air_cols,
                    "state_cols":
                        state_cols,
                }

    return (
        pd.DataFrame(rows),
        cache,
    )


def bootstrap_analysis(
    decomposition,
    cache,
):
    common_index = None

    for item in cache.values():

        index = item[
            "df"
        ].index

        common_index = (
            index
            if common_index is None
            else common_index.intersection(
                index
            )
        )

    n = len(
        common_index
    )

    if n != 510:
        raise ValueError(
            f"Expected bootstrap n=510, "
            f"found {n}."
        )

    air_boot = {
        key:
            np.full(
                BOOTSTRAP_REPS,
                np.nan,
            )
        for key in cache
    }

    state_boot = {
        key:
            np.full(
                BOOTSTRAP_REPS,
                np.nan,
            )
        for key in cache
    }

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    for b in range(
        BOOTSTRAP_REPS
    ):
        idx = circular_block_indices(
            n,
            BLOCK_LENGTH,
            rng,
        )

        for key, item in (
            cache.items()
        ):
            model_df = (
                item["df"]
                .loc[
                    common_index
                ]
            )

            air_cols = item[
                "air_cols"
            ]

            state_cols = item[
                "state_cols"
            ]

            y = model_df[
                "soil"
            ].to_numpy(
                dtype=float
            )[idx]

            X_air = model_df[
                air_cols
            ].to_numpy(
                dtype=float
            )[idx]

            X_state = model_df[
                state_cols
            ].to_numpy(
                dtype=float
            )[idx]

            X_full = model_df[
                air_cols
                + state_cols
            ].to_numpy(
                dtype=float
            )[idx]

            sse_air = ols_sse_numpy(
                y,
                X_air,
            )

            sse_state = ols_sse_numpy(
                y,
                X_state,
            )

            sse_full = ols_sse_numpy(
                y,
                X_full,
            )

            air_boot[key][b] = (
                partial_r2(
                    sse_state,
                    sse_full,
                )
            )

            state_boot[key][b] = (
                partial_r2(
                    sse_air,
                    sse_full,
                )
            )

    r2_rows = []

    for (
        dataset,
        layer,
        k,
    ) in cache:

        subset = decomposition.loc[
            (
                decomposition["dataset"]
                == dataset
            )
            & (
                decomposition["layer"]
                == layer
            )
            & (
                decomposition["K"]
                == k
            )
        ].iloc[0]

        for metric, values in [
            (
                "Air_partial_R2_given_state",
                air_boot[
                    (
                        dataset,
                        layer,
                        k,
                    )
                ],
            ),
            (
                "State_partial_R2_given_air",
                state_boot[
                    (
                        dataset,
                        layer,
                        k,
                    )
                ],
            ),
        ]:

            ci_low, ci_high = (
                percentile_ci(
                    values
                )
            )

            r2_rows.append(
                {
                    "dataset":
                        dataset,

                    "layer":
                        layer,

                    "depth":
                        LAYER_DEPTHS[
                            dataset
                        ][layer],

                    "K":
                        k,

                    "AR_order":
                        AR_ORDER,

                    "metric":
                        metric,

                    "point_estimate":
                        float(
                            subset[
                                metric
                            ]
                        ),

                    "bootstrap_mean":
                        float(
                            np.mean(
                                values
                            )
                        ),

                    "CI95_lower":
                        ci_low,

                    "CI95_upper":
                        ci_high,

                    "bootstrap_reps":
                        BOOTSTRAP_REPS,

                    "block_length_months":
                        BLOCK_LENGTH,
                }
            )

    contrast_rows = []

    for dataset in DATASETS:
        for k in K_VALUES:

            state_l4_l1 = (
                state_boot[
                    (
                        dataset,
                        "L4",
                        k,
                    )
                ]
                - state_boot[
                    (
                        dataset,
                        "L1",
                        k,
                    )
                ]
            )

            state_l4_l3 = (
                state_boot[
                    (
                        dataset,
                        "L4",
                        k,
                    )
                ]
                - state_boot[
                    (
                        dataset,
                        "L3",
                        k,
                    )
                ]
            )

            air_l1_l4 = (
                air_boot[
                    (
                        dataset,
                        "L1",
                        k,
                    )
                ]
                - air_boot[
                    (
                        dataset,
                        "L4",
                        k,
                    )
                ]
            )

            dataset_table = (
                decomposition.loc[
                    (
                        decomposition[
                            "dataset"
                        ]
                        == dataset
                    )
                    & (
                        decomposition[
                            "K"
                        ]
                        == k
                    )
                ]
                .set_index(
                    "layer"
                )
            )

            contrasts = [
                (
                    "State_L4_minus_L1",
                    state_l4_l1,
                    (
                        dataset_table.loc[
                            "L4",
                            "State_partial_R2_given_air",
                        ]
                        - dataset_table.loc[
                            "L1",
                            "State_partial_R2_given_air",
                        ]
                    ),
                ),
                (
                    "State_L4_minus_L3",
                    state_l4_l3,
                    (
                        dataset_table.loc[
                            "L4",
                            "State_partial_R2_given_air",
                        ]
                        - dataset_table.loc[
                            "L3",
                            "State_partial_R2_given_air",
                        ]
                    ),
                ),
                (
                    "Air_L1_minus_L4",
                    air_l1_l4,
                    (
                        dataset_table.loc[
                            "L1",
                            "Air_partial_R2_given_state",
                        ]
                        - dataset_table.loc[
                            "L4",
                            "Air_partial_R2_given_state",
                        ]
                    ),
                ),
            ]

            for (
                name,
                values,
                point,
            ) in contrasts:

                ci_low, ci_high = (
                    percentile_ci(
                        values
                    )
                )

                contrast_rows.append(
                    {
                        "dataset":
                            dataset,

                        "K":
                            k,

                        "AR_order":
                            AR_ORDER,

                        "contrast":
                            name,

                        "point_estimate":
                            float(point),

                        "bootstrap_mean":
                            float(
                                np.mean(
                                    values
                                )
                            ),

                        "CI95_lower":
                            ci_low,

                        "CI95_upper":
                            ci_high,

                        "Probability_gt_0":
                            float(
                                np.mean(
                                    values > 0
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
            r2_rows
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

    series = load_datasets()

    temporal = (
        temporal_characteristics(
            series
        )
    )

    (
        decomposition,
        cache,
    ) = conditional_analysis(
        series
    )

    (
        bootstrap_r2,
        contrasts,
    ) = bootstrap_analysis(
        decomposition,
        cache,
    )

    temporal.to_csv(
        OUTPUT_DIR
        / "cross_product_temporal_characteristics.csv",
        index=False,
    )

    decomposition.to_csv(
        OUTPUT_DIR
        / "cross_product_conditional_contributions.csv",
        index=False,
    )

    bootstrap_r2.to_csv(
        OUTPUT_DIR
        / "cross_product_bootstrap_partial_r2.csv",
        index=False,
    )

    contrasts.to_csv(
        OUTPUT_DIR
        / "cross_product_vertical_contrasts.csv",
        index=False,
    )

    print(
        "\nCross-product temporal characteristics"
    )

    print(
        temporal[
            [
                "dataset",
                "layer",
                "depth",
                "Maximum_r",
                "Peak_lag",
                "Lag1_ACF",
                "Efolding_months",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nCross-product conditional contributions: "
        "K3-AR3"
    )

    print(
        decomposition.loc[
            decomposition["K"] == 3,
            [
                "dataset",
                "layer",
                "depth",
                "Air_partial_R2_given_state",
                "State_partial_R2_given_air",
            ],
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nVertical contrasts"
    )

    print(
        contrasts[
            [
                "dataset",
                "K",
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