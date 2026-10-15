"""
Conditional contribution analysis and robustness tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.diagnostic import acorr_ljungbox


REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    REPO_ROOT
    / "data"
    / "derived"
    / "era5land_beijing_monthly_1979_2024.csv"
)

OUTPUT_DIR = REPO_ROOT / "results" / "tables"

LAYERS = ["stl1", "stl2", "stl3", "stl4"]

LAYER_LABELS = {
    "stl1": "Layer 1",
    "stl2": "Layer 2",
    "stl3": "Layer 3",
    "stl4": "Layer 4",
}

K_VALUES = [3, 6]
AR_VALUES = [1, 2, 3]

PRIMARY_K = 3
PRIMARY_AR = 3
COMMON_LAG = 6

BOOTSTRAP_REPS = 2000
BLOCK_LENGTH = 36
RANDOM_SEED = 20260814

EXPECTED_MONTHS = 552

CV_FOLDS = [
    (
        "Fold 1",
        "1979-01-01",
        "2004-12-01",
        "2005-01-01",
        "2008-12-01",
    ),
    (
        "Fold 2",
        "1979-01-01",
        "2008-12-01",
        "2009-01-01",
        "2012-12-01",
    ),
    (
        "Fold 3",
        "1979-01-01",
        "2012-12-01",
        "2013-01-01",
        "2016-12-01",
    ),
    (
        "Fold 4",
        "1979-01-01",
        "2016-12-01",
        "2017-01-01",
        "2020-12-01",
    ),
    (
        "Fold 5",
        "1979-01-01",
        "2020-12-01",
        "2021-01-01",
        "2024-12-01",
    ),
]


def build_model_dataframe(
    air,
    soil,
    k,
    ar_order,
):
    df = pd.DataFrame(index=soil.index)
    df["soil"] = soil

    air_cols = []
    state_cols = []

    for lag in range(k + 1):
        col = f"air_lag{lag}"
        df[col] = air.shift(lag)
        air_cols.append(col)

    for lag in range(1, ar_order + 1):
        col = f"soil_lag{lag}"
        df[col] = soil.shift(lag)
        state_cols.append(col)

    return (
        df.dropna(),
        air_cols,
        state_cols,
    )


def fit_ols(df, predictors):
    y = df["soil"].astype(float)

    X = sm.add_constant(
        df[predictors].astype(float),
        has_constant="add",
    )

    return sm.OLS(
        y,
        X,
    ).fit()


def sse(result):
    residuals = np.asarray(
        result.resid,
        dtype=float,
    )

    return float(
        np.sum(residuals ** 2)
    )


def conditional_partial_r2(
    reduced_sse,
    full_sse,
):
    if reduced_sse <= 0:
        return np.nan

    return float(
        (reduced_sse - full_sse)
        / reduced_sse
    )


def fit_three_models(
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

    sse_air = sse(air_model)
    sse_state = sse(state_model)
    sse_full = sse(full_model)

    return {
        "air_model": air_model,
        "state_model": state_model,
        "full_model": full_model,
        "R2_air_only":
            float(air_model.rsquared),
        "R2_state_only":
            float(state_model.rsquared),
        "R2_full":
            float(full_model.rsquared),
        "Air_partial_R2_given_state":
            conditional_partial_r2(
                sse_state,
                sse_full,
            ),
        "State_partial_R2_given_air":
            conditional_partial_r2(
                sse_air,
                sse_full,
            ),
    }


def residual_diagnostics(
    residuals,
):
    residuals = pd.Series(
        np.asarray(
            residuals,
            dtype=float,
        )
    ).dropna()

    acf1 = float(
        residuals.autocorr(lag=1)
    )

    lb = acorr_ljungbox(
        residuals,
        lags=[12, 24],
        return_df=True,
    )

    return {
        "Residual_ACF1": acf1,
        "LB12_p":
            float(lb.loc[12, "lb_pvalue"]),
        "LB24_p":
            float(lb.loc[24, "lb_pvalue"]),
    }


def train_based_transform(
    train_raw,
    full_raw,
):
    climatology = (
        train_raw
        .groupby(train_raw.index.month)
        .mean()
    )

    train_clim = pd.Series(
        train_raw.index.month,
        index=train_raw.index,
    ).map(climatology)

    full_clim = pd.Series(
        full_raw.index.month,
        index=full_raw.index,
    ).map(climatology)

    train_deseason = (
        train_raw - train_clim
    )

    full_deseason = (
        full_raw - full_clim
    )

    origin = full_raw.index.min()

    full_x = np.asarray(
        [
            (date.year - origin.year) * 12
            + date.month
            - origin.month
            for date in full_raw.index
        ],
        dtype=float,
    )

    positions = full_raw.index.get_indexer(
        train_raw.index
    )

    train_x = full_x[positions]
    train_y = train_deseason.to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(train_x)
        & np.isfinite(train_y)
    )

    slope, intercept = np.polyfit(
        train_x[valid],
        train_y[valid],
        1,
    )

    transformed = (
        full_deseason.to_numpy(dtype=float)
        - (
            intercept
            + slope * full_x
        )
    )

    return pd.Series(
        transformed,
        index=full_raw.index,
    )


def predict(
    result,
    df,
    predictors,
):
    X = sm.add_constant(
        df[predictors].astype(float),
        has_constant="add",
    )

    X = X.reindex(
        columns=result.model.exog_names
    )

    return np.asarray(
        result.predict(X),
        dtype=float,
    )


def rmse(
    observed,
    predicted,
):
    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(observed)
                    - np.asarray(predicted)
                ) ** 2
            )
        )
    )


def mae(
    observed,
    predicted,
):
    return float(
        np.mean(
            np.abs(
                np.asarray(observed)
                - np.asarray(predicted)
            )
        )
    )


def test_r2(
    observed,
    predicted,
):
    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    error = np.sum(
        (observed - predicted) ** 2
    )

    total = np.sum(
        (
            observed
            - observed.mean()
        ) ** 2
    )

    if total <= 0:
        return np.nan

    return float(
        1 - error / total
    )


def circular_block_indices(
    n,
    block_length,
    rng,
):
    n_blocks = int(
        np.ceil(n / block_length)
    )

    blocks = []

    for _ in range(n_blocks):
        start = int(
            rng.integers(0, n)
        )

        blocks.append(
            (
                start
                + np.arange(block_length)
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

    residuals = (
        y - X @ beta
    )

    return float(
        np.sum(residuals ** 2)
    )


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
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def load_data():
    df = pd.read_csv(
        INPUT_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df
        .sort_values("date")
        .set_index("date")
    )

    required = [
        "t2m_raw_C",
        "t2m_anom_deseason_detrend_C",
    ]

    for layer in LAYERS:
        required.extend(
            [
                f"{layer}_raw_C",
                f"{layer}_anom_deseason_detrend_C",
            ]
        )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if len(df) != EXPECTED_MONTHS:
        raise ValueError(
            f"Expected {EXPECTED_MONTHS} months, "
            f"found {len(df)}."
        )

    return df


def specification_analysis(
    df,
):
    air = df[
        "t2m_anom_deseason_detrend_C"
    ]

    soils = {
        layer: df[
            f"{layer}_anom_deseason_detrend_C"
        ]
        for layer in LAYERS
    }

    common_start = df.index[
        COMMON_LAG
    ]

    rows = []
    cache = {}

    for layer in LAYERS:
        for k in K_VALUES:
            for ar_order in AR_VALUES:

                model_df, air_cols, state_cols = (
                    build_model_dataframe(
                        air,
                        soils[layer],
                        k,
                        ar_order,
                    )
                )

                model_df = model_df.loc[
                    common_start:
                ].copy()

                fitted = fit_three_models(
                    model_df,
                    air_cols,
                    state_cols,
                )

                diag = residual_diagnostics(
                    fitted[
                        "full_model"
                    ].resid
                )

                rows.append(
                    {
                        "soil_layer": layer,
                        "layer_label":
                            LAYER_LABELS[layer],
                        "K": k,
                        "AR_order": ar_order,
                        "n":
                            len(model_df),
                        "R2_air_only":
                            fitted[
                                "R2_air_only"
                            ],
                        "R2_state_only":
                            fitted[
                                "R2_state_only"
                            ],
                        "R2_full":
                            fitted[
                                "R2_full"
                            ],
                        "Air_partial_R2_given_state":
                            fitted[
                                "Air_partial_R2_given_state"
                            ],
                        "State_partial_R2_given_air":
                            fitted[
                                "State_partial_R2_given_air"
                            ],
                        "AIC_full":
                            float(
                                fitted[
                                    "full_model"
                                ].aic
                            ),
                        "BIC_full":
                            float(
                                fitted[
                                    "full_model"
                                ].bic
                            ),
                        **diag,
                    }
                )

                cache[
                    (
                        layer,
                        k,
                        ar_order,
                    )
                ] = {
                    "df": model_df,
                    "air_cols": air_cols,
                    "state_cols": state_cols,
                    **fitted,
                }

    return pd.DataFrame(
        rows
    ), cache


def cross_validation(
    df,
):
    rows = []

    variables = [
        "t2m",
        *LAYERS,
    ]

    for (
        fold,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in CV_FOLDS:

        train_start = pd.Timestamp(
            train_start
        )

        train_end = pd.Timestamp(
            train_end
        )

        test_start = pd.Timestamp(
            test_start
        )

        test_end = pd.Timestamp(
            test_end
        )

        fold_full = df.loc[
            train_start:test_end
        ]

        fold_train = df.loc[
            train_start:train_end
        ]

        transformed = {}

        for variable in variables:
            raw_col = (
                "t2m_raw_C"
                if variable == "t2m"
                else f"{variable}_raw_C"
            )

            transformed[variable] = (
                train_based_transform(
                    fold_train[raw_col],
                    fold_full[raw_col],
                )
            )

        for layer in LAYERS:
            for k in K_VALUES:
                for ar_order in AR_VALUES:

                    model_df, air_cols, state_cols = (
                        build_model_dataframe(
                            transformed["t2m"],
                            transformed[layer],
                            k,
                            ar_order,
                        )
                    )

                    common_start = (
                        train_start
                        + pd.DateOffset(
                            months=COMMON_LAG
                        )
                    )

                    train_df = model_df.loc[
                        common_start:train_end
                    ]

                    test_df = model_df.loc[
                        test_start:test_end
                    ]

                    fitted = fit_three_models(
                        train_df,
                        air_cols,
                        state_cols,
                    )

                    y_test = test_df[
                        "soil"
                    ].to_numpy(
                        dtype=float
                    )

                    pred_air = predict(
                        fitted["air_model"],
                        test_df,
                        air_cols,
                    )

                    pred_state = predict(
                        fitted["state_model"],
                        test_df,
                        state_cols,
                    )

                    pred_full = predict(
                        fitted["full_model"],
                        test_df,
                        air_cols
                        + state_cols,
                    )

                    rmse_air = rmse(
                        y_test,
                        pred_air,
                    )

                    rmse_state = rmse(
                        y_test,
                        pred_state,
                    )

                    rmse_full = rmse(
                        y_test,
                        pred_full,
                    )

                    rows.append(
                        {
                            "fold": fold,
                            "soil_layer": layer,
                            "layer_label":
                                LAYER_LABELS[layer],
                            "K": k,
                            "AR_order":
                                ar_order,
                            "n_train":
                                len(train_df),
                            "n_test":
                                len(test_df),
                            "RMSE_air_only":
                                rmse_air,
                            "RMSE_state_only":
                                rmse_state,
                            "RMSE_full":
                                rmse_full,
                            "MAE_air_only":
                                mae(
                                    y_test,
                                    pred_air,
                                ),
                            "MAE_state_only":
                                mae(
                                    y_test,
                                    pred_state,
                                ),
                            "MAE_full":
                                mae(
                                    y_test,
                                    pred_full,
                                ),
                            "Test_R2_air_only":
                                test_r2(
                                    y_test,
                                    pred_air,
                                ),
                            "Test_R2_state_only":
                                test_r2(
                                    y_test,
                                    pred_state,
                                ),
                            "Test_R2_full":
                                test_r2(
                                    y_test,
                                    pred_full,
                                ),
                            "State_skill_gain_vs_air_only":
                                (
                                    rmse_air
                                    - rmse_full
                                )
                                / rmse_air,
                            "Air_skill_gain_vs_state_only":
                                (
                                    rmse_state
                                    - rmse_full
                                )
                                / rmse_state,
                            "Full_better_than_air_only":
                                rmse_full
                                < rmse_air,
                            "Full_better_than_state_only":
                                rmse_full
                                < rmse_state,
                        }
                    )

    folds = pd.DataFrame(
        rows
    )

    summary = (
        folds.groupby(
            [
                "soil_layer",
                "layer_label",
                "K",
                "AR_order",
            ],
            as_index=False,
        )
        .agg(
            Mean_RMSE_air_only=(
                "RMSE_air_only",
                "mean",
            ),
            Mean_RMSE_state_only=(
                "RMSE_state_only",
                "mean",
            ),
            Mean_RMSE_full=(
                "RMSE_full",
                "mean",
            ),
            Mean_State_skill_gain=(
                "State_skill_gain_vs_air_only",
                "mean",
            ),
            Mean_Air_skill_gain=(
                "Air_skill_gain_vs_state_only",
                "mean",
            ),
            Folds_full_better_than_air=(
                "Full_better_than_air_only",
                "sum",
            ),
            Folds_full_better_than_state=(
                "Full_better_than_state_only",
                "sum",
            ),
            Mean_Test_R2_full=(
                "Test_R2_full",
                "mean",
            ),
        )
    )

    return folds, summary


def bootstrap_primary(
    cache,
):
    air_boot = {
        layer: np.full(
            BOOTSTRAP_REPS,
            np.nan,
        )
        for layer in LAYERS
    }

    state_boot = {
        layer: np.full(
            BOOTSTRAP_REPS,
            np.nan,
        )
        for layer in LAYERS
    }

    primary = {
        layer: cache[
            (
                layer,
                PRIMARY_K,
                PRIMARY_AR,
            )
        ]
        for layer in LAYERS
    }

    common_index = None

    for layer in LAYERS:
        idx = primary[
            layer
        ]["df"].index

        common_index = (
            idx
            if common_index is None
            else common_index.intersection(idx)
        )

    n = len(common_index)

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

        for layer in LAYERS:
            info = primary[layer]

            model_df = (
                info["df"]
                .loc[common_index]
            )

            air_cols = info[
                "air_cols"
            ]

            state_cols = info[
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

            air_boot[layer][b] = (
                conditional_partial_r2(
                    sse_state,
                    sse_full,
                )
            )

            state_boot[layer][b] = (
                conditional_partial_r2(
                    sse_air,
                    sse_full,
                )
            )

    rows = []

    for layer in LAYERS:
        info = primary[
            layer
        ]

        for metric, values, point in [
            (
                "Air_partial_R2_given_state",
                air_boot[layer],
                info[
                    "Air_partial_R2_given_state"
                ],
            ),
            (
                "State_partial_R2_given_air",
                state_boot[layer],
                info[
                    "State_partial_R2_given_air"
                ],
            ),
        ]:

            ci_low, ci_high = (
                percentile_ci(values)
            )

            rows.append(
                {
                    "soil_layer": layer,
                    "layer_label":
                        LAYER_LABELS[layer],
                    "metric": metric,
                    "point_estimate":
                        point,
                    "bootstrap_mean":
                        float(
                            np.mean(values)
                        ),
                    "CI95_lower":
                        ci_low,
                    "CI95_upper":
                        ci_high,
                    "K":
                        PRIMARY_K,
                    "AR_order":
                        PRIMARY_AR,
                    "n":
                        n,
                    "bootstrap_reps":
                        BOOTSTRAP_REPS,
                    "block_length_months":
                        BLOCK_LENGTH,
                }
            )

    contrasts = {
        "State_L4_minus_L1":
            state_boot["stl4"]
            - state_boot["stl1"],

        "State_L4_minus_L3":
            state_boot["stl4"]
            - state_boot["stl3"],

        "Air_L1_minus_L4":
            air_boot["stl1"]
            - air_boot["stl4"],
    }

    contrast_rows = []

    for name, values in contrasts.items():

        if name == "State_L4_minus_L1":
            point = (
                primary["stl4"][
                    "State_partial_R2_given_air"
                ]
                - primary["stl1"][
                    "State_partial_R2_given_air"
                ]
            )

        elif name == "State_L4_minus_L3":
            point = (
                primary["stl4"][
                    "State_partial_R2_given_air"
                ]
                - primary["stl3"][
                    "State_partial_R2_given_air"
                ]
            )

        else:
            point = (
                primary["stl1"][
                    "Air_partial_R2_given_state"
                ]
                - primary["stl4"][
                    "Air_partial_R2_given_state"
                ]
            )

        ci_low, ci_high = (
            percentile_ci(values)
        )

        contrast_rows.append(
            {
                "contrast": name,
                "point_estimate":
                    point,
                "bootstrap_mean":
                    float(
                        np.mean(values)
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
                "K": PRIMARY_K,
                "AR_order":
                    PRIMARY_AR,
            }
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            contrast_rows
        ),
    )


def save_primary_residuals(
    cache,
):
    output = None

    for layer in LAYERS:
        info = cache[
            (
                layer,
                PRIMARY_K,
                PRIMARY_AR,
            )
        ]

        model_df = info["df"]

        residual = pd.Series(
            np.asarray(
                info[
                    "full_model"
                ].resid,
                dtype=float,
            ),
            index=model_df.index,
            name=f"{layer}_residual",
        )

        if output is None:
            output = pd.DataFrame(
                index=model_df.index
            )

        output[
            f"{layer}_residual"
        ] = residual

    output.index.name = "date"

    return (
        output
        .reset_index()
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    sensitivity, cache = (
        specification_analysis(df)
    )

    primary = sensitivity.loc[
        (
            sensitivity["K"]
            == PRIMARY_K
        )
        & (
            sensitivity[
                "AR_order"
            ]
            == PRIMARY_AR
        )
    ].copy()

    cv_folds, cv_summary = (
        cross_validation(df)
    )

    bootstrap_r2, contrasts = (
        bootstrap_primary(cache)
    )

    residuals = (
        save_primary_residuals(
            cache
        )
    )

    sensitivity.to_csv(
        OUTPUT_DIR
        / "model_specification_K3_K6.csv",
        index=False,
    )

    primary.to_csv(
        OUTPUT_DIR
        / "primary_K3_AR3_summary.csv",
        index=False,
    )

    cv_folds.to_csv(
        OUTPUT_DIR
        / "time_blocked_cv_folds.csv",
        index=False,
    )

    cv_summary.to_csv(
        OUTPUT_DIR
        / "time_blocked_cv_summary.csv",
        index=False,
    )

    bootstrap_r2.to_csv(
        OUTPUT_DIR
        / "primary_bootstrap_partial_r2.csv",
        index=False,
    )

    contrasts.to_csv(
        OUTPUT_DIR
        / "primary_bootstrap_vertical_contrasts.csv",
        index=False,
    )

    residuals.to_csv(
        OUTPUT_DIR
        / "primary_full_model_residuals.csv",
        index=False,
    )

    print("\nPrimary K3-AR3 results")
    print(
        primary[
            [
                "layer_label",
                "Air_partial_R2_given_state",
                "State_partial_R2_given_air",
                "R2_full",
                "Residual_ACF1",
                "LB12_p",
                "LB24_p",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    primary_cv = cv_summary.loc[
        (
            cv_summary["K"]
            == PRIMARY_K
        )
        & (
            cv_summary[
                "AR_order"
            ]
            == PRIMARY_AR
        )
    ]

    print("\nFive-fold CV: K3-AR3")
    print(
        primary_cv[
            [
                "layer_label",
                "Mean_State_skill_gain",
                "Mean_Air_skill_gain",
                "Mean_RMSE_full",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nSaved results to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()