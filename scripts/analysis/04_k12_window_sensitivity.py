"""
Sensitivity analysis for K=3, 6, and 12 month air-temperature windows.
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

K_VALUES = [3, 6, 12]
AR_VALUES = [1, 2, 3]

COMMON_LAG = 12
EXPECTED_MONTHS = 552
EXPECTED_COMMON_N = 540


def build_dataframe(
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


def model_sse(result):
    residuals = np.asarray(
        result.resid,
        dtype=float,
    )

    return float(
        np.sum(residuals ** 2)
    )


def partial_r2(
    reduced_sse,
    full_sse,
):
    if reduced_sse <= 0:
        return np.nan

    return float(
        (reduced_sse - full_sse)
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
        "air_model": air_model,
        "state_model": state_model,
        "full_model": full_model,
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


def residual_diagnostics(
    residuals,
):
    residuals = pd.Series(
        np.asarray(
            residuals,
            dtype=float,
        )
    ).dropna()

    lb = acorr_ljungbox(
        residuals,
        lags=[12, 24],
        return_df=True,
    )

    return {
        "Residual_ACF1":
            float(
                residuals.autocorr(
                    lag=1
                )
            ),
        "LB12_p":
            float(
                lb.loc[
                    12,
                    "lb_pvalue",
                ]
            ),
        "LB24_p":
            float(
                lb.loc[
                    24,
                    "lb_pvalue",
                ]
            ),
    }


def load_data():
    df = pd.read_csv(
        INPUT_FILE
    )

    if "date" not in df.columns:
        raise ValueError(
            "Input file has no 'date' column."
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
        "t2m_anom_deseason_detrend_C",
        *[
            f"{layer}_anom_deseason_detrend_C"
            for layer in LAYERS
        ],
    ]

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


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

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

    for layer in LAYERS:
        for k in K_VALUES:
            for ar_order in AR_VALUES:

                model_df, air_cols, state_cols = (
                    build_dataframe(
                        air,
                        soils[layer],
                        k,
                        ar_order,
                    )
                )

                model_df = model_df.loc[
                    common_start:
                ].copy()

                if len(model_df) != EXPECTED_COMMON_N:
                    raise ValueError(
                        f"{layer}, K{k}, AR{ar_order}: "
                        f"expected n={EXPECTED_COMMON_N}, "
                        f"found n={len(model_df)}."
                    )

                fitted = fit_models(
                    model_df,
                    air_cols,
                    state_cols,
                )

                full_model = fitted[
                    "full_model"
                ]

                diagnostics = (
                    residual_diagnostics(
                        full_model.resid
                    )
                )

                state_coefficients = [
                    float(
                        full_model.params[
                            f"soil_lag{lag}"
                        ]
                    )
                    for lag in range(
                        1,
                        ar_order + 1,
                    )
                ]

                rows.append(
                    {
                        "soil_layer":
                            layer,
                        "layer_label":
                            LAYER_LABELS[layer],
                        "K":
                            k,
                        "AR_order":
                            ar_order,
                        "n":
                            len(model_df),
                        "common_start":
                            common_start.strftime(
                                "%Y-%m"
                            ),
                        "R2_air_only":
                            float(
                                fitted[
                                    "air_model"
                                ].rsquared
                            ),
                        "R2_state_only":
                            float(
                                fitted[
                                    "state_model"
                                ].rsquared
                            ),
                        "R2_full":
                            float(
                                full_model.rsquared
                            ),
                        "Adjusted_R2_full":
                            float(
                                full_model.rsquared_adj
                            ),
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
                                full_model.aic
                            ),
                        "BIC_full":
                            float(
                                full_model.bic
                            ),
                        "Residual_ACF1":
                            diagnostics[
                                "Residual_ACF1"
                            ],
                        "LB12_p":
                            diagnostics[
                                "LB12_p"
                            ],
                        "LB24_p":
                            diagnostics[
                                "LB24_p"
                            ],
                        "AR_coefficient_sum":
                            float(
                                np.sum(
                                    state_coefficients
                                )
                            ),
                        "AR1_coefficient":
                            state_coefficients[0],
                        "AR2_coefficient":
                            (
                                state_coefficients[1]
                                if ar_order >= 2
                                else np.nan
                            ),
                        "AR3_coefficient":
                            (
                                state_coefficients[2]
                                if ar_order >= 3
                                else np.nan
                            ),
                    }
                )

    results = pd.DataFrame(
        rows
    )

    results[
        "Delta_BIC_within_layer"
    ] = (
        results
        .groupby(
            "soil_layer"
        )["BIC_full"]
        .transform(
            lambda x:
                x - x.min()
        )
    )

    results = results.sort_values(
        [
            "soil_layer",
            "K",
            "AR_order",
        ]
    )

    results.to_csv(
        OUTPUT_DIR
        / "model_sensitivity_K3_K6_K12.csv",
        index=False,
    )

    layer4_ar3 = results.loc[
        (
            results["soil_layer"]
            == "stl4"
        )
        & (
            results["AR_order"]
            == 3
        )
    ].copy()

    layer4_ar3.to_csv(
        OUTPUT_DIR
        / "layer4_K_window_sensitivity.csv",
        index=False,
    )

    print(
        "\nK3/K6/K12 sensitivity "
        "(common n=540)"
    )

    print(
        results[
            [
                "layer_label",
                "K",
                "AR_order",
                "Air_partial_R2_given_state",
                "State_partial_R2_given_air",
                "R2_full",
                "Delta_BIC_within_layer",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nLayer 4, AR3:"
    )

    print(
        layer4_ar3[
            [
                "K",
                "Air_partial_R2_given_state",
                "State_partial_R2_given_air",
                "R2_full",
                "BIC_full",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()