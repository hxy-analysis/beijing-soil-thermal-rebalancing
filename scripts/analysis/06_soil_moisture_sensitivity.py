"""
Same-layer antecedent soil-moisture sensitivity analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


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

SOIL_MOISTURE = {
    "stl1": "swvl1",
    "stl2": "swvl2",
    "stl3": "swvl3",
    "stl4": "swvl4",
}

K_VALUES = [3, 6]

ST_ORDER = 3
SM_ORDER = 3
COMMON_LAG = 6

EXPECTED_MONTHS = 552
EXPECTED_COMMON_N = 546


def build_dataframe(
    air,
    soil,
    moisture,
    k,
):
    df = pd.DataFrame(index=soil.index)
    df["soil"] = soil

    air_cols = []
    state_cols = []
    moisture_cols = []

    for lag in range(k + 1):
        col = f"air_lag{lag}"
        df[col] = air.shift(lag)
        air_cols.append(col)

    for lag in range(1, ST_ORDER + 1):
        col = f"soil_lag{lag}"
        df[col] = soil.shift(lag)
        state_cols.append(col)

    for lag in range(1, SM_ORDER + 1):
        col = f"moisture_lag{lag}"
        df[col] = moisture.shift(lag)
        moisture_cols.append(col)

    return (
        df.dropna(),
        air_cols,
        state_cols,
        moisture_cols,
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


def fit_three_block_model(
    df,
    air_cols,
    state_cols,
    moisture_cols,
):
    air_only = fit_ols(
        df,
        air_cols,
    )

    air_state = fit_ols(
        df,
        air_cols + state_cols,
    )

    state_moisture = fit_ols(
        df,
        state_cols + moisture_cols,
    )

    air_moisture = fit_ols(
        df,
        air_cols + moisture_cols,
    )

    full = fit_ols(
        df,
        air_cols
        + state_cols
        + moisture_cols,
    )

    sse_air = model_sse(
        air_only
    )

    sse_air_state = model_sse(
        air_state
    )

    sse_state_moisture = model_sse(
        state_moisture
    )

    sse_air_moisture = model_sse(
        air_moisture
    )

    sse_full = model_sse(
        full
    )

    state_without_moisture = (
        partial_r2(
            sse_air,
            sse_air_state,
        )
    )

    state_with_moisture = (
        partial_r2(
            sse_air_moisture,
            sse_full,
        )
    )

    return {
        "R2_full":
            float(full.rsquared),

        "Air_partial_R2_given_state_moisture":
            partial_r2(
                sse_state_moisture,
                sse_full,
            ),

        "State_partial_R2_given_air_moisture":
            state_with_moisture,

        "Moisture_partial_R2_given_air_state":
            partial_r2(
                sse_air_state,
                sse_full,
            ),

        "State_partial_R2_without_moisture":
            state_without_moisture,

        "State_partial_R2_absolute_change":
            (
                state_with_moisture
                - state_without_moisture
            ),

        "State_partial_R2_absolute_reduction":
            (
                state_without_moisture
                - state_with_moisture
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
    ]

    for layer in LAYERS:
        required.append(
            f"{layer}_anom_deseason_detrend_C"
        )

        required.append(
            f"{SOIL_MOISTURE[layer]}"
            "_anom_deseason_detrend"
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


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    air = df[
        "t2m_anom_deseason_detrend_C"
    ]

    common_start = df.index[
        COMMON_LAG
    ]

    rows = []

    for layer in LAYERS:

        soil = df[
            f"{layer}_anom_deseason_detrend_C"
        ]

        sm_layer = SOIL_MOISTURE[
            layer
        ]

        moisture = df[
            f"{sm_layer}_anom_deseason_detrend"
        ]

        for k in K_VALUES:

            (
                model_df,
                air_cols,
                state_cols,
                moisture_cols,
            ) = build_dataframe(
                air,
                soil,
                moisture,
                k,
            )

            model_df = model_df.loc[
                common_start:
            ].copy()

            if len(model_df) != EXPECTED_COMMON_N:
                raise ValueError(
                    f"{layer}, K{k}: "
                    f"expected n={EXPECTED_COMMON_N}, "
                    f"found n={len(model_df)}."
                )

            result = fit_three_block_model(
                model_df,
                air_cols,
                state_cols,
                moisture_cols,
            )

            before = result[
                "State_partial_R2_without_moisture"
            ]

            after = result[
                "State_partial_R2_given_air_moisture"
            ]

            relative_reduction = (
                (
                    before - after
                )
                / before
                if before != 0
                else np.nan
            )

            rows.append(
                {
                    "soil_layer":
                        layer,

                    "layer_label":
                        LAYER_LABELS[layer],

                    "soil_moisture_layer":
                        sm_layer,

                    "K":
                        k,

                    "ST_order":
                        ST_ORDER,

                    "SM_order":
                        SM_ORDER,

                    "n":
                        len(model_df),

                    "R2_full":
                        result[
                            "R2_full"
                        ],

                    "Air_partial_R2_given_state_moisture":
                        result[
                            "Air_partial_R2_given_state_moisture"
                        ],

                    "State_partial_R2_without_moisture":
                        before,

                    "State_partial_R2_given_air_moisture":
                        after,

                    "Moisture_partial_R2_given_air_state":
                        result[
                            "Moisture_partial_R2_given_air_state"
                        ],

                    "State_partial_R2_absolute_reduction":
                        result[
                            "State_partial_R2_absolute_reduction"
                        ],

                    "State_partial_R2_relative_reduction":
                        relative_reduction,
                }
            )

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        [
            "soil_layer",
            "K",
        ]
    )

    results.to_csv(
        OUTPUT_DIR
        / "soil_moisture_sensitivity.csv",
        index=False,
    )

    primary = results.loc[
        results["K"] == 3
    ].copy()

    primary.to_csv(
        OUTPUT_DIR
        / "soil_moisture_sensitivity_K3.csv",
        index=False,
    )

    print(
        "\nSame-layer soil-moisture "
        "sensitivity: K3-AR3"
    )

    print(
        primary[
            [
                "layer_label",
                "State_partial_R2_without_moisture",
                "State_partial_R2_given_air_moisture",
                "Moisture_partial_R2_given_air_state",
                "State_partial_R2_absolute_reduction",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()