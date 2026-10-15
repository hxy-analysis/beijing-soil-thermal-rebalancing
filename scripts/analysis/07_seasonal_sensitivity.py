"""
Season-stratified conditional contribution analysis.
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

SEASONS = [
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
]

K_VALUES = [3, 6]

AR_ORDER = 3
COMMON_LAG = 6
EXPECTED_MONTHS = 552


def month_to_season(month):
    if month in [3, 4, 5]:
        return "Spring"

    if month in [6, 7, 8]:
        return "Summer"

    if month in [9, 10, 11]:
        return "Autumn"

    return "Winter"


def build_dataframe(
    air,
    soil,
    k,
):
    df = pd.DataFrame(index=soil.index)

    df["soil"] = soil
    df["season"] = [
        month_to_season(month)
        for month in df.index.month
    ]

    air_cols = []
    state_cols = []

    for lag in range(k + 1):
        col = f"air_lag{lag}"
        df[col] = air.shift(lag)
        air_cols.append(col)

    for lag in range(1, AR_ORDER + 1):
        col = f"soil_lag{lag}"
        df[col] = soil.shift(lag)
        state_cols.append(col)

    required = (
        ["soil"]
        + air_cols
        + state_cols
    )

    df = df.dropna(
        subset=required
    )

    return (
        df,
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

    if df.index.duplicated().any():
        raise ValueError(
            "Duplicate months found."
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

            model_df = model_df.loc[
                common_start:
            ].copy()

            for season in SEASONS:

                seasonal_df = (
                    model_df.loc[
                        model_df["season"]
                        == season
                    ]
                    .copy()
                )

                if len(seasonal_df) < 20:
                    raise ValueError(
                        f"{layer}, K{k}, {season}: "
                        "insufficient sample size."
                    )

                result = fit_models(
                    seasonal_df,
                    air_cols,
                    state_cols,
                )

                rows.append(
                    {
                        "soil_layer":
                            layer,

                        "layer_label":
                            LAYER_LABELS[layer],

                        "season":
                            season,

                        "K":
                            k,

                        "AR_order":
                            AR_ORDER,

                        "n":
                            len(seasonal_df),

                        "R2_air_only":
                            result[
                                "R2_air_only"
                            ],

                        "R2_state_only":
                            result[
                                "R2_state_only"
                            ],

                        "R2_full":
                            result[
                                "R2_full"
                            ],

                        "Air_partial_R2_given_state":
                            result[
                                "Air_partial_R2_given_state"
                            ],

                        "State_partial_R2_given_air":
                            result[
                                "State_partial_R2_given_air"
                            ],
                    }
                )

    results = pd.DataFrame(
        rows
    )

    results["season"] = pd.Categorical(
        results["season"],
        categories=SEASONS,
        ordered=True,
    )

    results = results.sort_values(
        [
            "K",
            "season",
            "soil_layer",
        ]
    )

    results.to_csv(
        OUTPUT_DIR
        / "seasonal_sensitivity.csv",
        index=False,
    )

    primary = results.loc[
        results["K"] == 3
    ].copy()

    primary.to_csv(
        OUTPUT_DIR
        / "seasonal_sensitivity_K3.csv",
        index=False,
    )

    print(
        "\nSeasonal conditional contributions: "
        "K3-AR3"
    )

    print(
        primary[
            [
                "season",
                "layer_label",
                "n",
                "Air_partial_R2_given_state",
                "State_partial_R2_given_air",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()