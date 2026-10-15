"""
Five-fold time-blocked cross-validation with a persistence benchmark.
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

K = 3
AR_ORDER = 3
COMMON_TRAIN_OFFSET = 6

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


def train_based_transform(
    train_raw,
    full_raw,
):
    climatology = (
        train_raw
        .groupby(train_raw.index.month)
        .mean()
    )

    train_month = pd.Series(
        train_raw.index.month,
        index=train_raw.index,
    )

    full_month = pd.Series(
        full_raw.index.month,
        index=full_raw.index,
    )

    train_deseason = (
        train_raw
        - train_month.map(climatology)
    )

    full_deseason = (
        full_raw
        - full_month.map(climatology)
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

    train_positions = (
        full_raw.index.get_indexer(
            train_raw.index
        )
    )

    if np.any(
        train_positions < 0
    ):
        raise ValueError(
            "Training dates are not contained "
            "in the full fold series."
        )

    train_x = full_x[
        train_positions
    ]

    train_y = (
        train_deseason
        .to_numpy(dtype=float)
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
        full_deseason.to_numpy(
            dtype=float
        )
        - (
            intercept
            + slope * full_x
        )
    )

    return pd.Series(
        transformed,
        index=full_raw.index,
    )


def build_model_dataframe(
    air,
    soil,
):
    df = pd.DataFrame(
        index=soil.index
    )

    df["soil"] = soil

    air_cols = []
    state_cols = []

    for lag in range(
        K + 1
    ):
        col = f"air_lag{lag}"

        df[col] = air.shift(
            lag
        )

        air_cols.append(
            col
        )

    for lag in range(
        1,
        AR_ORDER + 1
    ):
        col = f"soil_lag{lag}"

        df[col] = soil.shift(
            lag
        )

        state_cols.append(
            col
        )

    return (
        df.dropna(),
        air_cols,
        state_cols,
    )


def fit_ols(
    df,
    predictors,
):
    y = df[
        "soil"
    ].astype(float)

    X = sm.add_constant(
        df[predictors].astype(float),
        has_constant="add",
    )

    return sm.OLS(
        y,
        X,
    ).fit()


def predict(
    model,
    df,
    predictors,
):
    X = sm.add_constant(
        df[predictors].astype(float),
        has_constant="add",
    )

    X = X.reindex(
        columns=model.model.exog_names
    )

    return np.asarray(
        model.predict(X),
        dtype=float,
    )


def rmse(
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

    return float(
        np.sqrt(
            np.mean(
                (
                    observed
                    - predicted
                ) ** 2
            )
        )
    )


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
        "t2m_raw_C",
        *[
            f"{layer}_raw_C"
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

    records = []

    pooled = {
        layer: {
            "observed": [],
            "air": [],
            "full": [],
            "persistence": [],
        }
        for layer in LAYERS
    }

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
        ].copy()

        fold_train = df.loc[
            train_start:train_end
        ].copy()

        transformed = {}

        for variable in [
            "t2m",
            *LAYERS,
        ]:
            raw_col = (
                "t2m_raw_C"
                if variable == "t2m"
                else f"{variable}_raw_C"
            )

            transformed[
                variable
            ] = train_based_transform(
                fold_train[
                    raw_col
                ],
                fold_full[
                    raw_col
                ],
            )

        for layer in LAYERS:

            (
                model_df,
                air_cols,
                state_cols,
            ) = build_model_dataframe(
                transformed["t2m"],
                transformed[layer],
            )

            common_start = (
                train_start
                + pd.DateOffset(
                    months=COMMON_TRAIN_OFFSET
                )
            )

            train_df = model_df.loc[
                common_start:train_end
            ].copy()

            test_df = model_df.loc[
                test_start:test_end
            ].copy()

            if (
                len(train_df) < 60
                or len(test_df) < 12
            ):
                raise ValueError(
                    f"{fold}, {layer}: "
                    "insufficient train/test data."
                )

            air_model = fit_ols(
                train_df,
                air_cols,
            )

            full_model = fit_ols(
                train_df,
                air_cols + state_cols,
            )

            observed = (
                test_df["soil"]
                .to_numpy(dtype=float)
            )

            pred_air = predict(
                air_model,
                test_df,
                air_cols,
            )

            pred_full = predict(
                full_model,
                test_df,
                air_cols + state_cols,
            )

            pred_persistence = (
                test_df["soil_lag1"]
                .to_numpy(dtype=float)
            )

            rmse_air = rmse(
                observed,
                pred_air,
            )

            rmse_full = rmse(
                observed,
                pred_full,
            )

            rmse_persistence = rmse(
                observed,
                pred_persistence,
            )

            state_skill = (
                (
                    rmse_air
                    - rmse_full
                )
                / rmse_air
            )

            persistence_skill = (
                (
                    rmse_persistence
                    - rmse_full
                )
                / rmse_persistence
            )

            records.append(
                {
                    "fold":
                        fold,
                    "soil_layer":
                        layer,
                    "layer_label":
                        LAYER_LABELS[layer],
                    "train_start":
                        train_start.strftime(
                            "%Y-%m"
                        ),
                    "train_end":
                        train_end.strftime(
                            "%Y-%m"
                        ),
                    "test_start":
                        test_start.strftime(
                            "%Y-%m"
                        ),
                    "test_end":
                        test_end.strftime(
                            "%Y-%m"
                        ),
                    "n_train":
                        len(train_df),
                    "n_test":
                        len(test_df),
                    "RMSE_air_only":
                        rmse_air,
                    "RMSE_persistence":
                        rmse_persistence,
                    "RMSE_full":
                        rmse_full,
                    "State_skill_gain_vs_air_only":
                        state_skill,
                    "Full_skill_gain_vs_persistence":
                        persistence_skill,
                    "Full_better_than_persistence":
                        bool(
                            rmse_full
                            < rmse_persistence
                        ),
                }
            )

            pooled[layer][
                "observed"
            ].extend(
                observed.tolist()
            )

            pooled[layer][
                "air"
            ].extend(
                pred_air.tolist()
            )

            pooled[layer][
                "full"
            ].extend(
                pred_full.tolist()
            )

            pooled[layer][
                "persistence"
            ].extend(
                pred_persistence.tolist()
            )

    folds = pd.DataFrame(
        records
    )

    summary = (
        folds.groupby(
            [
                "soil_layer",
                "layer_label",
            ],
            as_index=False,
        )
        .agg(
            Mean_RMSE_air_only=(
                "RMSE_air_only",
                "mean",
            ),
            Mean_RMSE_persistence=(
                "RMSE_persistence",
                "mean",
            ),
            Mean_RMSE_full=(
                "RMSE_full",
                "mean",
            ),
            Mean_State_skill_gain_vs_air_only=(
                "State_skill_gain_vs_air_only",
                "mean",
            ),
            Mean_Full_skill_gain_vs_persistence=(
                "Full_skill_gain_vs_persistence",
                "mean",
            ),
            Folds_full_better_than_persistence=(
                "Full_better_than_persistence",
                "sum",
            ),
        )
    )

    pooled_rows = []

    for layer in LAYERS:

        observed = np.asarray(
            pooled[layer]["observed"],
            dtype=float,
        )

        pred_air = np.asarray(
            pooled[layer]["air"],
            dtype=float,
        )

        pred_full = np.asarray(
            pooled[layer]["full"],
            dtype=float,
        )

        pred_persistence = np.asarray(
            pooled[layer][
                "persistence"
            ],
            dtype=float,
        )

        pooled_rmse_air = rmse(
            observed,
            pred_air,
        )

        pooled_rmse_full = rmse(
            observed,
            pred_full,
        )

        pooled_rmse_persistence = (
            rmse(
                observed,
                pred_persistence,
            )
        )

        pooled_rows.append(
            {
                "soil_layer":
                    layer,
                "Pooled_RMSE_air_only":
                    pooled_rmse_air,
                "Pooled_RMSE_persistence":
                    pooled_rmse_persistence,
                "Pooled_RMSE_full":
                    pooled_rmse_full,
                "Pooled_Full_skill_gain_vs_persistence":
                    (
                        pooled_rmse_persistence
                        - pooled_rmse_full
                    )
                    / pooled_rmse_persistence,
            }
        )

    summary = summary.merge(
        pd.DataFrame(
            pooled_rows
        ),
        on="soil_layer",
        how="left",
    )

    folds.to_csv(
        OUTPUT_DIR
        / "persistence_benchmark_folds.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "persistence_benchmark_summary.csv",
        index=False,
    )

    display = summary.copy()

    display[
        "Mean_State_skill_gain_vs_air_only"
    ] *= 100

    display[
        "Mean_Full_skill_gain_vs_persistence"
    ] *= 100

    display[
        "Pooled_Full_skill_gain_vs_persistence"
    ] *= 100

    print(
        "\nPersistence benchmark"
    )

    print(
        display[
            [
                "layer_label",
                "Mean_RMSE_air_only",
                "Mean_RMSE_persistence",
                "Mean_RMSE_full",
                "Mean_State_skill_gain_vs_air_only",
                "Mean_Full_skill_gain_vs_persistence",
                "Folds_full_better_than_persistence",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()