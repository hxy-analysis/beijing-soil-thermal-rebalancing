"""
Lag-correlation and soil-temperature persistence diagnostics.
"""

from pathlib import Path

import numpy as np
import pandas as pd


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

LAYER_DEPTHS = {
    "stl1": "0-7 cm",
    "stl2": "7-28 cm",
    "stl3": "28-100 cm",
    "stl4": "100-289 cm",
}

LAGS = range(-6, 7)
ACF_MAX_LAG = 24

BOOTSTRAP_REPS = 2000
BLOCK_LENGTH = 36
RANDOM_SEED = 20261002

EFOLD_BOOTSTRAP_REPS = 2000
EFOLD_BLOCK_LENGTH_YEARS = 3
EFOLD_RANDOM_SEED = 42

EXPECTED_MONTHS = 552
EXPECTED_START = pd.Timestamp("1979-01-01")
EXPECTED_END = pd.Timestamp("2024-12-01")

EFOLD_THRESHOLD = 1 / np.e


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def calculate_acf(values, max_lag=24):
    values = np.asarray(values, dtype=float)
    acf = [1.0]

    for lag in range(1, max_lag + 1):
        x = values[:-lag]
        y = values[lag:]

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 3:
            acf.append(np.nan)
        else:
            acf.append(
                float(
                    np.corrcoef(
                        x[valid],
                        y[valid],
                    )[0, 1]
                )
            )

    return np.asarray(acf, dtype=float)


def calculate_efolding(acf):
    for lag in range(1, len(acf)):
        previous = acf[lag - 1]
        current = acf[lag]

        if not (
            np.isfinite(previous)
            and np.isfinite(current)
        ):
            continue

        if current <= EFOLD_THRESHOLD:
            denominator = previous - current

            if denominator == 0:
                return float(lag)

            fraction = (
                previous - EFOLD_THRESHOLD
            ) / denominator

            return float(
                (lag - 1) + fraction
            )

    return np.nan


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

        block = (
            start
            + np.arange(block_length)
        ) % n

        blocks.append(block)

    return np.concatenate(blocks)[:n]


def percentile_ci(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def load_data():
    df = pd.read_csv(INPUT_FILE)

    if "date" not in df.columns:
        raise ValueError("Input file has no 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

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

    if (
        df.index.min() != EXPECTED_START
        or df.index.max() != EXPECTED_END
    ):
        raise ValueError(
            "Unexpected time range."
        )

    if df.index.duplicated().any():
        raise ValueError(
            "Duplicate months found."
        )

    return df


def lag_correlation_bootstrap(df):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    air = df[
        "t2m_anom_deseason_detrend_C"
    ]

    soil = {
        layer: df[
            f"{layer}_anom_deseason_detrend_C"
        ]
        for layer in LAYERS
    }

    records = []

    for lag in LAGS:
        aligned = pd.DataFrame(
            {
                "air": air.shift(lag),
                **soil,
            }
        ).dropna()

        n = len(aligned)
        x = aligned["air"].to_numpy(
            dtype=float
        )

        point_r = {
            layer: pearson_r(
                x,
                aligned[layer].to_numpy(
                    dtype=float
                ),
            )
            for layer in LAYERS
        }

        boot_r = {
            layer: np.full(
                BOOTSTRAP_REPS,
                np.nan,
            )
            for layer in LAYERS
        }

        layer_values = {
            layer: aligned[
                layer
            ].to_numpy(dtype=float)
            for layer in LAYERS
        }

        for b in range(
            BOOTSTRAP_REPS
        ):
            idx = circular_block_indices(
                n,
                BLOCK_LENGTH,
                rng,
            )

            x_boot = x[idx]

            for layer in LAYERS:
                boot_r[layer][b] = (
                    pearson_r(
                        x_boot,
                        layer_values[layer][idx],
                    )
                )

        for layer in LAYERS:
            ci_low, ci_high = percentile_ci(
                boot_r[layer]
            )

            records.append(
                {
                    "soil_layer": layer,
                    "layer_label":
                        LAYER_LABELS[layer],
                    "depth":
                        LAYER_DEPTHS[layer],
                    "lag_months": lag,
                    "n": n,
                    "correlation_r":
                        point_r[layer],
                    "CI95_lower": ci_low,
                    "CI95_upper": ci_high,
                    "bootstrap_reps":
                        BOOTSTRAP_REPS,
                    "block_length_months":
                        BLOCK_LENGTH,
                }
            )

    return pd.DataFrame(records)


def peak_summary(lag_df):
    rows = []

    for layer in LAYERS:
        sub = lag_df.loc[
            lag_df["soil_layer"] == layer
        ].copy()

        peak = sub.loc[
            sub["correlation_r"].idxmax()
        ]

        zero = sub.loc[
            sub["lag_months"] == 0
        ].iloc[0]

        rows.append(
            {
                "soil_layer": layer,
                "layer_label":
                    LAYER_LABELS[layer],
                "depth":
                    LAYER_DEPTHS[layer],
                "peak_lag_months":
                    int(peak["lag_months"]),
                "peak_r":
                    float(
                        peak["correlation_r"]
                    ),
                "peak_CI95_lower":
                    float(
                        peak["CI95_lower"]
                    ),
                "peak_CI95_upper":
                    float(
                        peak["CI95_upper"]
                    ),
                "lag0_r":
                    float(
                        zero["correlation_r"]
                    ),
                "lag0_CI95_lower":
                    float(
                        zero["CI95_lower"]
                    ),
                "lag0_CI95_upper":
                    float(
                        zero["CI95_upper"]
                    ),
            }
        )

    return pd.DataFrame(rows)


def persistence_diagnostics(df):
    soil = {
        layer: df[
            f"{layer}_anom_deseason_detrend_C"
        ].to_numpy(dtype=float)
        for layer in LAYERS
    }

    acf_rows = []
    summary_rows = []

    for layer in LAYERS:
        acf = calculate_acf(
            soil[layer],
            ACF_MAX_LAG,
        )

        for lag, value in enumerate(acf):
            acf_rows.append(
                {
                    "soil_layer": layer,
                    "layer_label":
                        LAYER_LABELS[layer],
                    "depth":
                        LAYER_DEPTHS[layer],
                    "lag_months": lag,
                    "ACF": value,
                }
            )

        summary_rows.append(
            {
                "soil_layer": layer,
                "layer_label":
                    LAYER_LABELS[layer],
                "depth":
                    LAYER_DEPTHS[layer],
                "Lag1_ACF":
                    float(acf[1]),
                "Efolding_months":
                    calculate_efolding(acf),
            }
        )

    acf_df = pd.DataFrame(
        acf_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    n_months = len(df)

    if n_months % 12 != 0:
        raise ValueError(
            "Monthly series length must be "
            "divisible by 12 for the "
            "year-block e-folding bootstrap."
        )

    n_years = (
        n_months // 12
    )

    if n_years != 46:
        raise ValueError(
            f"Expected 46 years, "
            f"found {n_years}."
        )

    rng = np.random.default_rng(
        EFOLD_RANDOM_SEED
    )

    bootstrap_year_indices = (
        np.empty(
            (
                EFOLD_BOOTSTRAP_REPS,
                n_years,
            ),
            dtype=int,
        )
    )

    for b in range(
        EFOLD_BOOTSTRAP_REPS
    ):
        sampled = []

        while len(sampled) < n_years:
            start = int(
                rng.integers(
                    0,
                    n_years,
                )
            )

            block = [
                (
                    start + offset
                ) % n_years
                for offset in range(
                    EFOLD_BLOCK_LENGTH_YEARS
                )
            ]

            sampled.extend(
                block
            )

        bootstrap_year_indices[
            b
        ] = sampled[
            :n_years
        ]

    for layer in LAYERS:
        values_by_year = (
            soil[layer]
            .reshape(
                n_years,
                12,
            )
        )

        boot_efolding = np.full(
            EFOLD_BOOTSTRAP_REPS,
            np.nan,
        )

        for b in range(
            EFOLD_BOOTSTRAP_REPS
        ):
            sample = (
                values_by_year[
                    bootstrap_year_indices[b]
                ]
                .reshape(-1)
            )

            boot_acf = calculate_acf(
                sample,
                ACF_MAX_LAG,
            )

            boot_efolding[b] = (
                calculate_efolding(
                    boot_acf
                )
            )

        ci_low, ci_high = (
            percentile_ci(
                boot_efolding
            )
        )

        mask = (
            summary_df[
                "soil_layer"
            ]
            == layer
        )

        summary_df.loc[
            mask,
            "Efolding_CI95_lower",
        ] = ci_low

        summary_df.loc[
            mask,
            "Efolding_CI95_upper",
        ] = ci_high

    summary_df[
        "bootstrap_reps"
    ] = EFOLD_BOOTSTRAP_REPS

    summary_df[
        "block_length_months"
    ] = (
        EFOLD_BLOCK_LENGTH_YEARS
        * 12
    )

    return (
        acf_df,
        summary_df,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    lag_df = lag_correlation_bootstrap(
        df
    )

    peak_df = peak_summary(
        lag_df
    )

    acf_df, persistence_df = (
        persistence_diagnostics(df)
    )

    temporal_summary = (
        peak_df.merge(
            persistence_df,
            on=[
                "soil_layer",
                "layer_label",
                "depth",
            ],
            how="left",
        )
    )

    lag_df.to_csv(
        OUTPUT_DIR
        / "lag_correlation_bootstrap_all_lags.csv",
        index=False,
    )

    acf_df.to_csv(
        OUTPUT_DIR
        / "soil_temperature_acf.csv",
        index=False,
    )

    temporal_summary.to_csv(
        OUTPUT_DIR
        / "temporal_characteristics_summary.csv",
        index=False,
    )

    print("\nTemporal characteristics")
    print(
        temporal_summary[
            [
                "layer_label",
                "peak_r",
                "peak_lag_months",
                "lag0_r",
                "Lag1_ACF",
                "Efolding_months",
                "Efolding_CI95_lower",
                "Efolding_CI95_upper",
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