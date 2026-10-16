"""
===============================================================================
EXPERIMENT 25
STATISTICAL VALIDATION OF COMMON FOUR-METHOD DETECTION PERFORMANCE
===============================================================================

Purpose
-------
Statistically compare DQN, PPO, DDPG and GA using the COMMON detection
evaluation produced by Experiment 24.

The analysis covers:
    1. Leak detection
    2. Contamination detection
    3. Combined detection
    4. Coverage

Design
------
Four methods:
    DQN, PPO, DDPG, GA

Networks:
    Net3, C-Town, Richmond

Seeds:
    42-51

Expected paired conditions:
    3 networks x 10 seeds = 30

For each metric, pairwise comparisons are performed using the same
network-seed condition for both algorithms.

Statistical test:
    Wilcoxon signed-rank test

Multiple-comparison correction:
    Holm correction

Effect size:
    Rank-biserial correlation

Outputs
-------
data/processed/evaluation/detection_statistical_analysis/

    detection_paired_observations.csv
    leak_detection_statistics.csv
    contamination_detection_statistics.csv
    combined_detection_statistics.csv
    coverage_statistics.csv
    detection_method_summary.csv
    detection_network_summary.csv
    detection_statistical_summary.json

Figures:
    figures/
        leak_detection_comparison.png
        contamination_detection_comparison.png
        combined_detection_comparison.png
        coverage_comparison.png
        detection_heatmap.png

===============================================================================
"""

from __future__ import annotations

import json
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import os

GENERATE_FIGURES = (
    os.environ.get("EXPERIMENT25_GENERATE_FIGURES", "0") == "1"
)

if GENERATE_FIGURES:
    import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
except ImportError:
    raise ImportError(
        "\nSciPy is required for Experiment 25.\n\n"
        "Install it in the rl_env environment using:\n"
        "conda install scipy\n\n"
        "or:\n"
        "python -m pip install scipy\n"
    )


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"C:\Blockchain_DRL_WDN")

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "four_method_detection"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "detection_statistical_analysis"
)

FIGURE_DIR = OUTPUT_DIR / "figures"

METHODS = ["DQN", "PPO", "DDPG", "GA"]

NETWORKS = ["Net3", "C-Town", "Richmond"]

SEEDS = list(range(42, 52))

ALPHA = 0.05


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 79)
    print(title)
    print("=" * 79)


# =============================================================================
# LOAD DATA
# =============================================================================

def find_detection_file() -> Path:
    candidates = [
        INPUT_DIR / "four_method_detection_seed_results.csv",
        INPUT_DIR / "four_method_detection_network_summary.csv",
        INPUT_DIR / "four_method_detection_method_summary.csv",
    ]

    seed_file = candidates[0]

    if not seed_file.exists():
        raise FileNotFoundError(
            "\nExperiment 24 seed-level detection results were not found.\n"
            f"Expected:\n{seed_file}\n\n"
            "Run Experiment 24 first."
        )

    return seed_file


def load_detection_data() -> pd.DataFrame:

    print_header("LOADING EXPERIMENT 24 DETECTION DATA")

    file_path = find_detection_file()

    print(f"Input file:")
    print(file_path)

    df = pd.read_csv(file_path)

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {list(df.columns)}")

    return df


# =============================================================================
# COLUMN NORMALISATION
# =============================================================================

def identify_column(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:

    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in lower_map:
            return lower_map[key]

    raise ValueError(
        f"\nCould not identify the required {label} column.\n"
        f"Candidates checked: {candidates}\n"
        f"Columns found: {list(df.columns)}"
    )


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:

    network_col = identify_column(
        df,
        ["network"],
        "network",
    )

    method_col = identify_column(
        df,
        ["method"],
        "method",
    )

    seed_col = identify_column(
        df,
        ["seed"],
        "seed",
    )

    leak_col = identify_column(
        df,
        [
            "leak_detection_percent",
            "mean_leak_detection_percent",
            "leak_detection",
            "leak_detection_rate",
            "leak_detection_percentage",
        ],
        "leak detection",
    )

    contamination_col = identify_column(
        df,
        [
            "contamination_detection_percent",
            "mean_contamination_detection_percent",
            "contamination_detection",
            "contamination_detection_rate",
            "contamination_detection_percentage",
        ],
        "contamination detection",
    )

    coverage_col = identify_column(
        df,
        [
            "coverage_percent",
            "mean_coverage_percent",
            "coverage",
        ],
        "coverage",
    )

    rename_map = {
        network_col: "network",
        method_col: "method",
        seed_col: "seed",
        leak_col: "leak_detection_percent",
        contamination_col: "contamination_detection_percent",
        coverage_col: "coverage_percent",
    }

    df = df.rename(columns=rename_map).copy()

    df["network"] = df["network"].astype(str)
    df["method"] = df["method"].astype(str)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

    numeric_columns = [
        "leak_detection_percent",
        "contamination_detection_percent",
        "coverage_percent",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Combined detection is the arithmetic mean of the two
    # event-detection measures.
    df["combined_detection_percent"] = (
        df["leak_detection_percent"]
        + df["contamination_detection_percent"]
    ) / 2.0

    return df


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data(df: pd.DataFrame) -> None:

    print_header("DATA VALIDATION")

    unexpected_methods = sorted(
        set(df["method"]) - set(METHODS)
    )

    unexpected_networks = sorted(
        set(df["network"]) - set(NETWORKS)
    )

    unexpected_seeds = sorted(
        set(df["seed"].dropna().astype(int)) - set(SEEDS)
    )

    print(f"Unexpected methods: {unexpected_methods}")
    print(f"Unexpected networks: {unexpected_networks}")
    print(f"Unexpected seeds: {unexpected_seeds}")

    if unexpected_methods:
        raise ValueError(
            f"Unexpected methods found: {unexpected_methods}"
        )

    if unexpected_networks:
        raise ValueError(
            f"Unexpected networks found: {unexpected_networks}"
        )

    if unexpected_seeds:
        raise ValueError(
            f"Unexpected seeds found: {unexpected_seeds}"
        )

    expected_rows = (
        len(METHODS)
        * len(NETWORKS)
        * len(SEEDS)
    )

    print(f"Expected rows: {expected_rows}")
    print(f"Actual rows: {len(df)}")

    duplicate_mask = df.duplicated(
        subset=["method", "network", "seed"],
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    print(f"Duplicate rows: {duplicate_count}")

    if duplicate_count > 0:

        duplicate_rows = df.loc[
            duplicate_mask,
            ["method", "network", "seed"],
        ]

        print("\nDuplicate conditions:")
        print(duplicate_rows.to_string(index=False))

        raise ValueError(
            "Duplicate method/network/seed conditions detected."
        )

    missing_conditions = []

    expected_conditions = list(
        itertools.product(
            METHODS,
            NETWORKS,
            SEEDS,
        )
    )

    observed_conditions = set(
        zip(
            df["method"],
            df["network"],
            df["seed"].astype(int),
        )
    )

    for condition in expected_conditions:

        if condition not in observed_conditions:
            missing_conditions.append(condition)

    print(f"Missing conditions: {len(missing_conditions)}")

    if missing_conditions:
        print(missing_conditions)

        raise ValueError(
            "Missing method/network/seed conditions detected."
        )

    print("Data integrity: PASS")


# =============================================================================
# PAIRED DATASET
# =============================================================================

def construct_paired_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header("CONSTRUCTING BALANCED PAIRED DATASET")

    required_rows = (
        len(NETWORKS)
        * len(SEEDS)
    )

    records = []

    for network in NETWORKS:

        for seed in SEEDS:

            subset = df[
                (df["network"] == network)
                & (df["seed"] == seed)
            ].copy()

            available_methods = set(
                subset["method"]
            )

            missing_methods = (
                set(METHODS) - available_methods
            )

            if missing_methods:
                raise ValueError(
                    f"Missing methods for "
                    f"{network}, seed {seed}: "
                    f"{missing_methods}"
                )

            for method in METHODS:

                row = subset[
                    subset["method"] == method
                ].iloc[0]

                records.append(
                    {
                        "network": network,
                        "seed": int(seed),
                        "method": method,

                        "leak_detection_percent":
                            float(
                                row[
                                    "leak_detection_percent"
                                ]
                            ),

                        "contamination_detection_percent":
                            float(
                                row[
                                    "contamination_detection_percent"
                                ]
                            ),

                        "combined_detection_percent":
                            float(
                                row[
                                    "combined_detection_percent"
                                ]
                            ),

                        "coverage_percent":
                            float(
                                row[
                                    "coverage_percent"
                                ]
                            ),
                    }
                )

    paired_df = pd.DataFrame(records)

    print(
        f"Balanced observations: "
        f"{len(paired_df)}"
    )

    print(
        f"Expected observations: "
        f"{required_rows * len(METHODS)}"
    )

    expected_unique_conditions = required_rows

    unique_conditions = (
        paired_df[
            ["network", "seed"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Unique paired conditions: "
        f"{unique_conditions}"
    )

    if unique_conditions != expected_unique_conditions:
        raise ValueError(
            "Paired dataset is incomplete."
        )

    print("Pairing status: COMPLETE")

    return paired_df


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def rank_biserial_effect(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calculate matched-pairs rank-biserial correlation for x - y.

    Positive values indicate that x tends to exceed y.
    Negative values indicate that x tends to be below y.
    Zero differences are excluded, consistent with the Wilcoxon
    signed-rank test using zero_method="wilcox".
    """

    from scipy.stats import rankdata

    differences = (
        np.asarray(x, dtype=float)
        - np.asarray(y, dtype=float)
    )

    differences = differences[
        ~np.isclose(
            differences,
            0.0,
            atol=1e-12,
        )
    ]

    if len(differences) == 0:
        return 0.0

    # Rank absolute differences, averaging tied ranks.
    ranks = rankdata(
        np.abs(differences),
        method="average",
    )

    positive_rank_sum = float(
        np.sum(ranks[differences > 0])
    )

    negative_rank_sum = float(
        np.sum(ranks[differences < 0])
    )

    total_rank_sum = (
        positive_rank_sum + negative_rank_sum
    )

    if total_rank_sum == 0:
        return 0.0

    return (
        positive_rank_sum - negative_rank_sum
    ) / total_rank_sum

def interpret_effect(effect: float) -> str:

    magnitude = abs(effect)

    if magnitude < 0.10:
        return "negligible"

    if magnitude < 0.30:
        return "small"

    if magnitude < 0.50:
        return "moderate"

    return "large"


def safe_wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
):

    differences = x - y

    nonzero = ~np.isclose(
        differences,
        0.0,
        atol=1e-12,
    )

    x_nz = x[nonzero]
    y_nz = y[nonzero]

    if len(x_nz) == 0:
        return 0.0, 1.0, 0

    try:

        result = wilcoxon(
            x_nz,
            y_nz,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )

        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    except Exception:

        statistic = np.nan
        p_value = np.nan

    return (
        statistic,
        p_value,
        len(x_nz),
    )


def holm_adjust(
    p_values: list[float],
) -> list[float]:

    p = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p)

    order = np.argsort(p)

    adjusted = np.empty(m)

    previous = 0.0

    for rank, index in enumerate(order):

        value = (
            (m - rank)
            * p[index]
        )

        value = max(
            value,
            previous,
        )

        value = min(
            value,
            1.0,
        )

        adjusted[index] = value

        previous = value

    return adjusted.tolist()


# =============================================================================
# PAIRWISE ANALYSIS
# =============================================================================

def pairwise_analysis(
    paired_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:

    results = []

    for method_a, method_b in itertools.combinations(
        METHODS,
        2,
    ):

        pivot = paired_df.pivot_table(
            index=["network", "seed"],
            columns="method",
            values=metric,
            aggfunc="first",
        )

        pivot = pivot.dropna(
            subset=[method_a, method_b]
        )

        x = pivot[method_a].to_numpy(
            dtype=float
        )

        y = pivot[method_b].to_numpy(
            dtype=float
        )

        statistic, p_value, nonzero_n = (
            safe_wilcoxon(x, y)
        )

        effect = rank_biserial_effect(
            x,
            y,
        )

        mean_a = float(np.mean(x))
        mean_b = float(np.mean(y))

        mean_difference = mean_a - mean_b

        median_difference = float(
            np.median(x - y)
        )

        results.append(
            {
                "metric": metric,

                "method_a": method_a,
                "method_b": method_b,

                "n_pairs": len(x),
                "nonzero_pairs": nonzero_n,

                "mean_method_a": mean_a,
                "mean_method_b": mean_b,

                "mean_difference":
                    mean_difference,

                "median_difference":
                    median_difference,

                "wilcoxon_statistic":
                    statistic,

                "raw_p_value":
                    p_value,

                "rank_biserial_effect":
                    effect,

                "effect_interpretation":
                    interpret_effect(effect),
            }
        )

    result_df = pd.DataFrame(results)

    result_df[
        "holm_adjusted_p_value"
    ] = holm_adjust(
        result_df[
            "raw_p_value"
        ].fillna(1.0).tolist()
    )

    result_df[
        "significant_after_holm"
    ] = (
        result_df[
            "holm_adjusted_p_value"
        ]
        < ALPHA
    )

    return result_df


# =============================================================================
# DESCRIPTIVE SUMMARY
# =============================================================================

def descriptive_summary(
    paired_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:

    summary = (
        paired_df
        .groupby("method")[metric]
        .agg(
            n_observations="count",
            mean="mean",
            std="std",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "mean": "mean_value",
            "std": "std_value",
            "median": "median_value",
            "minimum": "minimum_value",
            "maximum": "maximum_value",
        }
    )

    summary = summary.sort_values(
        "mean_value",
        ascending=False,
    )

    summary["rank"] = (
        summary["mean_value"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return summary


# =============================================================================
# NETWORK-LEVEL SUMMARY
# =============================================================================

def network_summary(
    paired_df: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [
        "leak_detection_percent",
        "contamination_detection_percent",
        "combined_detection_percent",
        "coverage_percent",
    ]

    rows = []

    for network in NETWORKS:

        subset = paired_df[
            paired_df["network"] == network
        ]

        for method in METHODS:

            method_data = subset[
                subset["method"] == method
            ]

            row = {
                "network": network,
                "method": method,
            }

            for metric in metrics:

                values = method_data[
                    metric
                ]

                row[
                    f"mean_{metric}"
                ] = float(
                    values.mean()
                )

                row[
                    f"std_{metric}"
                ] = float(
                    values.std()
                )

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# FIGURES
# =============================================================================

def create_metric_boxplot(
    paired_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:

    data = [
        paired_df[
            paired_df["method"] == method
        ][metric]
        .to_numpy()
        for method in METHODS
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        data,
        labels=METHODS,
    )

    plt.ylabel(ylabel)

    plt.xlabel("Method")

    plt.title(
        f"{ylabel}: DQN, PPO, DDPG and GA"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_mean_barplot(
    paired_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:

    means = (
        paired_df
        .groupby("method")[metric]
        .mean()
        .reindex(METHODS)
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        METHODS,
        means.values,
    )

    plt.ylabel(ylabel)

    plt.xlabel("Method")

    plt.title(
        f"Mean {ylabel} Across All Networks and Seeds"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_detection_heatmap(
    paired_df: pd.DataFrame,
) -> None:

    summary = (
        paired_df
        .groupby("method")[
            [
                "leak_detection_percent",
                "contamination_detection_percent",
                "combined_detection_percent",
                "coverage_percent",
            ]
        ]
        .mean()
        .reindex(METHODS)
    )

    summary.columns = [
        "Leak",
        "Contamination",
        "Combined",
        "Coverage",
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.imshow(
        summary.values,
        aspect="auto",
    )

    plt.xticks(
        range(len(summary.columns)),
        summary.columns,
    )

    plt.yticks(
        range(len(summary.index)),
        summary.index,
    )

    plt.xlabel("Performance metric")

    plt.ylabel("Method")

    plt.title(
        "Four-Method Detection Performance"
    )

    for i in range(
        summary.shape[0]
    ):

        for j in range(
            summary.shape[1]
        ):

            plt.text(
                j,
                i,
                f"{summary.iloc[i, j]:.2f}",
                ha="center",
                va="center",
            )

    plt.colorbar(
        label="Mean percentage"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "detection_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    start_time = pd.Timestamp.now()

    print_header(
        "EXPERIMENT 25\n"
        "STATISTICAL VALIDATION OF COMMON "
        "FOUR-METHOD DETECTION PERFORMANCE"
    )

    print(
        "Methods:",
        METHODS,
    )

    print(
        "Networks:",
        NETWORKS,
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Alpha:",
        ALPHA,
    )

    print(
        "Expected paired conditions:",
        len(NETWORKS) * len(SEEDS),
    )

    # -------------------------------------------------------------------------
    # OUTPUT DIRECTORIES
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    df = load_detection_data()

    # -------------------------------------------------------------------------
    # NORMALISE
    # -------------------------------------------------------------------------

    df = normalise_columns(df)

    print_header("NORMALISED DETECTION DATA")

    print(
        df[
            [
                "network",
                "method",
                "seed",
                "leak_detection_percent",
                "contamination_detection_percent",
                "combined_detection_percent",
                "coverage_percent",
            ]
        ]
        .head()
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # VALIDATE
    # -------------------------------------------------------------------------

    validate_data(df)

    # -------------------------------------------------------------------------
    # PAIRED DATA
    # -------------------------------------------------------------------------

    paired_df = construct_paired_dataset(df)

    paired_df.to_csv(
        OUTPUT_DIR
        / "detection_paired_observations.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    metrics = {
        "leak_detection_percent":
            "Leak detection (%)",

        "contamination_detection_percent":
            "Contamination detection (%)",

        "combined_detection_percent":
            "Combined detection (%)",

        "coverage_percent":
            "Coverage (%)",
    }

    all_statistical_results = []

    # -------------------------------------------------------------------------
    # STATISTICAL ANALYSIS
    # -------------------------------------------------------------------------

    for metric, label in metrics.items():

        print_header(
            f"PAIRWISE WILCOXON TESTS\n{label}"
        )

        result_df = pairwise_analysis(
            paired_df,
            metric,
        )

        all_statistical_results.append(
            result_df
        )

        for _, row in result_df.iterrows():

            print(
                f"\n{row['method_a']} "
                f"vs "
                f"{row['method_b']}"
            )

            print(
                f"  n = {int(row['n_pairs'])}"
            )

            print(
                "  mean difference = "
                f"{row['mean_difference']:.6f}"
            )

            print(
                "  median difference = "
                f"{row['median_difference']:.6f}"
            )

            print(
                "  Wilcoxon p = "
                f"{row['raw_p_value']:.10g}"
            )

            print(
                "  Holm-adjusted p = "
                f"{row['holm_adjusted_p_value']:.10g}"
            )

            print(
                "  rank-biserial = "
                f"{row['rank_biserial_effect']:.6f}"
            )

            print(
                "  effect = "
                f"{row['effect_interpretation']}"
            )

            print(
                "  significant after Holm = "
                f"{bool(row['significant_after_holm'])}"
            )

        safe_name = metric.replace(
            "_percent",
            "",
        )

        result_df.to_csv(
            OUTPUT_DIR
            / f"{safe_name}_statistics.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # COMBINED STATISTICS
    # -------------------------------------------------------------------------

    statistical_df = pd.concat(
        all_statistical_results,
        ignore_index=True,
    )

    statistical_df.to_csv(
        OUTPUT_DIR
        / "all_detection_statistical_results.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # DESCRIPTIVE SUMMARIES
    # -------------------------------------------------------------------------

    print_header(
        "DESCRIPTIVE DETECTION PERFORMANCE SUMMARY"
    )

    summary_rows = []

    for metric, label in metrics.items():

        summary = descriptive_summary(
            paired_df,
            metric,
        )

        print()
        print(label)
        print(
            summary.to_string(
                index=False
            )
        )

        for _, row in summary.iterrows():

            summary_rows.append(
                {
                    "metric": label,
                    "method": row["method"],
                    "n_observations":
                        int(row["n_observations"]),
                    "mean":
                        float(row["mean_value"]),
                    "std":
                        float(row["std_value"]),
                    "median":
                        float(row["median_value"]),
                    "minimum":
                        float(row["minimum_value"]),
                    "maximum":
                        float(row["maximum_value"]),
                    "rank":
                        int(row["rank"]),
                }
            )

    method_summary_df = pd.DataFrame(
        summary_rows
    )

    method_summary_df.to_csv(
        OUTPUT_DIR
        / "detection_method_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # NETWORK SUMMARY
    # -------------------------------------------------------------------------

    network_summary_df = network_summary(
        paired_df
    )

    network_summary_df.to_csv(
        OUTPUT_DIR
        / "detection_network_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # FIGURES

    if GENERATE_FIGURES:
        print_header(
            "GENERATING STATISTICAL COMPARISON FIGURES"
        )
        # FIGURES
        # -------------------------------------------------------------------------

        print_header(
            "GENERATING STATISTICAL COMPARISON FIGURES"
        )

        create_metric_boxplot(
            paired_df,
            "leak_detection_percent",
            "Leak detection (%)",
            "leak_detection_comparison.png",
        )

        create_metric_boxplot(
            paired_df,
            "contamination_detection_percent",
            "Contamination detection (%)",
            "contamination_detection_comparison.png",
        )

        create_metric_boxplot(
            paired_df,
            "combined_detection_percent",
            "Combined detection (%)",
            "combined_detection_comparison.png",
        )

        create_metric_boxplot(
            paired_df,
            "coverage_percent",
            "Coverage (%)",
            "coverage_comparison.png",
        )

        create_mean_barplot(
            paired_df,
            "combined_detection_percent",
            "Combined detection (%)",
            "combined_detection_mean_barplot.png",
        )

        create_detection_heatmap(
            paired_df
        )

        print(
            f"Figures saved: {FIGURE_DIR}"
        )

        # -------------------------------------------------------------------------

    else:
        print("Figure generation skipped; statistical analysis continues.")
        # FINAL RESULTS
    # -------------------------------------------------------------------------

    print_header(
        "FINAL DETECTION STATISTICAL VALIDATION"
    )

    print(
        "Expected paired conditions:",
        len(NETWORKS) * len(SEEDS),
    )

    print(
        "Actual paired conditions:",
        paired_df[
            ["network", "seed"]
        ]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "Methods:",
        ", ".join(METHODS),
    )

    print(
        "Pairwise comparisons per metric:",
        6,
    )

    print(
        "Metrics statistically analysed:",
        len(metrics),
    )

    significant_counts = (
        statistical_df
        .groupby("metric")[
            "significant_after_holm"
        ]
        .sum()
    )

    print("\nSignificant comparisons after Holm:")

    for metric, count in significant_counts.items():

        print(
            f"  {metric}: "
            f"{int(count)}/6"
        )

    # -------------------------------------------------------------------------
    # JSON SUMMARY
    # -------------------------------------------------------------------------

    overall_means = {}

    for metric in metrics:

        overall_means[metric] = (
            paired_df
            .groupby("method")[metric]
            .mean()
            .reindex(METHODS)
            .to_dict()
        )

    final_summary = {
        "experiment":
            "Experiment 25 - Statistical Validation "
            "of Common Four-Method Detection Performance",

        "methods":
            METHODS,

        "networks":
            NETWORKS,

        "seeds":
            SEEDS,

        "alpha":
            ALPHA,

        "expected_paired_conditions":
            len(NETWORKS) * len(SEEDS),

        "actual_paired_conditions":
            int(
                paired_df[
                    ["network", "seed"]
                ]
                .drop_duplicates()
                .shape[0]
            ),

        "pairwise_comparisons_per_metric":
            6,

        "multiple_comparison_correction":
            "Holm",

        "statistical_test":
            "Wilcoxon signed-rank test",

        "effect_size":
            "rank-biserial correlation",

        "overall_means":
            overall_means,

        "significant_comparisons_after_holm":
            {
                str(metric): int(count)
                for metric, count
                in significant_counts.items()
            },

        "data_integrity_pass":
            True,
    }

    with open(
        OUTPUT_DIR
        / "detection_statistical_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_summary,
            f,
            indent=4,
        )

    # -------------------------------------------------------------------------
    # COMPLETION
    # -------------------------------------------------------------------------

    elapsed = (
        pd.Timestamp.now()
        - start_time
    ).total_seconds()

    print_header(
        "EXPERIMENT 25 COMPLETE"
    )

    print(
        f"Total analysis time: "
        f"{elapsed:.3f} s"
    )

    print()
    print(
        "Results saved:"
    )

    print(
        OUTPUT_DIR
        / "detection_paired_observations.csv"
    )

    print(
        OUTPUT_DIR
        / "leak_detection_statistics.csv"
    )

    print(
        OUTPUT_DIR
        / "contamination_detection_statistics.csv"
    )

    print(
        OUTPUT_DIR
        / "combined_detection_statistics.csv"
    )

    print(
        OUTPUT_DIR
        / "coverage_statistics.csv"
    )

    print(
        OUTPUT_DIR
        / "all_detection_statistical_results.csv"
    )

    print(
        OUTPUT_DIR
        / "detection_method_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "detection_network_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "detection_statistical_summary.json"
    )

    print()
    print(
        "Figures saved:"
    )

    print(FIGURE_DIR)

    print(
        "=" * 79
    )


if __name__ == "__main__":
    main()