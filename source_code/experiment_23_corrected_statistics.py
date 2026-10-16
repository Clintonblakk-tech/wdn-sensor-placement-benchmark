"""
===============================================================================
EXPERIMENT 23
CORRECTED PAIRED STATISTICAL ANALYSIS
===============================================================================

Article:
Blockchain-Enabled Reinforcement Learning for Sensor Placement in Water
Distribution Networks: A Comparative Evaluation of DQN, PPO, DDPG and
Genetic Algorithms

Purpose
-------
Perform the final statistical comparison of DQN, PPO, DDPG and GA using the
complete balanced experimental design verified by Experiment 22.

Experimental design
-------------------
Methods:
    DQN
    PPO
    DDPG
    GA

Networks:
    Net3
    C-Town
    Richmond

Seeds:
    42-51

Coverage radii:
    1-4

Total paired conditions:

    3 networks × 4 radii × 10 seeds = 120

Therefore every pairwise method comparison must contain exactly:

    n = 120 paired observations

Primary statistical test:
    Wilcoxon signed-rank test

Effect size:
    Rank-biserial correlation

Multiple-comparison correction:
    Holm-Bonferroni

This experiment DOES NOT retrain any algorithm.

===============================================================================
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
)

INPUT_DIR = (
    EVALUATION_DIR
    / "four_method_comparison"
)

INPUT_FILE = (
    INPUT_DIR
    / "four_method_coverage_raw.csv"
)

OUTPUT_DIR = (
    EVALUATION_DIR
    / "corrected_statistical_analysis"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

METHODS = [
    "DQN",
    "PPO",
    "DDPG",
    "GA",
]

NETWORKS = [
    "Net3",
    "C-Town",
    "Richmond",
]

SEEDS = list(
    range(42, 52)
)

RADII = [
    1,
    2,
    3,
    4,
]

ALPHA = 0.05


EXPECTED_PAIRS = (
    len(NETWORKS)
    * len(RADII)
    * len(SEEDS)
)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print("=" * 79)
    print("LOADING BALANCED FOUR-METHOD COVERAGE DATA")
    print("=" * 79)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nCoverage file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    required = [
        "method",
        "network",
        "seed",
        "coverage_radius",
        "coverage_percent",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["method"] = (
        df["method"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["network"] = (
        df["network"]
        .astype(str)
        .str.strip()
    )

    df["seed"] = pd.to_numeric(
        df["seed"],
        errors="coerce"
    )

    df["coverage_radius"] = pd.to_numeric(
        df["coverage_radius"],
        errors="coerce"
    )

    df["coverage_percent"] = pd.to_numeric(
        df["coverage_percent"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "method",
            "network",
            "seed",
            "coverage_radius",
            "coverage_percent",
        ]
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    return df


# =============================================================================
# CREATE COMPLETE EXPERIMENTAL GRID
# =============================================================================

def create_expected_grid():

    rows = []

    for network in NETWORKS:

        for radius in RADII:

            for seed in SEEDS:

                rows.append({
                    "network": network,
                    "coverage_radius": radius,
                    "seed": seed,
                })

    return pd.DataFrame(rows)


# =============================================================================
# PREPARE PAIRED DATA
# =============================================================================

def prepare_paired_data(
    df: pd.DataFrame,
):

    print()
    print("=" * 79)
    print("CONSTRUCTING BALANCED PAIRED DATASET")
    print("=" * 79)

    # ---------------------------------------------------------------
    # Keep only the required experimental domain.
    # ---------------------------------------------------------------

    df = df[
        df["method"].isin(METHODS)
        &
        df["network"].isin(NETWORKS)
        &
        df["seed"].isin(SEEDS)
        &
        df["coverage_radius"].isin(RADII)
    ].copy()

    # ---------------------------------------------------------------
    # Detect duplicate method-condition rows.
    # ---------------------------------------------------------------

    key_columns = [
        "method",
        "network",
        "coverage_radius",
        "seed",
    ]

    duplicates = (
        df
        .groupby(key_columns)
        .size()
        .reset_index(name="count")
    )

    duplicates = duplicates[
        duplicates["count"] > 1
    ]

    if not duplicates.empty:

        raise ValueError(
            "\nDuplicate experimental conditions detected.\n"
            "Experiment 23 requires one observation for each "
            "method/network/radius/seed combination.\n\n"
            f"{duplicates.to_string(index=False)}"
        )

    # ---------------------------------------------------------------
    # Pivot methods.
    # ---------------------------------------------------------------

    paired = df.pivot(
        index=[
            "network",
            "coverage_radius",
            "seed",
        ],
        columns="method",
        values="coverage_percent",
    ).reset_index()

    # ---------------------------------------------------------------
    # Ensure all methods are present.
    # ---------------------------------------------------------------

    for method in METHODS:

        if method not in paired.columns:

            raise ValueError(
                f"Method {method} is missing from paired dataset."
            )

    # ---------------------------------------------------------------
    # Reorder.
    # ---------------------------------------------------------------

    paired = paired[
        [
            "network",
            "coverage_radius",
            "seed",
            *METHODS,
        ]
    ]

    # ---------------------------------------------------------------
    # Check for missing values.
    # ---------------------------------------------------------------

    missing_mask = paired[
        METHODS
    ].isna().any(
        axis=1
    )

    missing_rows = paired[
        missing_mask
    ]

    if not missing_rows.empty:

        raise ValueError(
            "\nIncomplete paired observations detected:\n\n"
            f"{missing_rows.to_string(index=False)}"
        )

    # ---------------------------------------------------------------
    # Check exact expected number.
    # ---------------------------------------------------------------

    if len(paired) != EXPECTED_PAIRS:

        raise ValueError(
            "\nUnexpected number of paired conditions.\n"
            f"Expected: {EXPECTED_PAIRS}\n"
            f"Observed: {len(paired)}"
        )

    print(
        f"Balanced paired observations: {len(paired)}"
    )

    print(
        f"Expected paired observations: {EXPECTED_PAIRS}"
    )

    print(
        "Pairing status: COMPLETE"
    )

    return paired


# =============================================================================
# RANK-BISERIAL EFFECT SIZE
# =============================================================================

def rank_biserial(
    x,
    y,
):

    differences = (
        np.asarray(x)
        -
        np.asarray(y)
    )

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:

        return 0.0

    positive = np.sum(
        differences > 0
    )

    negative = np.sum(
        differences < 0
    )

    n = (
        positive
        +
        negative
    )

    return (
        positive
        -
        negative
    ) / n


# =============================================================================
# EFFECT INTERPRETATION
# =============================================================================

def interpret_effect(
    value
):

    magnitude = abs(
        value
    )

    if magnitude < 0.10:
        return "negligible"

    elif magnitude < 0.30:
        return "small"

    elif magnitude < 0.50:
        return "moderate"

    else:
        return "large"


# =============================================================================
# HOLM-BONFERRONI
# =============================================================================

def holm_adjust(
    p_values
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    m = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty(
        m
    )

    running = 0.0

    for rank, index in enumerate(
        order
    ):

        value = (
            m - rank
        ) * p_values[index]

        running = max(
            running,
            value
        )

        adjusted[index] = min(
            running,
            1.0
        )

    return adjusted


# =============================================================================
# PAIRWISE STATISTICAL TEST
# =============================================================================

def perform_pairwise_tests(
    paired: pd.DataFrame,
):

    rows = []

    pairs = list(
        combinations(
            METHODS,
            2
        )
    )

    print()
    print("=" * 79)
    print("PAIRWISE WILCOXON TESTS")
    print("=" * 79)

    for method_a, method_b in pairs:

        x = paired[
            method_a
        ].to_numpy(
            dtype=float
        )

        y = paired[
            method_b
        ].to_numpy(
            dtype=float
        )

        # -----------------------------------------------------------
        # Exact same 120 observations.
        # -----------------------------------------------------------

        if len(x) != EXPECTED_PAIRS:

            raise ValueError(
                f"{method_a} vs {method_b}: "
                f"expected {EXPECTED_PAIRS} pairs, "
                f"found {len(x)}."
            )

        differences = (
            x - y
        )

        nonzero = (
            differences != 0
        )

        effective_n = int(
            np.sum(nonzero)
        )

        if effective_n >= 2:

            result = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            statistic = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        else:

            statistic = np.nan
            p_value = 1.0

        effect = rank_biserial(
            x,
            y
        )

        mean_a = float(
            np.mean(x)
        )

        mean_b = float(
            np.mean(y)
        )

        mean_difference = float(
            np.mean(
                differences
            )
        )

        median_difference = float(
            np.median(
                differences
            )
        )

        print()
        print(
            f"{method_a} vs {method_b}"
        )

        print(
            f"  n = {len(x)}"
        )

        print(
            f"  mean difference = "
            f"{mean_difference:.6f}"
        )

        print(
            f"  Wilcoxon p = "
            f"{p_value:.10g}"
        )

        print(
            f"  rank-biserial = "
            f"{effect:.6f}"
        )

        rows.append({

            "method_a":
                method_a,

            "method_b":
                method_b,

            "n_pairs":
                len(x),

            "nonzero_pairs":
                effective_n,

            "mean_method_a":
                mean_a,

            "mean_method_b":
                mean_b,

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
                interpret_effect(
                    effect
                ),
        })

    result_df = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------------
    # Holm correction across six pairwise comparisons.
    # ---------------------------------------------------------------

    adjusted = holm_adjust(
        result_df[
            "raw_p_value"
        ].to_numpy()
    )

    result_df[
        "holm_adjusted_p_value"
    ] = adjusted

    result_df[
        "significant_after_holm"
    ] = (
        result_df[
            "holm_adjusted_p_value"
        ]
        <
        ALPHA
    )

    return result_df


# =============================================================================
# NETWORK-SPECIFIC STATISTICS
# =============================================================================

def perform_network_statistics(
    df: pd.DataFrame,
):

    rows = []

    pairs = list(
        combinations(
            METHODS,
            2
        )
    )

    print()
    print("=" * 79)
    print("NETWORK-SPECIFIC STATISTICS")
    print("=" * 79)

    for network in NETWORKS:

        for method_a, method_b in pairs:

            subset = df[
                df["network"]
                == network
            ]

            pivot = subset.pivot(
                index=[
                    "coverage_radius",
                    "seed",
                ],
                columns="method",
                values="coverage_percent",
            )

            x = pivot[
                method_a
            ].to_numpy(
                dtype=float
            )

            y = pivot[
                method_b
            ].to_numpy(
                dtype=float
            )

            result = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            effect = rank_biserial(
                x,
                y
            )

            rows.append({

                "network":
                    network,

                "method_a":
                    method_a,

                "method_b":
                    method_b,

                "n_pairs":
                    len(x),

                "mean_method_a":
                    np.mean(x),

                "mean_method_b":
                    np.mean(y),

                "mean_difference":
                    np.mean(x - y),

                "wilcoxon_statistic":
                    float(
                        result.statistic
                    ),

                "raw_p_value":
                    float(
                        result.pvalue
                    ),

                "rank_biserial_effect":
                    effect,

                "effect_interpretation":
                    interpret_effect(
                        effect
                    ),
            })

    result = pd.DataFrame(
        rows
    )

    # Holm correction within the complete set of
    # network-level pairwise tests.

    result[
        "holm_adjusted_p_value"
    ] = holm_adjust(
        result[
            "raw_p_value"
        ].to_numpy()
    )

    result[
        "significant_after_holm"
    ] = (
        result[
            "holm_adjusted_p_value"
        ]
        <
        ALPHA
    )

    return result


# =============================================================================
# RADIUS-SPECIFIC STATISTICS
# =============================================================================

def perform_radius_statistics(
    df: pd.DataFrame,
):

    rows = []

    pairs = list(
        combinations(
            METHODS,
            2
        )
    )

    print()
    print("=" * 79)
    print("COVERAGE-RADIUS STATISTICS")
    print("=" * 79)

    for radius in RADII:

        subset = df[
            df["coverage_radius"]
            == radius
        ]

        pivot = subset.pivot(
            index=[
                "network",
                "seed",
            ],
            columns="method",
            values="coverage_percent",
        )

        for method_a, method_b in pairs:

            x = pivot[
                method_a
            ].to_numpy(
                dtype=float
            )

            y = pivot[
                method_b
            ].to_numpy(
                dtype=float
            )

            result = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            effect = rank_biserial(
                x,
                y
            )

            rows.append({

                "coverage_radius":
                    radius,

                "method_a":
                    method_a,

                "method_b":
                    method_b,

                "n_pairs":
                    len(x),

                "mean_method_a":
                    np.mean(x),

                "mean_method_b":
                    np.mean(y),

                "mean_difference":
                    np.mean(x - y),

                "wilcoxon_statistic":
                    float(
                        result.statistic
                    ),

                "raw_p_value":
                    float(
                        result.pvalue
                    ),

                "rank_biserial_effect":
                    effect,

                "effect_interpretation":
                    interpret_effect(
                        effect
                    ),
            })

    result = pd.DataFrame(
        rows
    )

    result[
        "holm_adjusted_p_value"
    ] = holm_adjust(
        result[
            "raw_p_value"
        ].to_numpy()
    )

    result[
        "significant_after_holm"
    ] = (
        result[
            "holm_adjusted_p_value"
        ]
        <
        ALPHA
    )

    return result


# =============================================================================
# DESCRIPTIVE SUMMARY
# =============================================================================

def descriptive_summary(
    paired: pd.DataFrame,
):

    rows = []

    for method in METHODS:

        values = paired[
            method
        ].to_numpy(
            dtype=float
        )

        rows.append({

            "method":
                method,

            "n_observations":
                len(values),

            "mean_coverage_percent":
                np.mean(values),

            "std_coverage_percent":
                np.std(
                    values,
                    ddof=1
                ),

            "median_coverage_percent":
                np.median(values),

            "minimum_coverage_percent":
                np.min(values),

            "maximum_coverage_percent":
                np.max(values),
        })

    result = pd.DataFrame(
        rows
    )

    result["rank"] = (
        result[
            "mean_coverage_percent"
        ]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    return result.sort_values(
        "rank"
    )


# =============================================================================
# WIN ANALYSIS
# =============================================================================

def win_analysis(
    paired: pd.DataFrame,
):

    wins = {
        method: 0
        for method in METHODS
    }

    total = len(
        paired
    )

    for _, row in paired.iterrows():

        values = {
            method: row[method]
            for method in METHODS
        }

        winner = max(
            values,
            key=values.get
        )

        wins[winner] += 1

    rows = []

    for method in METHODS:

        rows.append({

            "method":
                method,

            "wins":
                wins[method],

            "total_conditions":
                total,

            "win_percentage":
                wins[method]
                /
                total
                *
                100,
        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "win_percentage",
        ascending=False
    )


# =============================================================================
# FIGURES
# =============================================================================

def generate_figures(
    paired: pd.DataFrame,
):

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "WARNING: matplotlib unavailable."
        )

        return

    # -----------------------------------------------------------------
    # Overall coverage
    # -----------------------------------------------------------------

    means = [
        paired[method].mean()
        for method in METHODS
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        METHODS,
        means
    )

    plt.xlabel(
        "Method"
    )

    plt.ylabel(
        "Mean coverage (%)"
    )

    plt.title(
        "Overall sensor-placement coverage"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "overall_method_coverage.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------------------------------------------
    # Coverage by radius
    # -----------------------------------------------------------------

    radius_means = (
        paired
        .groupby(
            "coverage_radius"
        )[METHODS]
        .mean()
    )

    plt.figure(
        figsize=(10, 6)
    )

    for method in METHODS:

        plt.plot(
            radius_means.index,
            radius_means[method],
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Coverage radius (hops)"
    )

    plt.ylabel(
        "Mean coverage (%)"
    )

    plt.title(
        "Coverage performance across spatial radii"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "coverage_by_radius.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------------------------------------------
    # Network-specific coverage
    # -----------------------------------------------------------------

    network_means = (
        paired
        .groupby(
            "network"
        )[METHODS]
        .mean()
    )

    for network in NETWORKS:

        values = [
            network_means.loc[
                network,
                method
            ]
            for method in METHODS
        ]

        plt.figure(
            figsize=(9, 6)
        )

        plt.bar(
            METHODS,
            values
        )

        plt.xlabel(
            "Method"
        )

        plt.ylabel(
            "Mean coverage (%)"
        )

        plt.title(
            f"{network} sensor-placement coverage"
        )

        plt.tight_layout()

        safe_name = (
            network
            .lower()
            .replace(
                "-",
                "_"
            )
        )

        plt.savefig(
            FIGURE_DIR
            / f"{safe_name}_coverage.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    print()
    print(
        f"Figures saved: {FIGURE_DIR}"
    )


# =============================================================================
# SAVE JSON
# =============================================================================

def save_summary(
    descriptive,
    pairwise,
    network_stats,
    radius_stats,
    wins,
):

    summary = {

        "experiment":
            "Experiment 23 - Corrected Paired Statistical Analysis",

        "methods":
            METHODS,

        "networks":
            NETWORKS,

        "seeds":
            SEEDS,

        "coverage_radii":
            RADII,

        "expected_pairs_per_method_comparison":
            EXPECTED_PAIRS,

        "alpha":
            ALPHA,

        "statistical_test":
            "Wilcoxon signed-rank test",

        "effect_size":
            "Rank-biserial correlation",

        "multiple_comparison_correction":
            "Holm-Bonferroni",

        "descriptive_summary":
            descriptive.to_dict(
                orient="records"
            ),

        "overall_pairwise_statistics":
            pairwise.to_dict(
                orient="records"
            ),

        "network_pairwise_statistics":
            network_stats.to_dict(
                orient="records"
            ),

        "radius_pairwise_statistics":
            radius_stats.to_dict(
                orient="records"
            ),

        "win_analysis":
            wins.to_dict(
                orient="records"
            ),
    }

    with open(
        OUTPUT_DIR
        / "experiment_23_statistical_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=str
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    start = time.perf_counter()

    print("=" * 79)
    print("EXPERIMENT 23")
    print("CORRECTED PAIRED STATISTICAL ANALYSIS")
    print("=" * 79)

    print()
    print(
        "Methods:",
        METHODS
    )

    print(
        "Networks:",
        NETWORKS
    )

    print(
        "Seeds:",
        SEEDS
    )

    print(
        "Coverage radii:",
        RADII
    )

    print(
        "Expected paired conditions:",
        EXPECTED_PAIRS
    )

    # =================================================================
    # LOAD
    # =================================================================

    df = load_data()

    # =================================================================
    # PREPARE
    # =================================================================

    paired = prepare_paired_data(
        df
    )

    # =================================================================
    # SAVE PAIRED DATASET
    # =================================================================

    paired.to_csv(
        OUTPUT_DIR
        / "complete_balanced_paired_dataset.csv",
        index=False
    )

    # =================================================================
    # DESCRIPTIVE
    # =================================================================

    descriptive = descriptive_summary(
        paired
    )

    # =================================================================
    # OVERALL PAIRWISE TESTS
    # =================================================================

    pairwise = perform_pairwise_tests(
        paired
    )

    # =================================================================
    # NETWORK TESTS
    # =================================================================

    network_stats = (
        perform_network_statistics(
            df
        )
    )

    # =================================================================
    # RADIUS TESTS
    # =================================================================

    radius_stats = (
        perform_radius_statistics(
            df
        )
    )

    # =================================================================
    # WIN ANALYSIS
    # =================================================================

    wins = win_analysis(
        paired
    )

    # =================================================================
    # SAVE TABLES
    # =================================================================

    descriptive.to_csv(
        OUTPUT_DIR
        / "corrected_descriptive_statistics.csv",
        index=False
    )

    pairwise.to_csv(
        OUTPUT_DIR
        / "corrected_overall_pairwise_tests.csv",
        index=False
    )

    network_stats.to_csv(
        OUTPUT_DIR
        / "corrected_network_pairwise_tests.csv",
        index=False
    )

    radius_stats.to_csv(
        OUTPUT_DIR
        / "corrected_radius_pairwise_tests.csv",
        index=False
    )

    wins.to_csv(
        OUTPUT_DIR
        / "corrected_method_win_analysis.csv",
        index=False
    )

    # =================================================================
    # FIGURES
    # =================================================================

    generate_figures(
        paired
    )

    # =================================================================
    # JSON
    # =================================================================

    save_summary(
        descriptive,
        pairwise,
        network_stats,
        radius_stats,
        wins,
    )

    # =================================================================
    # DISPLAY
    # =================================================================

    print()
    print("=" * 79)
    print("CORRECTED OVERALL STATISTICAL RESULTS")
    print("=" * 79)

    print(
        pairwise.to_string(
            index=False
        )
    )

    print()
    print("=" * 79)
    print("DESCRIPTIVE COVERAGE RANKING")
    print("=" * 79)

    print(
        descriptive.to_string(
            index=False
        )
    )

    print()
    print("=" * 79)
    print("METHOD WIN ANALYSIS")
    print("=" * 79)

    print(
        wins.to_string(
            index=False
        )
    )

    # =================================================================
    # FINAL VALIDATION
    # =================================================================

    all_n_correct = (
        pairwise["n_pairs"]
        == EXPECTED_PAIRS
    ).all()

    if not all_n_correct:

        raise RuntimeError(
            "FINAL VALIDATION FAILED: "
            "not all pairwise comparisons contain "
            f"{EXPECTED_PAIRS} observations."
        )

    print()
    print("=" * 79)
    print("FINAL STATISTICAL VALIDATION")
    print("=" * 79)

    print(
        f"Expected pairs per comparison: "
        f"{EXPECTED_PAIRS}"
    )

    print(
        "All six comparisons have exactly 120 pairs:",
        all_n_correct
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print("=" * 79)
    print("EXPERIMENT 23 COMPLETE")
    print("=" * 79)

    print()
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
    )

    print("=" * 79)


if __name__ == "__main__":
    main()