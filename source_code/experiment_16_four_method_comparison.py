"""
===========================================================================
EXPERIMENT 16
FOUR-METHOD SENSOR-PLACEMENT COMPARISON
===========================================================================

DQN  : Proposed RL method
PPO  : Secondary RL benchmark
DDPG : Secondary RL benchmark
GA   : Metaheuristic baseline

No training is performed.

DQN coverage is reconstructed from the already-existing per-seed
evaluation_selected_sensors.csv files because DQN did not create a
consolidated coverage CSV.

PPO, DDPG and GA coverage are loaded from their existing Experiment
12, 14 and 15 CSV outputs.

Expected observations:

4 methods x 3 networks x 10 seeds x 4 radii = 480
===========================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Blockchain_DRL_WDN"
)

DQN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "dqn"
)

PPO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "ppo_coverage"
)

DDPG_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "ddpg_coverage"
)

GA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "ga_coverage"
)

COVERAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coverage_analysis"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "four_method_comparison"
)

FIGURE_ROOT = OUTPUT_ROOT / "figures"


# ============================================================================
# CONFIGURATION
# ============================================================================

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

COVERAGE_RADII = [
    1,
    2,
    3,
    4,
]

SENSOR_BUDGET = 5


# ============================================================================
# DIRECTORIES
# ============================================================================

def create_directories():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# COVERAGE MATRIX LOADER
# ============================================================================

def load_coverage_matrix(
    network,
    radius,
):

    matrix_path = (
        COVERAGE_ROOT
        / network
        / f"coverage_matrix_r{radius}.npz"
    )

    if not matrix_path.exists():

        raise FileNotFoundError(
            f"\nCoverage matrix not found:\n"
            f"{matrix_path}"
        )

    data = np.load(
        matrix_path,
        allow_pickle=True,
    )

    matrix = None

    for key in [
        "coverage_matrix",
        "matrix",
        "arr_0",
    ]:

        if key in data.files:

            candidate = np.asarray(
                data[key]
            )

            if candidate.ndim == 2:

                matrix = candidate
                break

    if matrix is None:

        for key in data.files:

            candidate = np.asarray(
                data[key]
            )

            if candidate.ndim == 2:

                matrix = candidate
                break

    if matrix is None:

        raise ValueError(
            f"No 2-D coverage matrix found in "
            f"{matrix_path}"
        )

    # ------------------------------------------------------------------------
    # Determine node ordering.
    # ------------------------------------------------------------------------

    node_ids = None

    for key in [
        "node_ids",
        "candidate_nodes",
        "nodes",
        "node_names",
    ]:

        if key in data.files:

            node_ids = [
                str(x)
                for x in np.asarray(
                    data[key]
                ).reshape(-1)
            ]

            break

    # ------------------------------------------------------------------------
    # Fall back to statistics CSV.
    # ------------------------------------------------------------------------

    if node_ids is None:

        stats_path = (
            COVERAGE_ROOT
            / network
            / f"coverage_statistics_r{radius}.csv"
        )

        if stats_path.exists():

            stats = pd.read_csv(
                stats_path
            )

            if "node_id" in stats.columns:

                node_ids = [
                    str(x)
                    for x in stats[
                        "node_id"
                    ].tolist()
                ]

    if node_ids is None:

        raise ValueError(
            f"Could not determine candidate-node "
            f"ordering for {network}, radius {radius}"
        )

    # ------------------------------------------------------------------------
    # Matrix convention:
    #
    # rows    = nodes covered
    # columns = candidate sensor nodes
    # ------------------------------------------------------------------------

    if matrix.shape[1] == len(node_ids):

        pass

    elif matrix.shape[0] == len(node_ids):

        matrix = matrix.T

    else:

        raise ValueError(
            f"Node count {len(node_ids)} does not "
            f"match matrix dimensions {matrix.shape}"
        )

    return (
        matrix,
        node_ids,
    )


# ============================================================================
# DQN SENSOR SELECTION LOADER
# ============================================================================

def load_dqn_sensor_selection(
    network,
    seed,
):

    path = (
        DQN_ROOT
        / network
        / f"seed_{seed}"
        / "evaluation_selected_sensors.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\nDQN sensor-selection file not found:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required = [
        "network",
        "seed",
        "selection_rank",
        "node_id",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"DQN file {path} is missing "
            f"columns: {missing}"
        )

    df = (
        df
        .sort_values(
            "selection_rank"
        )
        .reset_index(
            drop=True
        )
    )

    sensors = [
        str(x).strip()
        for x in df[
            "node_id"
        ].tolist()
    ]

    return sensors


# ============================================================================
# COVERAGE CALCULATION
# ============================================================================

def calculate_coverage(
    sensors,
    matrix,
    node_ids,
):

    node_index = {
        str(node): i
        for i, node
        in enumerate(node_ids)
    }

    valid_sensors = []
    missing_sensors = []

    for sensor in sensors:

        sensor = str(
            sensor
        ).strip()

        if sensor in node_index:

            valid_sensors.append(
                sensor
            )

        else:

            missing_sensors.append(
                sensor
            )

    if not valid_sensors:

        return {
            "covered_nodes": 0,
            "coverage_percent": 0.0,
            "redundant_assignments": 0,
            "mean_sensor_coverage_percent": 0.0,
            "selected_sensor_count":
                len(sensors),
            "valid_sensor_count": 0,
            "missing_sensor_count":
                len(missing_sensors),
        }

    indices = [
        node_index[
            sensor
        ]
        for sensor in valid_sensors
    ]

    selected = matrix[
        :,
        indices
    ]

    covered_by_sensor = (
        selected > 0
    )

    covered_mask = (
        covered_by_sensor.any(
            axis=1
        )
    )

    covered_nodes = int(
        covered_mask.sum()
    )

    total_nodes = (
        matrix.shape[0]
    )

    coverage_percent = (
        covered_nodes
        / total_nodes
        * 100.0
    )

    coverage_counts = (
        covered_by_sensor.sum(
            axis=1
        )
    )

    redundant_assignments = int(
        np.maximum(
            coverage_counts - 1,
            0,
        ).sum()
    )

    individual_sensor_coverage = (
        covered_by_sensor.sum(
            axis=0
        )
        / total_nodes
        * 100.0
    )

    return {
        "covered_nodes":
            covered_nodes,

        "coverage_percent":
            float(
                coverage_percent
            ),

        "redundant_assignments":
            redundant_assignments,

        "mean_sensor_coverage_percent":
            float(
                individual_sensor_coverage.mean()
            ),

        "selected_sensor_count":
            len(sensors),

        "valid_sensor_count":
            len(valid_sensors),

        "missing_sensor_count":
            len(missing_sensors),
    }


# ============================================================================
# BUILD DQN COVERAGE DATASET
# ============================================================================

def build_dqn_coverage():

    print()
    print(
        "=" * 75
    )

    print(
        "RECONSTRUCTING DQN COVERAGE"
    )

    print(
        "=" * 75
    )

    records = []

    matrix_cache = {}

    for network in NETWORKS:

        for seed in SEEDS:

            sensors = (
                load_dqn_sensor_selection(
                    network,
                    seed,
                )
            )

            print(
                f"{network:<10} "
                f"seed {seed}: "
                f"sensors={sensors}"
            )

            if len(sensors) != SENSOR_BUDGET:

                print(
                    "WARNING: "
                    f"{network} seed {seed} "
                    f"contains {len(sensors)} sensors."
                )

            for radius in COVERAGE_RADII:

                key = (
                    network,
                    radius,
                )

                if key not in matrix_cache:

                    matrix_cache[
                        key
                    ] = load_coverage_matrix(
                        network,
                        radius,
                    )

                matrix, node_ids = (
                    matrix_cache[
                        key
                    ]
                )

                metrics = (
                    calculate_coverage(
                        sensors,
                        matrix,
                        node_ids,
                    )
                )

                records.append(
                    {
                        "method":
                            "DQN",

                        "network":
                            network,

                        "seed":
                            seed,

                        "coverage_radius":
                            radius,

                        "selected_sensor_count":
                            metrics[
                                "selected_sensor_count"
                            ],

                        "covered_nodes":
                            metrics[
                                "covered_nodes"
                            ],

                        "coverage_percent":
                            metrics[
                                "coverage_percent"
                            ],

                        "redundant_assignments":
                            metrics[
                                "redundant_assignments"
                            ],

                        "mean_sensor_coverage_percent":
                            metrics[
                                "mean_sensor_coverage_percent"
                            ],

                        "valid_sensor_count":
                            metrics[
                                "valid_sensor_count"
                            ],

                        "missing_sensor_count":
                            metrics[
                                "missing_sensor_count"
                            ],
                    }
                )

    return pd.DataFrame(
        records
    )


# ============================================================================
# LOAD PPO/DDPG/GA COVERAGE
# ============================================================================

def load_existing_coverage(
    method,
    path,
):

    print()
    print(
        f"Loading {method} coverage..."
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired {method} coverage file "
            f"was not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    print(
        f"Loaded {len(df)} rows."
    )

    # ------------------------------------------------------------------------
    # Standardise column names.
    # ------------------------------------------------------------------------

    rename_map = {}

    if (
        "coverage_percentage"
        in df.columns
    ):

        rename_map[
            "coverage_percentage"
        ] = "coverage_percent"

    if (
        "mean_coverage_percent"
        in df.columns
        and
        "coverage_percent"
        not in df.columns
    ):

        rename_map[
            "mean_coverage_percent"
        ] = "coverage_percent"

    if rename_map:

        df = df.rename(
            columns=rename_map
        )

    required = [
        "network",
        "seed",
        "coverage_radius",
        "coverage_percent",
    ]

    missing = [
        x
        for x in required
        if x not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{method} coverage file is missing "
            f"columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------------------
    # Add missing secondary metrics.
    # ------------------------------------------------------------------------

    if "selected_sensor_count" not in df.columns:

        df[
            "selected_sensor_count"
        ] = SENSOR_BUDGET

    if "covered_nodes" not in df.columns:

        df[
            "covered_nodes"
        ] = np.nan

    if "redundant_assignments" not in df.columns:

        df[
            "redundant_assignments"
        ] = np.nan

    if (
        "mean_sensor_coverage_percent"
        not in df.columns
    ):

        df[
            "mean_sensor_coverage_percent"
        ] = np.nan

    if "valid_sensor_count" not in df.columns:

        df[
            "valid_sensor_count"
        ] = SENSOR_BUDGET

    if "missing_sensor_count" not in df.columns:

        df[
            "missing_sensor_count"
        ] = 0

    df[
        "method"
    ] = method

    columns = [
        "method",
        "network",
        "seed",
        "coverage_radius",
        "selected_sensor_count",
        "covered_nodes",
        "coverage_percent",
        "redundant_assignments",
        "mean_sensor_coverage_percent",
        "valid_sensor_count",
        "missing_sensor_count",
    ]

    return df[
        columns
    ].copy()


# ============================================================================
# LOAD ALL FOUR METHODS
# ============================================================================

def load_all_methods():

    dqn = (
        build_dqn_coverage()
    )

    ppo = load_existing_coverage(
        "PPO",
        PPO_ROOT
        / "ppo_coverage_seed_results.csv",
    )

    ddpg = load_existing_coverage(
        "DDPG",
        DDPG_ROOT
        / "ddpg_coverage_seed_results.csv",
    )

    ga = load_existing_coverage(
        "GA",
        GA_ROOT
        / "ga_coverage_seed_results.csv",
    )

    combined = pd.concat(
        [
            dqn,
            ppo,
            ddpg,
            ga,
        ],
        ignore_index=True,
    )

    return combined


# ============================================================================
# VALIDATION
# ============================================================================

def validate_dataset(
    df,
):

    print()
    print(
        "=" * 75
    )

    print(
        "FOUR-METHOD DATA VALIDATION"
    )

    print(
        "=" * 75
    )

    expected_each = (
        len(NETWORKS)
        * len(SEEDS)
        * len(COVERAGE_RADII)
    )

    expected_total = (
        len(METHODS)
        * expected_each
    )

    print()
    print(
        f"Expected observations per method: "
        f"{expected_each}"
    )

    print(
        f"Expected total observations: "
        f"{expected_total}"
    )

    print(
        f"Actual observations: "
        f"{len(df)}"
    )

    print()
    print(
        "Observations by method:"
    )

    print(
        df.groupby(
            "method"
        ).size()
    )

    print()
    print(
        "Observations by method/network:"
    )

    print(
        df.groupby(
            [
                "method",
                "network",
            ]
        ).size()
    )

    duplicates = (
        df.duplicated(
            subset=[
                "method",
                "network",
                "seed",
                "coverage_radius",
            ]
        )
        .sum()
    )

    print()
    print(
        f"Duplicate observations: "
        f"{duplicates}"
    )

    if duplicates:

        raise ValueError(
            "Duplicate observations detected."
        )

    missing = (
        df[
            "coverage_percent"
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing coverage values: "
        f"{missing}"
    )

    if missing:

        raise ValueError(
            "Missing coverage values detected."
        )

    print()

    if len(df) == expected_total:

        print(
            "TOTAL OBSERVATIONS: SUCCESS"
        )

    else:

        print(
            "WARNING: observation count differs "
            "from expected."
        )


# ============================================================================
# NETWORK/RADIUS SUMMARY
# ============================================================================

def create_network_radius_summary(
    df,
):

    return (
        df
        .groupby(
            [
                "method",
                "network",
                "coverage_radius",
            ],
            as_index=False,
        )
        .agg(
            number_of_evaluations=(
                "seed",
                "count",
            ),

            mean_coverage_percent=(
                "coverage_percent",
                "mean",
            ),

            std_coverage_percent=(
                "coverage_percent",
                "std",
            ),

            minimum_coverage_percent=(
                "coverage_percent",
                "min",
            ),

            maximum_coverage_percent=(
                "coverage_percent",
                "max",
            ),

            mean_covered_nodes=(
                "covered_nodes",
                "mean",
            ),

            mean_redundant_assignments=(
                "redundant_assignments",
                "mean",
            ),

            std_redundant_assignments=(
                "redundant_assignments",
                "std",
            ),

            mean_sensor_coverage_percent=(
                "mean_sensor_coverage_percent",
                "mean",
            ),

            mean_selected_sensor_count=(
                "selected_sensor_count",
                "mean",
            ),
        )
    )


# ============================================================================
# NETWORK SUMMARY
# ============================================================================

def create_network_summary(
    df,
):

    return (
        df
        .groupby(
            [
                "method",
                "network",
            ],
            as_index=False,
        )
        .agg(
            number_of_evaluations=(
                "coverage_percent",
                "count",
            ),

            mean_coverage_percent=(
                "coverage_percent",
                "mean",
            ),

            std_coverage_percent=(
                "coverage_percent",
                "std",
            ),

            minimum_coverage_percent=(
                "coverage_percent",
                "min",
            ),

            maximum_coverage_percent=(
                "coverage_percent",
                "max",
            ),

            mean_covered_nodes=(
                "covered_nodes",
                "mean",
            ),

            mean_redundant_assignments=(
                "redundant_assignments",
                "mean",
            ),

            std_redundant_assignments=(
                "redundant_assignments",
                "std",
            ),
        )
    )


# ============================================================================
# RADIUS SUMMARY
# ============================================================================

def create_radius_summary(
    df,
):

    return (
        df
        .groupby(
            [
                "method",
                "coverage_radius",
            ],
            as_index=False,
        )
        .agg(
            number_of_evaluations=(
                "coverage_percent",
                "count",
            ),

            mean_coverage_percent=(
                "coverage_percent",
                "mean",
            ),

            std_coverage_percent=(
                "coverage_percent",
                "std",
            ),

            minimum_coverage_percent=(
                "coverage_percent",
                "min",
            ),

            maximum_coverage_percent=(
                "coverage_percent",
                "max",
            ),

            mean_redundant_assignments=(
                "redundant_assignments",
                "mean",
            ),

            std_redundant_assignments=(
                "redundant_assignments",
                "std",
            ),
        )
    )


# ============================================================================
# PAIRED METHOD COMPARISON
# ============================================================================

def paired_comparison(
    df,
):

    records = []

    # Focus on the complete seed-matched data.

    comparison_pairs = [
        ("DQN", "PPO"),
        ("DQN", "DDPG"),
        ("DQN", "GA"),
    ]

    for network in NETWORKS:

        for radius in COVERAGE_RADII:

            subset = df[
                (
                    df["network"]
                    == network
                )
                &
                (
                    df[
                        "coverage_radius"
                    ]
                    == radius
                )
            ]

            for method_a, method_b in (
                comparison_pairs
            ):

                a = (
                    subset[
                        subset["method"]
                        == method_a
                    ]
                    .set_index(
                        "seed"
                    )[
                        "coverage_percent"
                    ]
                )

                b = (
                    subset[
                        subset["method"]
                        == method_b
                    ]
                    .set_index(
                        "seed"
                    )[
                        "coverage_percent"
                    ]
                )

                common = (
                    a.index.intersection(
                        b.index
                    )
                )

                a = a.loc[
                    common
                ]

                b = b.loc[
                    common
                ]

                difference = (
                    a - b
                )

                mean_difference = (
                    difference.mean()
                )

                std_difference = (
                    difference.std(
                        ddof=1
                    )
                )

                if (
                    std_difference
                    and
                    not np.isnan(
                        std_difference
                    )
                ):

                    effect_size = (
                        mean_difference
                        / std_difference
                    )

                else:

                    effect_size = np.nan

                records.append(
                    {
                        "network":
                            network,

                        "coverage_radius":
                            radius,

                        "method_a":
                            method_a,

                        "method_b":
                            method_b,

                        "paired_seed_count":
                            len(common),

                        "mean_method_a":
                            a.mean(),

                        "mean_method_b":
                            b.mean(),

                        "mean_difference":
                            mean_difference,

                        "std_difference":
                            std_difference,

                        "cohens_dz":
                            effect_size,

                        "higher_mean_method":
                            (
                                method_a
                                if a.mean()
                                > b.mean()
                                else method_b
                                if b.mean()
                                > a.mean()
                                else "Equal"
                            ),
                    }
                )

    return pd.DataFrame(
        records
    )


# ============================================================================
# FIGURE: NETWORK COMPARISONS
# ============================================================================

def generate_network_figures(
    summary,
):

    for network in NETWORKS:

        plt.figure(
            figsize=(10, 6)
        )

        subset = summary[
            summary[
                "network"
            ]
            == network
        ]

        for method in METHODS:

            method_data = (
                subset[
                    subset[
                        "method"
                    ]
                    == method
                ]
                .sort_values(
                    "coverage_radius"
                )
            )

            plt.errorbar(
                method_data[
                    "coverage_radius"
                ],
                method_data[
                    "mean_coverage_percent"
                ],
                yerr=method_data[
                    "std_coverage_percent"
                ].fillna(0),
                marker="o",
                capsize=4,
                label=method,
            )

        plt.xlabel(
            "Coverage radius (hops)"
        )

        plt.ylabel(
            "Mean coverage (%)"
        )

        plt.title(
            f"Four-Method Coverage Comparison — "
            f"{network}"
        )

        plt.xticks(
            COVERAGE_RADII
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()

        filename = (
            "four_method_coverage_"
            f"{network.replace('-', '_')}.png"
        )

        plt.savefig(
            FIGURE_ROOT / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================================
# FIGURE: RADIUS COMPARISON
# ============================================================================

def generate_radius_figure(
    summary,
):

    plt.figure(
        figsize=(10, 6)
    )

    for method in METHODS:

        data = (
            summary[
                summary[
                    "method"
                ]
                == method
            ]
            .sort_values(
                "coverage_radius"
            )
        )

        plt.errorbar(
            data[
                "coverage_radius"
            ],
            data[
                "mean_coverage_percent"
            ],
            yerr=data[
                "std_coverage_percent"
            ].fillna(0),
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xlabel(
        "Coverage radius (hops)"
    )

    plt.ylabel(
        "Mean coverage (%)"
    )

    plt.title(
        "Overall Four-Method Coverage "
        "Across Coverage Radii"
    )

    plt.xticks(
        COVERAGE_RADII
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_ROOT
        / "four_method_coverage_by_radius.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================================
# FIGURE: SEED VARIABILITY
# ============================================================================

def generate_seed_figure(
    df,
):

    radius = 1

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
    )

    for ax, network in zip(
        axes,
        NETWORKS,
    ):

        subset = df[
            (
                df[
                    "network"
                ]
                == network
            )
            &
            (
                df[
                    "coverage_radius"
                ]
                == radius
            )
        ]

        for method in METHODS:

            data = (
                subset[
                    subset[
                        "method"
                    ]
                    == method
                ]
                .sort_values(
                    "seed"
                )
            )

            ax.plot(
                data[
                    "seed"
                ],
                data[
                    "coverage_percent"
                ],
                marker="o",
                label=method,
            )

        ax.set_title(
            network
        )

        ax.set_xlabel(
            "Seed"
        )

        ax.set_ylabel(
            "Coverage (%)"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

    axes[-1].legend()

    fig.suptitle(
        "Seed-Level Coverage Variability "
        "(Coverage Radius = 1)"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_ROOT
        / "four_method_seed_variability.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================================
# FIGURE: OVERALL METHOD COMPARISON
# ============================================================================

def generate_overall_figure(
    df,
):

    summary = (
        df
        .groupby(
            "method"
        )[
            "coverage_percent"
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reindex(
            METHODS
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    x = np.arange(
        len(METHODS)
    )

    plt.bar(
        x,
        summary[
            "mean"
        ],
        yerr=summary[
            "std"
        ].fillna(0),
        capsize=5,
    )

    plt.xticks(
        x,
        METHODS,
    )

    plt.ylabel(
        "Mean coverage (%)"
    )

    plt.title(
        "Overall Four-Method "
        "Sensor-Placement Coverage"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_ROOT
        / "four_method_overall_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================================
# FIGURE: REDUNDANCY
# ============================================================================

def generate_redundancy_figure(
    summary,
):

    subset = summary[
        summary[
            "coverage_radius"
        ]
        == 1
    ]

    pivot = subset.pivot(
        index="network",
        columns="method",
        values="mean_redundant_assignments",
    )

    pivot = pivot.reindex(
        NETWORKS
    )

    pivot = pivot[
        [
            method
            for method in METHODS
            if method in pivot.columns
        ]
    ]

    ax = pivot.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_xlabel(
        "Network"
    )

    ax.set_ylabel(
        "Mean redundant assignments"
    )

    ax.set_title(
        "Four-Method Redundancy "
        "(Coverage Radius = 1)"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_ROOT
        / "four_method_redundancy.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================================
# JSON
# ============================================================================

def dataframe_to_records(
    df,
):

    return (
        df
        .replace(
            {
                np.nan:
                    None
            }
        )
        .to_dict(
            orient="records"
        )
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    create_directories()

    print()
    print("=" * 75)
    print("EXPERIMENT 16")
    print("FOUR-METHOD SENSOR-PLACEMENT COMPARISON")
    print("=" * 75)

    print()
    print(
        "DQN coverage will be reconstructed from "
        "existing per-seed sensor-selection files."
    )

    print()
    print(
        "Loading all four methods..."
    )

    combined = (
        load_all_methods()
    )

    # ------------------------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------------------------

    validate_dataset(
        combined
    )

    # ------------------------------------------------------------------------
    # MASTER DATASET
    # ------------------------------------------------------------------------

    master_path = (
        OUTPUT_ROOT
        / "four_method_coverage_comparison.csv"
    )

    combined.to_csv(
        master_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # SUMMARIES
    # ------------------------------------------------------------------------

    network_radius_summary = (
        create_network_radius_summary(
            combined
        )
    )

    network_summary = (
        create_network_summary(
            combined
        )
    )

    radius_summary = (
        create_radius_summary(
            combined
        )
    )

    statistical_summary = (
        paired_comparison(
            combined
        )
    )

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    network_summary_path = (
        OUTPUT_ROOT
        / "four_method_network_summary.csv"
    )

    radius_summary_path = (
        OUTPUT_ROOT
        / "four_method_radius_summary.csv"
    )

    statistical_path = (
        OUTPUT_ROOT
        / "four_method_statistical_comparison.csv"
    )

    network_summary.to_csv(
        network_summary_path,
        index=False,
    )

    radius_summary.to_csv(
        radius_summary_path,
        index=False,
    )

    statistical_summary.to_csv(
        statistical_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # FIGURES
    # ------------------------------------------------------------------------

    print()
    print(
        "Generating figures..."
    )

    generate_network_figures(
        network_radius_summary
    )

    generate_radius_figure(
        radius_summary
    )

    generate_seed_figure(
        combined
    )

    generate_overall_figure(
        combined
    )

    generate_redundancy_figure(
        network_radius_summary
    )

    # ------------------------------------------------------------------------
    # OVERALL SUMMARY
    # ------------------------------------------------------------------------

    overall_summary = (
        combined
        .groupby(
            "method",
            as_index=False,
        )
        .agg(
            number_of_observations=(
                "coverage_percent",
                "count",
            ),

            mean_coverage_percent=(
                "coverage_percent",
                "mean",
            ),

            std_coverage_percent=(
                "coverage_percent",
                "std",
            ),

            minimum_coverage_percent=(
                "coverage_percent",
                "min",
            ),

            maximum_coverage_percent=(
                "coverage_percent",
                "max",
            ),

            mean_redundant_assignments=(
                "redundant_assignments",
                "mean",
            ),

            std_redundant_assignments=(
                "redundant_assignments",
                "std",
            ),
        )
    )

    # ------------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------------

    json_path = (
        OUTPUT_ROOT
        / "four_method_comparison.json"
    )

    json_data = {

        "experiment":
            "Experiment 16",

        "description":
            (
                "Unified comparison of DQN, PPO, "
                "DDPG and GA sensor-placement coverage."
            ),

        "methods":
            METHODS,

        "networks":
            NETWORKS,

        "seeds":
            SEEDS,

        "coverage_radii":
            COVERAGE_RADII,

        "sensor_budget":
            SENSOR_BUDGET,

        "total_observations":
            int(
                len(
                    combined
                )
            ),

        "dqn_coverage_source":
            (
                "Existing per-seed "
                "evaluation_selected_sensors.csv "
                "files"
            ),

        "overall_summary":
            dataframe_to_records(
                overall_summary
            ),

        "network_summary":
            dataframe_to_records(
                network_summary
            ),

        "radius_summary":
            dataframe_to_records(
                radius_summary
            ),

        "statistical_comparison":
            dataframe_to_records(
                statistical_summary
            ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_data,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # TERMINAL OUTPUT
    # ------------------------------------------------------------------------

    print()
    print("=" * 75)
    print("OVERALL FOUR-METHOD SUMMARY")
    print("=" * 75)

    print()

    print(
        overall_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 75)
    print("NETWORK-LEVEL SUMMARY")
    print("=" * 75)

    print()

    print(
        network_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 75)
    print("RADIUS-LEVEL SUMMARY")
    print("=" * 75)

    print()

    print(
        radius_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 75)
    print("DQN PAIRWISE COMPARISONS")
    print("=" * 75)

    print()

    dqn_stats = statistical_summary[
        statistical_summary[
            "method_a"
        ]
        == "DQN"
    ]

    print(
        dqn_stats.to_string(
            index=False
        )
    )

    print()
    print("=" * 75)
    print("EXPERIMENT 16 COMPLETE")
    print("=" * 75)

    print()
    print("Saved:")

    print(
        master_path
    )

    print(
        network_summary_path
    )

    print(
        radius_summary_path
    )

    print(
        statistical_path
    )

    print(
        json_path
    )

    print()
    print(
        "Figures saved:"
    )

    print(
        FIGURE_ROOT
    )

    print()
    print("=" * 75)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()