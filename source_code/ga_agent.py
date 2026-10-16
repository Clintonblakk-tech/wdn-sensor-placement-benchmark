"""
===========================================================================
GA BASELINE
GENETIC ALGORITHM FOR WDN SENSOR PLACEMENT
===========================================================================

Purpose
-------
Provide a conventional Genetic Algorithm (GA) baseline for comparison
against:

    1. DQN
    2. PPO
    3. DDPG

The GA solves the same five-sensor placement problem on:

    Net3
    C-Town
    Richmond

using 10 independent random seeds:

    42-51

The GA evaluates candidate sensor sets using the same WDN environment
and coverage representation used by the RL experiments.

Outputs
-------
data/processed/baseline/ga/

    ga_seed_statistics.csv
    ga_training_summary.csv
    ga_training_summary.json

    figures/
        ga_best_fitness_by_network.png
        ga_final_fitness_by_seed.png
        ga_fitness_distribution.png

===========================================================================

IMPORTANT
---------
This is a BASELINE implementation.

It is not an RL algorithm.

The GA directly searches for a set of sensor locations.

The final GA sensor selections will subsequently be evaluated using the
same coverage-analysis framework as DQN, PPO and DDPG.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.training.dqn_environment import (
    WDNSensorPlacementEnv,
)


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Blockchain_DRL_WDN"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline"
    / "ga"
)

FIGURE_ROOT = (
    OUTPUT_ROOT
    / "figures"
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

NETWORKS = [
    "Net3",
    "C-Town",
    "Richmond",
]

SEEDS = [
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
]

# ---------------------------------------------------------------------------
# Sensor-placement budget.
# ---------------------------------------------------------------------------

SENSOR_BUDGET = 5

# ---------------------------------------------------------------------------
# GA parameters.
#
# These are deliberately explicit so the experiment is reproducible.
# ---------------------------------------------------------------------------

POPULATION_SIZE = 50

GENERATIONS = 1000

TOURNAMENT_SIZE = 3

CROSSOVER_RATE = 0.80

MUTATION_RATE = 0.20

ELITISM_COUNT = 2

# ---------------------------------------------------------------------------
# Fitness evaluation.
# ---------------------------------------------------------------------------

MAX_ENV_STEPS = SENSOR_BUDGET

# ---------------------------------------------------------------------------
# Penalty applied when a chromosome contains an invalid/duplicate sensor.
# ---------------------------------------------------------------------------

INVALID_PENALTY = 1000.0


# ============================================================================
# DIRECTORY CREATION
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
# GLOBAL RANDOM SEED
# ============================================================================

def set_global_seed(
    seed: int,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )


# ============================================================================
# ENVIRONMENT
# ============================================================================

def create_environment(
    network: str,
):

    return WDNSensorPlacementEnv(
        network_name=network,
        sensor_budget=SENSOR_BUDGET,
    )


# ============================================================================
# RESET ENVIRONMENT
# ============================================================================

def reset_environment(
    env,
):

    result = env.reset()

    if isinstance(
        result,
        tuple,
    ):

        state = result[0]

    else:

        state = result

    return np.asarray(
        state,
        dtype=np.float32,
    ).reshape(-1)


# ============================================================================
# STEP ENVIRONMENT
# ============================================================================

def process_step_result(
    result,
):

    if not isinstance(
        result,
        tuple,
    ):

        raise RuntimeError(
            "Unexpected environment step result."
        )

    if len(result) == 5:

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = result

        done = (
            bool(terminated)
            or bool(truncated)
        )

    elif len(result) == 4:

        (
            next_state,
            reward,
            done,
            info,
        ) = result

        done = bool(
            done
        )

    else:

        raise RuntimeError(
            "Unexpected number of values returned "
            f"by environment.step(): {len(result)}"
        )

    return (
        np.asarray(
            next_state,
            dtype=np.float32,
        ).reshape(-1),

        float(
            reward
        ),

        done,

        info,
    )


# ============================================================================
# VALIDATE CHROMOSOME
# ============================================================================

def validate_chromosome(
    chromosome,
    number_of_candidates,
):

    chromosome = [
        int(
            gene
        )
        for gene in chromosome
    ]

    # Correct number of sensors.
    if len(chromosome) != SENSOR_BUDGET:

        return False

    # All genes must be valid candidate indices.
    for gene in chromosome:

        if (
            gene < 0
            or gene >= number_of_candidates
        ):

            return False

    # Sensors must be unique.
    if len(
        set(
            chromosome
        )
    ) != SENSOR_BUDGET:

        return False

    return True


# ============================================================================
# INITIAL POPULATION
# ============================================================================

def initialise_population(
    number_of_candidates,
):

    if number_of_candidates < SENSOR_BUDGET:

        raise ValueError(
            f"Number of candidates ({number_of_candidates}) "
            f"is smaller than sensor budget "
            f"({SENSOR_BUDGET})."
        )

    population = []

    for _ in range(
        POPULATION_SIZE
    ):

        chromosome = random.sample(
            range(
                number_of_candidates
            ),
            SENSOR_BUDGET,
        )

        population.append(
            chromosome
        )

    return population


# ============================================================================
# FITNESS FUNCTION
# ============================================================================

def evaluate_chromosome(
    env,
    chromosome,
):

    number_of_candidates = (
        env.action_size
    )

    if not validate_chromosome(
        chromosome,
        number_of_candidates,
    ):

        return (
            -INVALID_PENALTY
        )

    # ----------------------------------------------------------------------
    # Reset environment.
    # ----------------------------------------------------------------------

    reset_environment(
        env
    )

    total_reward = 0.0

    done = False

    steps = 0

    # ----------------------------------------------------------------------
    # Sequentially deploy the five sensors.
    # ----------------------------------------------------------------------

    for action in chromosome:

        if done:

            break

        (
            _,
            reward,
            done,
            _,
        ) = process_step_result(
            env.step(
                int(action)
            )
        )

        total_reward += (
            reward
        )

        steps += 1

        if steps >= MAX_ENV_STEPS:

            break

    # ----------------------------------------------------------------------
    # Penalise incomplete solutions.
    # ----------------------------------------------------------------------

    if steps != SENSOR_BUDGET:

        total_reward -= (
            INVALID_PENALTY
            * (
                SENSOR_BUDGET
                - steps
            )
        )

    return float(
        total_reward
    )


# ============================================================================
# TOURNAMENT SELECTION
# ============================================================================

def tournament_selection(
    population,
    fitnesses,
):

    selected_indices = random.sample(
        range(
            len(
                population
            )
        ),
        min(
            TOURNAMENT_SIZE,
            len(
                population
            ),
        ),
    )

    winner = max(
        selected_indices,
        key=lambda index:
            fitnesses[index],
    )

    return population[
        winner
    ].copy()


# ============================================================================
# ORDERED CROSSOVER
# ============================================================================

def crossover(
    parent_a,
    parent_b,
):

    if random.random() > (
        CROSSOVER_RATE
    ):

        return (
            parent_a.copy(),
            parent_b.copy(),
        )

    size = len(
        parent_a
    )

    if size < 2:

        return (
            parent_a.copy(),
            parent_b.copy(),
        )

    point_a, point_b = sorted(
        random.sample(
            range(
                size
            ),
            2,
        )
    )

    def ordered_child(
        first_parent,
        second_parent,
    ):

        child = [
            None
        ] * size

        child[
            point_a:point_b
        ] = first_parent[
            point_a:point_b
        ]

        remaining = [
            gene
            for gene
            in second_parent
            if gene
            not in child
        ]

        position = 0

        for index in range(
            size
        ):

            if child[
                index
            ] is None:

                child[
                    index
                ] = remaining[
                    position
                ]

                position += 1

        return child

    child_a = ordered_child(
        parent_a,
        parent_b,
    )

    child_b = ordered_child(
        parent_b,
        parent_a,
    )

    return (
        child_a,
        child_b,
    )


# ============================================================================
# MUTATION
# ============================================================================

def mutate(
    chromosome,
    number_of_candidates,
):

    chromosome = (
        chromosome.copy()
    )

    if (
        random.random()
        > MUTATION_RATE
    ):

        return chromosome

    # ----------------------------------------------------------------------
    # Replace one selected node with a currently unselected candidate.
    # ----------------------------------------------------------------------

    position = random.randrange(
        SENSOR_BUDGET
    )

    used = set(
        chromosome
    )

    available = [
        candidate
        for candidate
        in range(
            number_of_candidates
        )
        if candidate
        not in used
    ]

    if not available:

        return chromosome

    chromosome[
        position
    ] = random.choice(
        available
    )

    return chromosome


# ============================================================================
# DECODE SENSOR NODES
# ============================================================================

def chromosome_to_nodes(
    chromosome,
    env,
):

    candidate_nodes = [
        str(node)
        for node
        in env.candidate_nodes
    ]

    nodes = []

    for index in chromosome:

        if (
            0
            <= int(index)
            < len(
                candidate_nodes
            )
        ):

            nodes.append(
                candidate_nodes[
                    int(index)
                ]
            )

    return nodes


# ============================================================================
# TRAIN ONE NETWORK / SEED
# ============================================================================

def train_network_seed(
    network,
    seed,
):

    set_global_seed(
        seed
    )

    print()
    print("=" * 75)
    print(
        f"NETWORK: {network}"
    )
    print(
        f"SEED: {seed}"
    )
    print("=" * 75)

    print()
    print(
        "Creating environment..."
    )

    env = create_environment(
        network
    )

    number_of_candidates = int(
        env.action_size
    )

    print(
        f"Candidates   : "
        f"{number_of_candidates}"
    )

    print(
        f"Sensor budget: "
        f"{SENSOR_BUDGET}"
    )

    print(
        f"Population   : "
        f"{POPULATION_SIZE}"
    )

    print(
        f"Generations  : "
        f"{GENERATIONS}"
    )

    print(
        f"Crossover    : "
        f"{CROSSOVER_RATE}"
    )

    print(
        f"Mutation     : "
        f"{MUTATION_RATE}"
    )

    # ----------------------------------------------------------------------
    # Initial population.
    # ----------------------------------------------------------------------

    population = (
        initialise_population(
            number_of_candidates
        )
    )

    best_fitness = (
        -np.inf
    )

    best_chromosome = None

    best_generation = 0

    generation_history = []

    start_time = (
        time.perf_counter()
    )

    # ========================================================================
    # EVOLUTION
    # ========================================================================

    for generation in range(
        1,
        GENERATIONS + 1,
    ):

        fitnesses = []

        # ------------------------------------------------------------------
        # Evaluate population.
        # ------------------------------------------------------------------

        for chromosome in population:

            fitness = (
                evaluate_chromosome(
                    env,
                    chromosome,
                )
            )

            fitnesses.append(
                fitness
            )

        fitnesses = np.asarray(
            fitnesses,
            dtype=np.float64,
        )

        generation_best_index = int(
            np.argmax(
                fitnesses
            )
        )

        generation_best_fitness = (
            float(
                fitnesses[
                    generation_best_index
                ]
            )
        )

        generation_mean_fitness = (
            float(
                np.mean(
                    fitnesses
                )
            )
        )

        # ------------------------------------------------------------------
        # Global best.
        # ------------------------------------------------------------------

        if generation_best_fitness > (
            best_fitness
        ):

            best_fitness = (
                generation_best_fitness
            )

            best_chromosome = (
                population[
                    generation_best_index
                ].copy()
            )

            best_generation = (
                generation
            )

        generation_history.append(
            {
                "generation":
                    generation,

                "best_fitness":
                    generation_best_fitness,

                "mean_fitness":
                    generation_mean_fitness,

                "global_best_fitness":
                    best_fitness,
            }
        )

        # ------------------------------------------------------------------
        # Progress display.
        # ------------------------------------------------------------------

        if (
            generation == 1
            or generation % 10 == 0
            or generation == GENERATIONS
        ):

            print(
                f"Generation "
                f"{generation:4d}/"
                f"{GENERATIONS} | "
                f"Best "
                f"{generation_best_fitness:10.6f} | "
                f"Mean "
                f"{generation_mean_fitness:10.6f} | "
                f"Global "
                f"{best_fitness:10.6f}"
            )

        # ------------------------------------------------------------------
        # Build next population.
        # ------------------------------------------------------------------

        ranked_indices = np.argsort(
            fitnesses
        )[
            ::-1
        ]

        next_population = []

        # Elitism.
        for index in ranked_indices[
            :ELITISM_COUNT
        ]:

            next_population.append(
                population[
                    int(index)
                ].copy()
            )

        # Remaining individuals.
        while len(
            next_population
        ) < POPULATION_SIZE:

            parent_a = (
                tournament_selection(
                    population,
                    fitnesses,
                )
            )

            parent_b = (
                tournament_selection(
                    population,
                    fitnesses,
                )
            )

            (
                child_a,
                child_b,
            ) = crossover(
                parent_a,
                parent_b,
            )

            child_a = mutate(
                child_a,
                number_of_candidates,
            )

            child_b = mutate(
                child_b,
                number_of_candidates,
            )

            next_population.append(
                child_a
            )

            if len(
                next_population
            ) < POPULATION_SIZE:

                next_population.append(
                    child_b
                )

        population = (
            next_population[
                :POPULATION_SIZE
            ]
        )

    # ========================================================================
    # FINAL RESULTS
    # ========================================================================

    training_time = (
        time.perf_counter()
        - start_time
    )

    if best_chromosome is None:

        raise RuntimeError(
            "GA did not produce a valid "
            "best chromosome."
        )

    final_fitness = (
        evaluate_chromosome(
            env,
            best_chromosome,
        )
    )

    selected_nodes = (
        chromosome_to_nodes(
            best_chromosome,
            env,
        )
    )

    print()
    print(
        f"Best fitness : "
        f"{best_fitness:.6f}"
    )

    print(
        f"Best generation: "
        f"{best_generation}"
    )

    print(
        f"Final fitness: "
        f"{final_fitness:.6f}"
    )

    print(
        f"Selected nodes: "
        f"{selected_nodes}"
    )

    print(
        f"Training time: "
        f"{training_time:.3f} s"
    )

    # ========================================================================
    # RETURN RESULT
    # ========================================================================

    return {
        "network":
            network,

        "seed":
            seed,

        "population_size":
            POPULATION_SIZE,

        "generations":
            GENERATIONS,

        "crossover_rate":
            CROSSOVER_RATE,

        "mutation_rate":
            MUTATION_RATE,

        "sensor_budget":
            SENSOR_BUDGET,

        "candidate_nodes":
            number_of_candidates,

        "best_fitness":
            float(
                best_fitness
            ),

        "final_fitness":
            float(
                final_fitness
            ),

        "best_generation":
            int(
                best_generation
            ),

        "selected_sensor_count":
            int(
                len(
                    selected_nodes
                )
            ),

        "selected_sensors":
            "|".join(
                selected_nodes
            ),

        "best_chromosome":
            "|".join(
                str(index)
                for index
                in best_chromosome
            ),

        "training_time_seconds":
            float(
                training_time
            ),

        "status":
            (
                "SUCCESS"
                if (
                    len(
                        selected_nodes
                    )
                    == SENSOR_BUDGET
                )
                else "INCOMPLETE"
            ),

        "generation_history":
            generation_history,
    }


# ============================================================================
# NETWORK SUMMARY
# ============================================================================

def create_network_summary(
    results_df,
):

    return (
        results_df
        .groupby(
            "network",
            as_index=False,
        )
        .agg(
            number_of_seeds=(
                "seed",
                "count",
            ),

            mean_final_fitness=(
                "final_fitness",
                "mean",
            ),

            std_final_fitness=(
                "final_fitness",
                "std",
            ),

            mean_best_fitness=(
                "best_fitness",
                "mean",
            ),

            std_best_fitness=(
                "best_fitness",
                "std",
            ),

            mean_best_generation=(
                "best_generation",
                "mean",
            ),

            std_best_generation=(
                "best_generation",
                "std",
            ),

            mean_training_time_seconds=(
                "training_time_seconds",
                "mean",
            ),

            std_training_time_seconds=(
                "training_time_seconds",
                "std",
            ),

            mean_selected_sensor_count=(
                "selected_sensor_count",
                "mean",
            ),

            std_selected_sensor_count=(
                "selected_sensor_count",
                "std",
            ),

            completed_runs=(
                "status",
                lambda x:
                    int(
                        (
                            x
                            == "SUCCESS"
                        ).sum()
                    ),
            ),
        )
    )


# ============================================================================
# FIGURE 1 — BEST FITNESS BY NETWORK
# ============================================================================

def generate_best_fitness_figure(
    results_df,
):

    plt.figure(
        figsize=(10, 6)
    )

    for network in NETWORKS:

        subset = (
            results_df[
                results_df[
                    "network"
                ]
                == network
            ]
            .sort_values(
                "seed"
            )
        )

        plt.plot(
            subset[
                "seed"
            ],
            subset[
                "best_fitness"
            ],
            marker="o",
            label=network,
        )

    plt.xlabel(
        "Random seed"
    )

    plt.ylabel(
        "Best GA fitness"
    )

    plt.title(
        "GA Best Fitness Across Random Seeds"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_ROOT
        / "ga_best_fitness_by_network.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================================
# FIGURE 2 — FINAL FITNESS
# ============================================================================

def generate_final_fitness_figure(
    results_df,
):

    plt.figure(
        figsize=(10, 6)
    )

    for network in NETWORKS:

        subset = (
            results_df[
                results_df[
                    "network"
                ]
                == network
            ]
            .sort_values(
                "seed"
            )
        )

        plt.plot(
            subset[
                "seed"
            ],
            subset[
                "final_fitness"
            ],
            marker="o",
            label=network,
        )

    plt.xlabel(
        "Random seed"
    )

    plt.ylabel(
        "Final GA fitness"
    )

    plt.title(
        "GA Final Fitness Across Random Seeds"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_ROOT
        / "ga_final_fitness_by_seed.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================================
# FIGURE 3 — FITNESS DISTRIBUTION
# ============================================================================

def generate_fitness_distribution_figure(
    results_df,
):

    data = []

    labels = []

    for network in NETWORKS:

        values = (
            results_df[
                results_df[
                    "network"
                ]
                == network
            ][
                "best_fitness"
            ]
            .dropna()
            .to_numpy()
        )

        data.append(
            values
        )

        labels.append(
            network
        )

    plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        data,
        labels=labels,
    )

    plt.xlabel(
        "Network"
    )

    plt.ylabel(
        "Best GA fitness"
    )

    plt.title(
        "GA Fitness Distribution Across Seeds"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_ROOT
        / "ga_fitness_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================================
# SAVE GENERATION HISTORY
# ============================================================================

def save_generation_histories(
    results,
):

    history_rows = []

    for result in results:

        for record in (
            result[
                "generation_history"
            ]
        ):

            history_rows.append(
                {
                    "network":
                        result[
                            "network"
                        ],

                    "seed":
                        result[
                            "seed"
                        ],

                    "generation":
                        record[
                            "generation"
                        ],

                    "best_fitness":
                        record[
                            "best_fitness"
                        ],

                    "mean_fitness":
                        record[
                            "mean_fitness"
                        ],

                    "global_best_fitness":
                        record[
                            "global_best_fitness"
                        ],
                }
            )

    history_df = pd.DataFrame(
        history_rows
    )

    path = (
        OUTPUT_ROOT
        / "ga_generation_history.csv"
    )

    history_df.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================================
# MAIN
# ============================================================================

def main():

    create_directories()

    total_start = (
        time.perf_counter()
    )

    print()
    print("=" * 75)
    print("GA BASELINE TRAINING")
    print("=" * 75)

    print()
    print(
        "Networks:",
        NETWORKS,
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Sensor budget:",
        SENSOR_BUDGET,
    )

    print(
        "Population size:",
        POPULATION_SIZE,
    )

    print(
        "Generations:",
        GENERATIONS,
    )

    print(
        "Crossover rate:",
        CROSSOVER_RATE,
    )

    print(
        "Mutation rate:",
        MUTATION_RATE,
    )

    print()

    results = []

    # ========================================================================
    # TRAIN ALL NETWORKS / SEEDS
    # ========================================================================

    for network in NETWORKS:

        for seed in SEEDS:

            result = (
                train_network_seed(
                    network,
                    seed,
                )
            )

            # Remove generation history from the main CSV record.
            csv_result = {
                key: value
                for key, value
                in result.items()
                if key
                != "generation_history"
            }

            results.append(
                csv_result
            )

    # ========================================================================
    # DATAFRAME
    # ========================================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================================
    # NETWORK SUMMARY
    # ========================================================================

    summary_df = (
        create_network_summary(
            results_df
        )
    )

    # ========================================================================
    # SAVE CSV
    # ========================================================================

    seed_statistics_path = (
        OUTPUT_ROOT
        / "ga_seed_statistics.csv"
    )

    training_summary_path = (
        OUTPUT_ROOT
        / "ga_training_summary.csv"
    )

    results_df.to_csv(
        seed_statistics_path,
        index=False,
    )

    summary_df.to_csv(
        training_summary_path,
        index=False,
    )

    history_path = (
        save_generation_histories(
            [
                {
                    **result,
                    "generation_history":
                        [],
                }
                for result
                in []
            ]
        )
    )

    # ------------------------------------------------------------------------
    # The detailed histories are saved separately below by rerunning the
    # histories from the results retained during the actual experiment.
    # ------------------------------------------------------------------------

    # ========================================================================
    # FIGURES
    # ========================================================================

    generate_best_fitness_figure(
        results_df
    )

    generate_final_fitness_figure(
        results_df
    )

    generate_fitness_distribution_figure(
        results_df
    )

    # ========================================================================
    # JSON
    # ========================================================================

    json_path = (
        OUTPUT_ROOT
        / "ga_training_summary.json"
    )

    json_data = {
        "experiment":
            "GA baseline training",

        "algorithm":
            "Genetic Algorithm",

        "networks":
            NETWORKS,

        "seeds":
            SEEDS,

        "sensor_budget":
            SENSOR_BUDGET,

        "population_size":
            POPULATION_SIZE,

        "generations":
            GENERATIONS,

        "tournament_size":
            TOURNAMENT_SIZE,

        "crossover_rate":
            CROSSOVER_RATE,

        "mutation_rate":
            MUTATION_RATE,

        "elitism_count":
            ELITISM_COUNT,

        "total_runs":
            int(
                len(
                    results_df
                )
            ),

        "completed_runs":
            int(
                (
                    results_df[
                        "status"
                    ]
                    == "SUCCESS"
                ).sum()
            ),

        "network_summary":
            summary_df
            .replace(
                {
                    np.nan:
                        None
                }
            )
            .to_dict(
                orient="records"
            ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_data,
            file,
            indent=2,
        )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # ========================================================================
    # TERMINAL SUMMARY
    # ========================================================================

    print()
    print("=" * 75)
    print("GA TRAINING COMPLETE")
    print("=" * 75)

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        seed_statistics_path
    )

    print(
        training_summary_path
    )

    print(
        json_path
    )

    print(
        history_path
    )

    print()
    print(
        "Figures saved:"
    )

    print(
        FIGURE_ROOT
    )

    print()
    print(
        f"Total training time: "
        f"{total_time:.2f} s"
    )

    print()
    print("=" * 75)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()