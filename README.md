# Comparative Evaluation of Deep Reinforcement Learning and Genetic Algorithms for Sensor Placement in Water Distribution Networks

## Overview

This repository contains research code, experimental outputs, and statistical analyses for comparing four sensor-placement approaches in water distribution networks:

- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)
- Deep Deterministic Policy Gradient (DDPG)
- Genetic Algorithm (GA)

The study evaluates graph-based sensor placement using the Net3, C-Town, and Richmond benchmark networks.

## Experimental Design

- **Networks:** Net3, C-Town, and Richmond
- **Algorithms:** DQN, PPO, DDPG, and GA
- **Sensor budget:** Five sensors per placement
- **Random seeds:** 42–51
- **Spatial coverage radii:** One to four graph hops
- **Coverage observations:** 480 method-specific records across four methods, three networks, ten seeds, and four radii

The four-method coverage comparison combines existing algorithm evaluation outputs. Experiment 16 performs this comparison without retraining the algorithms.

## Evaluation

The research examines:

1. Spatial coverage at different graph-hop radii.
2. Graph-based leak-event detection.
3. Graph-based contamination-event detection.
4. Combined event-detection performance.
5. Computational performance and sensor redundancy.
6. DQN ablation experiments examining experience replay and the target network.

## Statistical Analysis

The analysis includes paired comparisons, Wilcoxon signed-rank tests where applicable, effect-size estimates, and Holm–Bonferroni correction for multiple comparisons.

The corrected paired statistical analysis uses the balanced experimental design across networks, coverage radii, and random seeds. Consult the relevant analysis scripts and output files for the precise comparison, sample size, and correction applied to each outcome.

## Repository Structure

The repository will contain the selected source code, processed research data, statistical outputs, figures, and documentation needed to understand the published experiments.

Only files relevant to the study should be included in the public research release.

## Reproducibility

The repository is being prepared to support reproducibility. Before running an experiment, consult its script documentation and verify that the required input datasets and network files are available.

The four-method comparison and corrected statistical analysis use existing evaluation outputs; they do not retrain the algorithms.

Python dependencies will be documented in equirements.txt after the dependencies of the selected scripts have been checked.

## Important Limitations

- Spatial coverage is defined using graph-hop neighbourhoods.
- Event detectability is based on the study's graph-based evaluation procedure.
- Graph-based detection results should not be interpreted as independent hydraulic or contaminant-transport validation.
- The algorithms were not evaluated under identical computational budgets merely because their experiment counts or generation counts were similar.
- Results should be interpreted in the context of the tested networks, sensor budget, random seeds, and evaluation procedures.

## License

The original code covered by this repository is released under the MIT License. Third-party components and datasets remain subject to their applicable licences and usage conditions.

## Citation

A permanent citation and DOI will be added after the research repository has been published and its versioned release has been archived.

## Author

Clinton Blakk
