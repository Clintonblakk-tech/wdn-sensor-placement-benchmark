# Reproducibility Guide

## 1. Research Overview

This repository supports a comparative evaluation of four sensor-placement approaches for water distribution networks:

- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)
- Deep Deterministic Policy Gradient (DDPG)
- Genetic Algorithm (GA)

The benchmark networks are Net3, C-Town, and Richmond.

## 2. Experimental Design

The spatial-coverage evaluation uses four methods, three networks, ten random seeds (42–51), and four graph-hop coverage radii.

The expected number of method-specific coverage records is:

4 methods x 3 networks x 10 seeds x 4 radii = 480 records.

The sensor-placement budget is five sensors per method and experimental condition.

## 3. Repository Contents

### data/

Contains the consolidated coverage observations, coverage summaries, event-detection results, DQN ablation results, and balanced paired dataset.

### results/

Contains corrected descriptive statistics, pairwise statistical comparisons, statistical summaries, and DQN ablation figures.

### source_code/

Contains scripts for four-method coverage comparison, corrected statistical analysis, event-detection evaluation, detection statistics, and the genetic algorithm.

## 4. Software Requirements

The preliminary Python dependency list is provided in the repository root as requirements.txt.

The listed versions reflect the development environment. A clean installation and full end-to-end reproduction have not yet been verified.

## 5. Analysis Workflow

1. Review the README and the relevant experiment script.
2. Install compatible Python dependencies in an isolated environment.
3. Confirm that all input files required by the selected script are present.
4. Run the relevant evaluation or statistical-analysis script.
5. Compare the generated output with the corresponding published CSV or JSON results.

The comparison and statistical-analysis scripts use existing experiment outputs and do not necessarily retrain the algorithms.

## 6. Statistical Analysis

The corrected paired analysis uses Wilcoxon signed-rank tests, rank-biserial effect sizes, and Holm–Bonferroni correction where specified by the analysis.

Consult the individual scripts and their outputs for the precise test design and comparison groups.

## 7. Interpretation and Limitations

Coverage is measured using graph-hop neighbourhoods. Event detection follows the graph-based evaluation procedure used in the study.

These measures do not, by themselves, establish performance under hydraulic simulation or contaminant-transport modelling.

The results apply to the networks, sensor budget, random seeds, and evaluation procedures tested in the study.

## 8. Data and Code Licensing

The original code covered by the repository is released under the MIT License. Third-party code, network models, and datasets remain subject to their own licences and usage conditions.

## 9. Reproducibility Status

This guide documents the intended analysis workflow. The selected scripts, dependencies, input requirements, and output consistency should be checked before claiming complete end-to-end reproducibility.
