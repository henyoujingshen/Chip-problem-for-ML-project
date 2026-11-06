# Project guide

This repository collects machine-learning coursework and research experiments for chip-packaging optimization, including surrogate models, evolutionary search, stacking ensembles, and Bayesian optimization. It also includes experiments on mathematical benchmark functions.

The original [README](README.md) records algorithm changes. This guide adds a map of the code and practical notes for reproducing individual experiments.

## Repository map

| Location | Contents |
| --- | --- |
| `EA_XGB*.py` | XGBoost surrogate-assisted evolutionary optimization, with clustering and sampling variants. |
| `EA_Stacking_xgb-poly-knn.py`, `stacking_*.py` | Stacking-model experiments and supporting training code. |
| [`mode_test/`](mode_test/) | Individual regression-model experiments, with training and test CSV files. |
| [`stacking_continuous_v2/`](stacking_continuous_v2/) | Successive versions of stacking-based optimization on continuous test functions. |
| [`BSSO/`](BSSO/) | Versioned optimization experiments. Version 3.2 targets the discrete chip-packaging dataset; later versions also include benchmark and simulation workflows. |
| [`Sip_Opt/`](Sip_Opt/) | Bayesian-optimization scripts for discrete domains and associated CSV datasets. |
| `shit_mountain_v0/` | Historical prototypes and comparative experiments, including PSO and CMA-ES. The original directory name is retained for compatibility. |

This is a collection of experimental scripts rather than a single installable package. Select one experiment and its supporting files before preparing an environment.

## Suggested starting point

[`mode_test/XGB_test.py`](mode_test/XGB_test.py) is a useful entry point for understanding the regression workflow. It reads `model_train_data.csv` and `model_test_data.csv` from the current working directory, treats the last column as the target, trains an XGBoost regressor, displays prediction plots, and prints error metrics.

1. Create an isolated Python environment.
2. Install the packages imported by this script: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`, and `lhsmdu`.
3. Run from the `mode_test` directory so the relative data paths resolve:

   ```sh
   cd mode_test
   python XGB_test.py
   ```

These steps describe the source-code workflow; they are not a claim that the historical scripts have been validated against current package releases. Python and dependency compatibility should be checked in the chosen environment.

## Choosing other experiments

- **Discrete chip-packaging optimization:** begin with [`BSSO/BSSO v3.2/ReadMe.md`](BSSO/BSSO%20v3.2/ReadMe.md), then inspect `BSSO_discrete_test.py` and `BSSO_discrete_model.py` in that directory. The dataset `3output_5d.csv` is included alongside them.
- **XGBoost and evolutionary search:** inspect `EA_XGB_Cluster_OptPoint_v3.py` and the version notes in the root README. The script expects `3output_5d.csv` in its working directory, but that file is not present at the repository root. Copies exist under `Sip_Opt/` and `BSSO/BSSO v3.2/`; verify the expected column order and units before configuring a path.
- **Bayesian optimization:** inspect `Sip_Opt/bayesian_loop_2d_discrete_domain.py` or `Sip_Opt/bayesian_loop_5d_discrete_domain.py`. These use PyTorch, BoTorch, and GPyTorch. `Sip_Opt/main.py` is a placeholder, not a runnable project launcher.
- **Continuous benchmarks:** inspect the problem settings near the beginning of the selected `stacking_continuous_v2/` or `BSSO/` script. Different versions have different defaults and evaluation budgets.

## Data and environment notes

- Input conventions differ between experiments. Some scripts use all columns except the last as features; others select specific columns, rescale responses, or rely on a fixed ordering of a five-dimensional grid. Do not interchange CSV files without checking the corresponding loader.
- Many paths are relative to the current working directory. Keep paired optimization/model files together and check input and output locations before running.
- Depending on the experiment, additional imports include DEAP, SciPy, SMT, PySwarms, CMA-ES, PyDOE, PyTorch, BoTorch, GPyTorch, or GPyOpt. There is no repository-wide environment specification.
- `shit_mountain_v0/CMA-ES/requirements.txt` records historical dependency versions for that experiment only. It should not be treated as a complete environment for every directory.
- `BSSO/BSSO v3.3/chip_simulation.py` imports Abaqus modules and contains machine-specific Windows paths. That workflow requires an appropriate Abaqus installation and local configuration.
- Some scripts run substantial work at module level, and experiments can write model files, CSV results, or figures into the working directory. Read the selected script before importing or running it; preserve previous outputs in a separate run folder.

## Recording a reproducible run

For a result intended for comparison or reporting, record the commit, exact script, Python and package versions, dataset and preprocessing, random seeds, model settings, sampling strategy, and evaluation budget. Keep generated outputs with that record and distinguish results on stored datasets from runs that invoke a simulator.

Existing source files, historical version notes, and datasets remain unchanged by this documentation addition.
