# Optuna Script Versions

This project contains several iterations of the `auto_optuna` script plus a legacy playbook. The table below summarizes their purpose and evolution.

| Script | Description |
| ------ | ----------- |
| `battle_tested_optuna_playbook.py` | Original proof-of-concept pipeline combining preprocessing, model selection and Optuna hyperparameter tuning. Superseded by later versions. |
| `auto_optuna-V1.py` | First major refactor of the playbook with improved dataset handling and more flexible preprocessing. |
| `auto_optuna-V1.1.py` | Adds a two-phase optimization (fast model selection followed by deep tuning). Useful when quick winner identification is needed. |
| `auto_optuna-V1.2.py` | Introduces a systematic three-phase approach: model family tournament, preprocessing stack search and deep hyperparameter optimization. Serves as the baseline for v1.3. |
| `auto_optuna-V1.3.py` | Current recommended version. Extends v1.2 with a centralized config dictionary and modular callbacks. It imports and extends v1.2 under the hood to avoid code duplication. |

For new experiments, start with **`auto_optuna-V1.3.py`**. Earlier versions are kept for reference and for reproducing older results.
