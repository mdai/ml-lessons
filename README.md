# ml-lessons

Deep learning/ML lessons for medical images. 

## Setup

## Repository Guidelines

All subdirectories should be named by task (e.g., `pneumothorax` or `chest_tubes`). For any modular code that is shared between the tasks, use `shared`.

### `data/`

All input data, including images and label/annotation json files should be kept in this directory. Download image archive to `data/` and unzip within this directory. All scripts/notebooks will use relative path to this directory. Preprocessed data should be kept here as well. Note: directory is ignored by git.

### `notebooks/`

Keep all notebooks within this directory.

### `scripts/`

Keep all python scripts here. Please follow python best-practices (PEP8, modularity, etc.). See [this repo](https://github.com/carpedm20/BEGAN-tensorflow) for a good example of a clean setup. Setting up a separate variable/parameter configuration system is essential for clarity and reproducibility. Any configuration setup is fine so long as it encourages these goals (argparse, python module, json, yaml, etc.). Such a setup is also essential for doing efficient hyperparameter search.

### `logs/`

All python script outputs (logs, model checkpoints, etc.) should be kept here. Note that this directory is ignored by git, but all outputs should be kept for reproducibility and guiding experimental direction. It should be clear where these files were produced from (i.e., script, configs, date/time).

---

&copy; 2018 MD.ai, Inc.
