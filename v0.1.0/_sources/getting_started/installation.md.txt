---
primary_layer: layer1
---

# Installation

## Requirements
- Python >= 3.10

See [pyproject.toml](https://github.com/saealib/saealib/blob/main/pyproject.toml) for the exact versions of the dependencies.

## Installing

Install with pip as follows.
This installs only the minimal dependencies.
```bash
pip install saealib
```
To install with all dependencies:
```bash
pip install "saealib[all]"
```
See [here](#install-options) for the available options.

## Specifying a version
Specify a version as follows.
```bash
pip install "saealib==X.Y.Z"
# example:
pip install "saealib==0.1.0"
```
Specify the latest pre-release version as follows.
```bash
pip install --pre saealib
```
:::{warning}
The API of pre-release versions may change or be removed without notice.
:::

(install-options)=
## Specifying options
Some packages are enabled by adding extra dependencies.
Install these dependencies at the same time by specifying options as follows.
```bash
pip install "saealib[opt1,opt2,...]"
# example:
pip install "saealib[sklearn,parallel]"
```
To install all dependencies:
```bash
pip install "saealib[all]"
```
All options are shown in the following table.
| Option | Adds |
|---|---|
| `sklearn` | scikit-learn-based surrogate models |
| `xgboost` | XGBoost surrogate model |
| `lightgbm` | LightGBM surrogate model |
| `torch` | PyTorch-based components |
| `pymoo` | `PymooCrossover`/`PymooMutation`/`PymooAlgorithm`/`PymooProblem` adapters |
| `parallel` | Parallel evaluation via joblib |
| `deap` | DEAP algorithm and operator adapters |
| `viz` | Matplotlib-based result and history plots |
| `tqdm` | tqdm progress bars for experiment sweeps |
| `rich` | Rich progress bars for experiment sweeps |
| `hdf5` | HDF5 trial-result storage and loading via h5py |
| `all` | All optional dependencies listed above |

## Verifying the installation
Once installation is complete, check the version with the following command.
```bash
python -c "import saealib; print(saealib.__version__)"
```

:::{seealso}
See [CONTRIBUTING.md](https://github.com/saealib/saealib/blob/main/CONTRIBUTING.md) for a development install from source.
:::
