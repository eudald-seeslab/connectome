# _Drosophila melanogaster_'s connectome models

Research analysis repository for studying neural connectomes using machine learning. The core model and training infrastructure live in the [train-your-fly](https://github.com/eudald-seeslab/train-your-fly) package. This repo contains experiment orchestration, analysis notebooks, and visualization code.

This repo is organized to fit the needs for the experiments I'm running, so it's not worth reusing it. If you want to experiment with the training of the fruit fly, I would advise to study how it's done here, but create a new repo that better fits your needs. 

**Note:** If you are a reviewer and would like to replicate a paper results, here are the install instructions and code organization:

## Installation

1. Clone the repository:
```bash
git clone https://github.com/eudald-seeslab/connectome.git
cd connectome
```

2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

3. Install `train-your-fly` (editable, from local clone):

```bash
pip install -e /path/to/train-your-fly[wandb]
```

4. Install this package:

```bash
pip install -e .
```

With research tools (plotly, numba, umap, scikit-learn, etc.):

```bash
pip install -e .[research]
```

Full (research + dev):

```bash
pip install -e .[all]
```

## Repository structure

```
connectome/
├── connectome/              # Python package (importable library code)
│   ├── visualization/       # All visualization code
│   ├── model_inspection/    # Neuron-level inspection utilities
│   ├── randomizers/         # Biological vs random circuit comparison
│   └── data_helpers.py      # Data loading helpers
│
├── configs/                 # Experiment configuration
├── training/                # Training scripts (train, multitask, sweep)
├── random_networks/         # Random circuit analysis notebooks & scripts
├── manifolds/               # Manifold / representation analysis
├── model_inspection/        # Model introspection notebooks & scripts
├── figures/                 # Paper figure generation
├── data_processing/         # Data preparation notebooks
└── tests/                   # Tests
```

## Data preparation

Quick start:
```bash
pip install cogstim
cogstim shapes --train-num 60 --test-num 20
cogstim colours --train-num 60 --test-num 20 --no-jitter
cogstim ans --ratios easy --train-num 100 --test-num 40
```

For more information and all available tasks, see the
[cogstim documentation](https://github.com/eudald-seeslab/cogstim).

## Training

Adjust parameters in `configs/config.py`, then:

```bash
python training/train.py
```

For multitask training, also configure `configs/config_multitasking_dirs.py`:

```bash
python training/train_multitask.py
```

For sweeps:

```bash
python training/sweep.py --sweep regularisation
```

## Other

Some weird bug sometimes make CUDA break. You can fix it with:

```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```
