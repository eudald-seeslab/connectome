# _Drosophila melanogaster_'s connectome models

This repository contains code for analyzing neural connectomes using machine learning techniques. 
The work is associated with our paper [pending].

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/connectome.git
cd connectome
```

2. Create and activate a virtual environment

````bash
python -m venv venv
source venv/bin/activate  
# On Windows: 
# venv\Scripts\activate
````

3. Install the packages

Basic install:

````bash
pip install -e .
````

With visualization tools:

```bash
pip install .[viz]
```

Development:

```bash
pip install .[dev]
```

Full:

```bash
pip install .[all]
```

## Data preparation

Quick start:
```bash
pip install cogstim
# Shape recognition (circle vs star)
cogstim shapes --train-num 60 --test-num 20
# Colour recognition (yellow vs blue), no positional jitter
cogstim colours --train-num 60 --test-num 20 --no-jitter
# Approximate Number System (ANS), easy ratios
cogstim ans --ratios easy --train-num 100 --test-num 40
# Single-colour dot arrays (1–5), equalized total surface
cogstim one-colour --train-num 50 --test-num 20 --min-point-num 1 --max-point-num 5
# Rotated stripe patterns (lines)
cogstim lines --train-num 50 --test-num 20 --angles 0 45 90 135 --min-stripes 3 --max-stripes 5
```

For more information and all available tasks, see the [cogstim documentation](https://github.com/eudald-seeslab/cogstim).

Output directory structure remains:
```
images/your_directory/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── class_n/
└── test/
    ├── class_1/
    ├── class_2/
    └── class_n/
```

Models will automatically infer the number of output classes from this structure.

## Training

For training, you first need to adjust the training parameters in `configs/config.py`. The explanation of what each
parameter means will be given elsewhere.

```bash
python scripts/train.py
```
If you want to multitask, you need to still adjust the main parameters in config.py, but you also need to
specify the multiple sources of training data in `configs/config_multitasking_dirs.py`

## Analysis

Jupyter notebooks for analysis are in the notebooks/ directory:

- `analysis/`: Detailed analysis notebooks
- `exploration/`: Exploratory data analysis
- `visualization/`: Visualization notebooks
- `manifolds/`: Manifold analysis of neuron activations

## Contributing

This project is in very early stages and highly unstable. If you would like to contribute, please write me at eudald.correig@urv.cat and we'll see how to organize it.

## TODOs

- Easier creation of ANS input images

## Other

These runs sometimes make cuda break (I don't know why). As a temporary patch, when this happens, run:

```{bash}
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

