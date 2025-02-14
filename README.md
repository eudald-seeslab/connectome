# Drosophila melanogaster's connectome models

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
source venv/bin/activate  # On Windows: venv\Scripts\activate
````

3. Install the packages

Install options:

Basic installation:

Torch stuff is special

```bash
pip install torch torchvision torchaudio
pip install torch-scatter torch-sparse
```

````bash
pip install -e .
````

With visualization tools

```bash
pip install .[viz]
```

Development installation

```bash
pip install .[dev]
```

Full installation:

```bash
pip install .[all]
```

## Data preparation

You can create input data using the scripts in 'input_data_creation', like such:

ANS:
```bash
python input_data_creation/points_creator.py --img_dir images/one_to_ten/train --easy
python input_data_creation/points_creator.py --img_dir images/one_to_ten/test --easy
```

Shapes:
```bash
python input_data_creation/shapes_creator.py --shapes
```
Colors:

```bash
python input_data_creation/shapes_creator.py --colors
```

You can also place your custom input data in the images directory. The structure of directories needs to be:
```
images/
├── your_directory/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── class_n/
    └── test/
        ├── class_1/
        ├── class_2/
        └── class_n/
```

And the model will automatically pick the number of output classes it needs to train on.

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

This project is in very early stages and highly unstable. If you would like to contribute, please write 
me at eudald.correig@urv.cat and we'll see how to organize it.

## TODOs

These runs sometimes make cuda break (I don't know why). As a temporary patch, when this happens, run:

```{bash}
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

