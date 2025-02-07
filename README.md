# Connectome Project

## Environment Setup

This guide will help you set up the Python environment required to run this project.

### Prerequisites

- Python 3.10
- CUDA 12.4 (if using GPU)

### Setting up the Virtual Environment

1. Clone the repository and navigate to the project directory:
```bash
cd connectome
```

2. Create and activate a new virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
# Install PyTorch first manually following https://pytorch.org/get-started/locally/
# Then, the rest of the requirements
pip install -r requirements
```

### Verifying the Installation

You can verify your installation by running:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import torch_scatter; print('torch_scatter imported successfully')"
```

### Common Issues

- If you get permission errors when creating the virtual environment, ensure you have the required packages:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-venv python3-pip
  ```

- Make sure to match the CUDA version exactly when installing torch-scatter. Using incorrect CUDA versions can lead to undefined symbol errors.

### Create training data

ANS:
```bash
python create_input_images/points_creator.py --img_dir images/one_to_ten/train --easy
python create_input_images/points_creator.py --img_dir images/one_to_ten/test --easy
```

Shapes:
```bash
python create_input_images/shapes_creator.py --shapes
```
Colors:

```bash
python create_input_images/shapes_creator.py --colors
```

--------------------


Instructions for when cuda breaks

```{bash}
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

