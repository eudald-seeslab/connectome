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
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
# Install PyTorch first
pip install -r requirements

# torch-scatter is a bit special
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
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

Coses a fer
[-] Agafar la posició més superficial
[x] Fer el canvi B i R
[x] Crear les cel·les de Voronoi amb centres en les R7 o R8
[-] Crear neurones acumuladores per veure si aprenen els números
[x] Mirar si hi ha alguna neurona que sempre ho endevina
[x] Posar millor els plans de la retina
[x] Mirar com evoluciona la imatge en les diferents capes del sistema visual
[x] Veure si es degrada randomitzant els edges

Idees
[x] Quins dos colors poden diferenciar millor les mosques
[x] Reshuffle pesos i no entrenar
[x] Treure neurones: entrenant o no entrenant
[x] Deixar fixes els pesos i mirar si sap alguna cosa: reservoir computing
[x] Treure neurones (sinapses) tipus a tipus
[x] Fer que el model entrenat per fer una cosa, en faci una altra
[x] Assegurar-se que les dues maneres de testejar són equivalents
[x] Analitzar les head direction neurons


[x] Refined matrix sense entrenar
[x] Refined matrix with restricted training (\theta  \in [0, 1])
[x] Head direction with single stripe
[x] Mirar la relació entre radi/distància/color i posició al manifold
[x] Augmentar el número de capes quan no surt
[x] mirar manifolds sobre el test set
[x] provar amb menys punts
[x] amb xarxa entrenada amb un color, veure si també funciona per altres colors
[-] transfer learning: entrenar a la vegada punts i símbols
[-] mnist
[x] Neuron subselection for decision making
[ ] Manifolds: see how neural representations evolve with training
[ ] Manifolds: check shapes trained only on edges

Paper:
- weber
- distingir coses
- manifolds: hi ha uns certs patrons que s'activen per unes raons i creen 

what can flies do?

query per la posició de les neurones: https://codex.flywire.ai/app/view_3d?query=cell_type+%7Bin%7D+R1-6%2CR7%2CR8+%7Band%7D+side+%7Bequal%7D+right&action=random_sample&color_by=Type&mode=3D+only