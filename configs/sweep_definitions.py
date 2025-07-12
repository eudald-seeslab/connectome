from __future__ import annotations

"""Centralised collection of W&B sweep configurations.

Each entry in ``SWEEP_DEFS`` is a self-contained sweep dictionary that can be
fed directly to ``wandb.sweep``.

The goal is to keep *scripts/sweep.py* free of bulky configuration data so that
execution logic and experiment definitions live in separate modules.
"""

from utils.helpers import load_cell_type_lists

# -----------------------------------------------------------------------------
# Cell-type list from shared utils
# -----------------------------------------------------------------------------

_cell_types, _ = load_cell_type_lists()

# -----------------------------------------------------------------------------
# Sweep definitions
# -----------------------------------------------------------------------------

SWEEP_DEFS: dict[str, dict] = {}

# 1. General hyper-parameter sweep (formerly sweep_config1)
SWEEP_DEFS["general"] = {
    "method": "bayes",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "NUM_CONNECTOME_PASSES": {"values": [3, 4, 5, 6]},
        "neurons": {"values": ["selected", "all"]},
        "voronoi_criteria": {"values": ["R7", "all"]},
        "random_synapses": {"values": [True, False]},
        "eye": {"values": ["left", "right"]},
        "train_edges": {"values": [True, False]},
        "train_neurons": {"values": [True, False]},
        "final_layer": {"values": ["mean", "nn"]},
    },
}

# 2. Cell-type dependent sweep (requires data)
if _cell_types:
    SWEEP_DEFS["celltype_scan"] = {
        "method": "bayes",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "filtered_celltypes": {"values": _cell_types},
        },
    }

# 3. Fractional filtering of neurons (formerly sweep_config3)
SWEEP_DEFS["filtered_fraction"] = {
    "method": "random",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "filtered_fraction": {"values": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    },
}

# 4. Dropout parameters (formerly sweep_config4)
SWEEP_DEFS["dropouts"] = {
    "method": "random",
    "metric": {"name": "Validation accuracy", "goal": "maximize"},
    "parameters": {
        "neuron_dropout": {"values": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        "decision_dropout": {"values": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    },
}

# 5. Regularisation / training strategy (formerly sweep_config5)
SWEEP_DEFS["regularisation"] = {
    "method": "random",
    "metric": {"name": "Validation accuracy", "goal": "maximize"},
    "parameters": {
        "train_neurons": {"values": [True, False]},
        "train_edges": {"values": [True, False]},
        "refined_synaptic_data": {"values": [True, False]},
        "final_layer": {"values": ["mean", "nn"]},
    },
}

# 6. Random seeds grid (formerly sweep_config_seeds)
SWEEP_DEFS["random_seeds"] = {
    "method": "grid",
    "parameters": {"random_seed": {"values": list(range(10))}},
}
