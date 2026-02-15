"""Connectome research analysis package.

Re-exports commonly used symbols from trainyourfly for convenience,
plus research-specific subpackages (visualization, model_inspection,
randomizers).
"""

import os
import sys
from pathlib import Path

# Re-export core model classes from trainyourfly
from trainyourfly.data.data_processing import DataProcessor
from trainyourfly.connectome_models.graph_models import FullGraphModel, Connectome
from connectome.model_inspection.manifold_funcs import store_intermediate_output
from trainyourfly.utils.utils import (
    clean_model_outputs,
    get_image_paths,
    get_iteration_number,
    initialize_results_df,
    select_random_images,
    update_results_df,
    update_running_loss,
)


def get_config():
    """Import and return the config module."""
    import importlib

    try:
        return importlib.import_module("configs.config")
    except ImportError:
        current_dir = Path(os.getcwd())
        project_root = current_dir
        while (
            not (project_root / "setup.py").exists()
            and project_root != project_root.parent
        ):
            project_root = project_root.parent

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        return importlib.import_module("configs.config")


def setup_notebook(use_project_root_as_cwd=False):
    """Setup function for notebooks to properly resolve paths.

    Args:
        use_project_root_as_cwd: If True, changes the working directory
            to the project root.

    Returns:
        Path object pointing to the project root.
    """
    notebook_dir = Path.cwd()
    project_root = notebook_dir

    while (
        not (project_root / "setup.py").exists()
        and project_root != project_root.parent
    ):
        project_root = project_root.parent

    if project_root == project_root.parent:
        raise RuntimeError("Could not find project root (directory with setup.py)")

    if use_project_root_as_cwd:
        os.chdir(project_root)
        print(f"Changed working directory to {project_root}")

    print(f"Project root: {project_root}")
    return project_root
