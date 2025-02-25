import importlib
import sys
import os
from pathlib import Path


def get_config():
    """Import and return the config module."""
    try:
        # Try direct import first (if installed properly)
        return importlib.import_module("configs.config")
    except ImportError:
        # Find project root as fallback
        current_dir = Path(os.getcwd())
        project_root = current_dir
        while (
            not (project_root / "setup.py").exists()
            and project_root != project_root.parent
        ):
            project_root = project_root.parent

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Try again after modifying path
        return importlib.import_module("configs.config")
