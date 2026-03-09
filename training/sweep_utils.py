from __future__ import annotations

"""Utility helpers shared by sweep-related scripts."""

import sys
from types import SimpleNamespace

__all__ = ["validate_sweep_config"]


def _get_attr(obj, name, default=None):
    """Like getattr but supports both modules and SimpleNamespace."""
    if isinstance(obj, SimpleNamespace):
        return getattr(obj, name, default)
    return getattr(obj, name, default)


def validate_sweep_config(cfg, skip_checks: bool) -> None:
    """Abort execution if potentially incompatible debug/test settings are set.

    Parameters
    ----------
    cfg : module | SimpleNamespace
        Configuration object.
    skip_checks : bool
        When *True* all checks are bypassed.
    """
    if skip_checks:
        return

    fail_reasons: list[str] = []

    if _get_attr(cfg, "debugging", False):
        fail_reasons.append("debugging is enabled (debugging=True)")

    if _get_attr(cfg, "filtered_fraction", None) is not None:
        fail_reasons.append("filtered_fraction is not None")

    if not _get_attr(cfg, "wandb_", True):
        fail_reasons.append("W&B logging is disabled (wandb_=False)")

    if _get_attr(cfg, "small_length", None) is not None:
        fail_reasons.append("small_length is set (small_length is not None)")

    if fail_reasons:
        reasons = "\n  - ".join(fail_reasons)
        msg = (
            f"Sweep aborted by pre-flight checks:\n  - {reasons}\n"
            "Use --skip_checks to override and run anyway."
        )
        print(msg, file=sys.stderr)
        sys.exit(1) 