"""Configuration for randomization plots."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns


def darken_color(color: str, factor: float = 0.6) -> str:
    """
    Return a darker version of a color.
    
    Parameters
    ----------
    color : str
        Color in any format matplotlib accepts (hex, name, etc.)
    factor : float
        Darkening factor (0 = black, 1 = original color)
    """
    rgb = mcolors.to_rgb(color)
    dark_rgb = tuple(c * factor for c in rgb)
    return mcolors.to_hex(dark_rgb)


T = TypeVar("T")


@dataclass(frozen=True)
class RandomizationSpec:
    key: str
    label: str
    color: str
    enabled: bool = True


class RandomizationCatalog:
    """Single source of truth for randomization metadata."""

    def __init__(self, specs: tuple[RandomizationSpec, ...], aliases: dict[str, str] | None = None):
        self._specs = specs
        self._aliases = aliases or {}
        self._by_key = {spec.key: spec for spec in specs}
        self._by_label = {spec.label: spec for spec in specs}

    @property
    def specs(self) -> tuple[RandomizationSpec, ...]:
        return self._specs

    @property
    def enabled_specs(self) -> tuple[RandomizationSpec, ...]:
        return tuple(spec for spec in self._specs if spec.enabled)

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in self.enabled_specs)

    @property
    def labels_in_order(self) -> tuple[str, ...]:
        return tuple(spec.label for spec in self.enabled_specs)

    @property
    def names(self) -> dict[str, str]:
        return {spec.key: spec.label for spec in self._specs}

    @property
    def colors(self) -> dict[str, str]:
        return {spec.key: spec.color for spec in self._specs}

    def resolve_key(self, key_or_label: str) -> str:
        if key_or_label in self._aliases:
            return self._aliases[key_or_label]
        if key_or_label in self._by_key:
            return key_or_label
        if key_or_label in self._by_label:
            return self._by_label[key_or_label].key
        raise KeyError(f"Unknown randomization '{key_or_label}'.")

    def get(self, key_or_label: str) -> RandomizationSpec:
        return self._by_key[self.resolve_key(key_or_label)]

    def label_for(self, key_or_label: str) -> str:
        return self.get(key_or_label).label

    def color_for(self, key_or_label: str) -> str:
        return self.get(key_or_label).color

    def enabled_for(self, key_or_label: str) -> bool:
        return self.get(key_or_label).enabled

    def get_enabled(self, items: Mapping[str, T]) -> dict[str, T]:
        enabled_items: dict[str, T] = {}

        for spec in self.enabled_specs:
            for item_key in (spec.label, spec.key):
                if item_key in items:
                    enabled_items[item_key] = items[item_key]
                    break

        for item_key, value in items.items():
            if item_key in enabled_items:
                continue
            try:
                if self.enabled_for(item_key):
                    enabled_items[item_key] = value
            except KeyError:
                enabled_items[item_key] = value

        return enabled_items

    def sort_keys(self, keys: list[str] | tuple[str, ...]) -> list[str]:
        order_index = {key: idx for idx, key in enumerate(spec.key for spec in self._specs)}
        return sorted(keys, key=lambda key: order_index.get(self.resolve_key(key), len(order_index)))

    def sort_labels(self, labels: list[str] | tuple[str, ...]) -> list[str]:
        order_index = {label: idx for idx, label in enumerate(spec.label for spec in self._specs)}
        return sorted(labels, key=lambda label: order_index.get(self.label_for(label), len(order_index)))


RANDOMIZATIONS = RandomizationCatalog(
    specs=(
        RandomizationSpec("biological", "Biological", "#4c6ef5", enabled=True),
        RandomizationSpec("unconstrained", "Unconstrained", "#8b1e1e", enabled=True),
        RandomizationSpec("random_pruned", "Random pruned", "#b23a48", enabled=False),
        RandomizationSpec("connection_pruned", "Connection-pruned", "#e07a5f", enabled=True),
        RandomizationSpec("random_binned", "Binned", "#cfae3b", enabled=True),
        RandomizationSpec("neuron_binned", "Neuron binned", "#e9c46a", enabled=True),
    ),
    aliases={
        "binned": "random_binned",
    },
)

RANDOMIZATION_NAMES = RANDOMIZATIONS.names
RANDOMIZATION_COLORS = RANDOMIZATIONS.colors


def get_randomization_colors(randomization_name: str) -> str:
    """Get the color for a randomization strategy by its display name."""
    return RANDOMIZATIONS.color_for(randomization_name)


PLOT_STYLE_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 1,
    "grid.linewidth": 0.5,
    "lines.linewidth": 2,
    "lines.markersize": 5,
}


def apply_plot_style(overrides: dict | None = None):
    """Apply the centralized matplotlib rcParams style."""
    sns.set_theme(style="white", font_scale=1.4)
    style = PLOT_STYLE_PARAMS.copy()
    if overrides:
        style.update(overrides)
    mpl.rcParams.update(style)


def split_title(title: str, max_length: int = 15) -> str:
    """Split long titles for better display in plots."""
    if title == "Unconstrained":
        return "Un-\nconstrained"
    if len(title) > max_length:
        return title.replace("-", "\n").replace(" ", "\n")
    return title

