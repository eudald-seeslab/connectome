"""Toy randomization strategy figure utilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch


DEFAULT_WIDTH_ONLY_EDGES = [
    {"id": 0, "target": 1, "syn_count": 3},
    {"id": 1, "target": 4, "syn_count": 6},
    {"id": 2, "target": 6, "syn_count": 3},
    {"id": 3, "target": 9, "syn_count": 3},
    {"id": 4, "target": 19, "syn_count": 2},
]

DEFAULT_TITLES = {
    "biological": "Biological reference",
    "unconstrained": "Random unconstrained",
    "connection_pruned": "Random connection-pruned",
    "random_binned": "Random bin-wise",
    "width_only": "Width-only reassignment",
}


def _build_state():
    n_targets = 20
    source_pos = np.array([0.0, 0.0])

    rng = np.random.default_rng(12)
    target_x = np.sort(rng.uniform(1.4, 10.0, n_targets))
    y_centers = rng.choice([-2.9, -1.4, 0.6, 2.4], size=n_targets, p=[0.2, 0.35, 0.3, 0.15])
    target_y = y_centers + rng.normal(0.0, 0.42, n_targets)
    target_pos = np.column_stack([target_x, target_y])

    edge_cmap = plt.get_cmap("tab10")
    edge_colors = [edge_cmap(i % 10) for i in range(20)]

    return {
        "n_targets": n_targets,
        "source_pos": source_pos,
        "target_x": target_x,
        "target_y": target_y,
        "target_pos": target_pos,
        "edge_colors": edge_colors,
    }


def _connection_probability(x_values):
    x_scaled = x_values / x_values.max()
    return np.clip(0.9 * np.exp(-3.8 * x_scaled**2) + 0.02, 0.02, 0.9)


def _build_original_edges(state, seed=5, min_edges=5):
    local_rng = np.random.default_rng(seed)
    probs = _connection_probability(state["target_x"])

    near_pool = np.argsort(probs)[-10:]
    near_weights = probs[near_pool] / probs[near_pool].sum()
    near_targets = local_rng.choice(near_pool, size=min_edges - 1, replace=False, p=near_weights)

    long_target = int(np.argmax(state["target_x"]))
    selected = np.sort(np.concatenate([near_targets, [long_target]]))

    x_scaled = state["target_x"][selected] / state["target_x"].max()
    mean_syn = 1.0 + 4.8 * (1.0 - x_scaled) ** 2
    syn_counts = np.clip(local_rng.poisson(mean_syn) + 1, 1, 6)

    long_pos = int(np.where(selected == long_target)[0][0])
    syn_counts[long_pos] = max(2, min(3, syn_counts[long_pos]))

    return [
        {"id": edge_id, "target": int(target_idx), "syn_count": int(syn_count)}
        for edge_id, (target_idx, syn_count) in enumerate(zip(selected, syn_counts))
    ]


def _clone_edges(edges):
    return [dict(edge) for edge in edges]


def _edge_length(edge, state):
    return float(np.linalg.norm(state["target_pos"][edge["target"]] - state["source_pos"]))


def _randomize_unconstrained(edges, state, seed=0):
    rng = np.random.default_rng(seed)
    randomized = _clone_edges(edges)

    available_targets = np.array(
        sorted(set(range(state["n_targets"])) - {edge["target"] for edge in edges}),
        dtype=int,
    )
    farthest = available_targets[np.argsort(state["target_x"][available_targets])[-len(edges):]]
    chosen_targets = farthest[rng.permutation(len(farthest))]

    edge_order = np.argsort([_edge_length(edge, state) for edge in edges])
    for edge_idx, new_target in zip(edge_order, chosen_targets):
        randomized[edge_idx]["target"] = int(new_target)

    return randomized


def _prune_connections(unconstrained_edges, keep_fraction=0.6, seed=0):
    rng = np.random.default_rng(seed)
    n_remove = int(round(len(unconstrained_edges) * (1.0 - keep_fraction)))
    remove_idx = set(rng.choice(len(unconstrained_edges), size=n_remove, replace=False).tolist())
    return [dict(edge) for idx, edge in enumerate(unconstrained_edges) if idx not in remove_idx]


def _randomize_binned(edges, state, seed=0):
    rng = np.random.default_rng(seed)
    randomized = _clone_edges(edges)

    original_targets = {edge["target"] for edge in edges}
    available_targets = list(sorted(set(range(state["n_targets"])) - original_targets))
    available_distances = {
        target: float(np.linalg.norm(state["target_pos"][target] - state["source_pos"]))
        for target in available_targets
    }

    order = rng.permutation(len(edges))
    for edge_idx in order:
        original_distance = _edge_length(edges[edge_idx], state)
        ranked_targets = sorted(
            available_targets,
            key=lambda target: abs(available_distances[target] - original_distance),
        )
        pick_pool = ranked_targets[: min(3, len(ranked_targets))]
        new_target = int(rng.choice(pick_pool))
        randomized[edge_idx]["target"] = new_target
        available_targets.remove(new_target)

    return randomized


def _randomize_width_only(edges, seed=0, override=None):
    if override is not None:
        return _clone_edges(override)

    rng = np.random.default_rng(seed)
    randomized = _clone_edges(edges)
    syn_counts = np.array([edge["syn_count"] for edge in edges], dtype=int)
    shuffled = rng.permutation(syn_counts)
    if np.array_equal(shuffled, syn_counts) and len(shuffled) > 1:
        shuffled = np.roll(shuffled, 1)
    for edge, syn_count in zip(randomized, shuffled):
        edge["syn_count"] = int(syn_count)
    return randomized


def _strand_radii(edge, state):
    count = edge["syn_count"]
    _, ty = state["target_pos"][edge["target"]]
    base = 0.09 * np.sign(ty) if abs(ty) > 0.25 else 0.10
    spread = min(0.42, 0.055 * max(count - 1, 1))
    if count == 1:
        return [base]
    return list(base + np.linspace(-spread, spread, count))


def _draw_edge(ax, edge, state, alpha=1.0, zorder=2):
    sx, sy = state["source_pos"]
    tx, ty = state["target_pos"][edge["target"]]
    color = state["edge_colors"][edge["id"]]

    for rad in _strand_radii(edge, state):
        patch = FancyArrowPatch(
            (sx, sy),
            (tx, ty),
            arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}",
            linewidth=0.95,
            color=color,
            alpha=alpha,
            zorder=zorder,
            capstyle="round",
            joinstyle="round",
        )
        ax.add_patch(patch)


def _display_target_indices(edge_sets):
    return np.array(sorted({edge["target"] for edges in edge_sets for edge in edges}), dtype=int)


def _draw_nodes(ax, state, display_idx, show_source_label=True):
    display_pos = state["target_pos"][display_idx]

    ax.scatter(
        display_pos[:, 0],
        display_pos[:, 1],
        s=62,
        facecolor="white",
        edgecolor="#334155",
        linewidth=1.0,
        zorder=5,
    )

    ax.scatter(
        state["source_pos"][0],
        state["source_pos"][1],
        s=120,
        facecolor="white",
        edgecolor="#111827",
        linewidth=2.0,
        zorder=6,
    )

    if show_source_label:
        ax.text(
            state["source_pos"][0] - 0.12,
            state["source_pos"][1] - 0.42,
            "source",
            ha="left",
            va="top",
            fontsize=10.5,
            color="#111827",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.8),
        )


def _style_axis(ax, state, display_idx):
    display_x = state["target_x"][display_idx]
    display_y = state["target_y"][display_idx]

    ax.set_xlim(-0.45, display_x.max() + 0.7)
    ax.set_ylim(display_y.min() - 1.0, display_y.max() + 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    for x in np.arange(0, np.ceil(display_x.max()) + 0.5, 1):
        ax.axvline(x, color="#F8FAFC", lw=0.8, zorder=0)


def _draw_panel(ax, edges, title, state, display_idx, show_source_label=True, panel_label=None):
    _style_axis(ax, state, display_idx)
    for edge in sorted(edges, key=lambda edge: edge["syn_count"]):
        _draw_edge(ax, edge, state, alpha=0.95, zorder=2)
    _draw_nodes(ax, state, display_idx, show_source_label=show_source_label)
    if panel_label is not None:
        ax.text(
            -0.02,
            1.02,
            panel_label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="#111827",
        )
    ax.text(
        0.05,
        1.02,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        # fontweight="bold",
        color="#111827",
    )


def plot_randomization_strategies_subfigure(
    subfig,
    *,
    width_only_edges_override=None,
    titles=None,
    panel_labels=None,
    show_source_label=True,
    apply_theme=False,
):
    if apply_theme:
        sns.set_theme(font_scale=1.4, style="white")

    state = _build_state()
    titles = DEFAULT_TITLES if titles is None else titles

    original_edges = _build_original_edges(state)
    unconstrained_edges = _randomize_unconstrained(original_edges, state, seed=23)
    connection_pruned_edges = _prune_connections(unconstrained_edges, keep_fraction=0.6, seed=24)
    binned_edges = _randomize_binned(original_edges, state, seed=41)
    width_only_edges = _randomize_width_only(
        original_edges,
        seed=52,
        override=DEFAULT_WIDTH_ONLY_EDGES if width_only_edges_override is None else width_only_edges_override,
    )

    edge_sets = [
        original_edges,
        unconstrained_edges,
        connection_pruned_edges,
        binned_edges,
        width_only_edges,
    ]
    display_idx = _display_target_indices(edge_sets)

    gs = subfig.add_gridspec(
        3,
        4,
        hspace=0.28,
        wspace=0.20,
        left=0.06,
        right=0.99,
        top=0.97,
        bottom=0.06,
    )
    ax_biological = subfig.add_subplot(gs[0, 1:3])
    ax_unconstrained = subfig.add_subplot(gs[1, 0:2], sharex=ax_biological, sharey=ax_biological)
    ax_pruned = subfig.add_subplot(gs[1, 2:4], sharex=ax_biological, sharey=ax_biological)
    ax_binned = subfig.add_subplot(gs[2, 0:2], sharex=ax_biological, sharey=ax_biological)
    ax_width = subfig.add_subplot(gs[2, 2:4], sharex=ax_biological, sharey=ax_biological)

    panels = [
        (ax_biological, original_edges, titles["biological"]),
        (ax_unconstrained, unconstrained_edges, titles["unconstrained"]),
        (ax_pruned, connection_pruned_edges, titles["connection_pruned"]),
        (ax_binned, binned_edges, titles["random_binned"]),
        (ax_width, width_only_edges, titles["width_only"]),
    ]

    panel_labels = [None] * len(panels) if panel_labels is None else panel_labels

    for (ax, edges, title), panel_label in zip(panels, panel_labels):
        _draw_panel(
            ax,
            edges,
            title,
            state,
            display_idx,
            show_source_label=show_source_label,
            panel_label=panel_label,
        )
        ax.spines[["left", "top", "right"]].set_visible(False)

    for ax in [ax_biological, ax_binned, ax_width]:
        ax.set_xlabel("Distance from source neuron", color="black", fontsize=16)

    return {
        "biological": ax_biological,
        "unconstrained": ax_unconstrained,
        "connection_pruned": ax_pruned,
        "random_binned": ax_binned,
        "width_only": ax_width,
    }


def create_randomization_strategies_figure(
    *,
    figure_size=(13.5, 14.5),
    width_only_edges_override=None,
    titles=None,
    show_source_label=True,
    apply_theme=True,
):
    fig = plt.figure(figsize=figure_size)
    plot_randomization_strategies_subfigure(
        fig,
        width_only_edges_override=width_only_edges_override,
        titles=titles,
        show_source_label=show_source_label,
        apply_theme=apply_theme,
    )
    return fig
