from utils.randomizers.randomizers_helpers import compute_individual_synapse_lengths


import matplotlib.pyplot as plt
import numpy as np

from .plot_config import RANDOMIZATION_NAMES, get_randomization_colors


def plot_synapse_length_distributions(neuron_coords, conns_dict, use_density=True, num_confidence_interval_se=1):
    """
    Plot synapse length distributions for multiple network types.

    Parameters:
    -----------
    neuron_coords : DataFrame
        Contains neuron coordinates
    conns_dict : dict
        Dictionary of network types with their connection DataFrames
    use_density : bool, default=True
        Whether to normalize histograms to density
    num_confidence_interval_se : int, default=1
        Number of standard errors for confidence interval bands

    Returns:
    --------
    tuple: (fig1, fig2) - Two figure objects for histogram and synapse strength vs distance
    """

    titles = list(conns_dict.keys())
    n_plots = len(titles)

    # Ensure we have no more than 6 plots
    if n_plots > 6:
        raise ValueError(f"Too many networks to plot ({n_plots}). Maximum supported is 6.")

    # Pre-calculate distances for each dataframe
    dists = {name: compute_individual_synapse_lengths(df, neuron_coords)
            for name, df in conns_dict.items()}
    weights = {name: df["syn_count"].to_numpy()
              for name, df in conns_dict.items()}

    # Get 99 percentile of all distances to avoid outliers
    all_d = np.concatenate(list(dists.values()))
    max_len = np.percentile(all_d, 99)
    bins = np.linspace(0, max_len, 100)

    # Get common y-max for all plots
    max_val = 0
    for name in titles:
        hist, _ = np.histogram(dists[name], bins=bins,
                              weights=weights[name], density=use_density)
        max_val = max(max_val, hist.max())
    max_val *= 1.1  # Add a small margin

    # ——— Figure 1: Histogram of distance distribution ———
    fig1, axs1 = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots),
                             sharex=True, constrained_layout=True)

    # Ensure axs1 is always iterable (when n_plots=1, axs1 is a single Axes object)
    if n_plots == 1:
        axs1 = [axs1]

    total_mm = {}  # Total wiring lengths for annotation

    for ax, title in zip(axs1, titles):
        w = weights[title]
        L = dists[title]
        ax.hist(L, bins=bins, weights=w, density=use_density,
               color=get_randomization_colors(title), alpha=0.7)

        # Weighted mean
        mean_nm = np.average(L, weights=w)
        ax.axvline(mean_nm, ls='--', c='k', lw=1)
        # Display mean in µm
        ax.text(mean_nm*1.05, 0.7*max_val,
               f"Mean: {mean_nm / 1e3:,.2f} µm")

        # Total wiring length (m)
        tot_nm = float(np.sum(L * w))
        tot_m = tot_nm / 1e12
        total_mm[title] = tot_m
        ax.text(0.95, 0.85, f"Total: {tot_m:,.2f} km",
               transform=ax.transAxes, ha='right',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_ylim(0, max_val)
        ax.set_ylabel("Density" if use_density else "Count")
        ax.set_title(RANDOMIZATION_NAMES.get(title, title))

    axs1[-1].set_xlabel("Synapse Length (nm)")

    # ——— Figure 2: Synapse strength vs distance ———
    # Create bins for distance ranges
    bin_edges = np.linspace(0, max_len, 20)  # Fewer bins for better statistics
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig2, axs2 = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots),
                             sharex=True, constrained_layout=True)

    # Ensure axs2 is always iterable
    if n_plots == 1:
        axs2 = [axs2]

    for ax, title in zip(axs2, titles):
        L = dists[title]
        w = weights[title]

        # Compute statistics for each bin
        means = []
        errors = []

        for i in range(len(bin_edges) - 1):
            mask = (L >= bin_edges[i]) & (L < bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_weights = w[mask]
                mean_weight = np.mean(bin_weights)
                # Standard error = std / sqrt(n)
                std_err = np.std(bin_weights) / np.sqrt(len(bin_weights))
                means.append(mean_weight)
                errors.append(std_err)
            else:
                means.append(0)
                errors.append(0)

        # Plot the mean line
        ax.plot(bin_centers, means, 'o-', color=get_randomization_colors(title), markersize=5, alpha=0.9, label=title)

        # Add confidence interval bands
        upper_bound = [m + e * num_confidence_interval_se for m, e in zip(means, errors)]
        lower_bound = [m - e * num_confidence_interval_se for m, e in zip(means, errors)]
        ax.fill_between(bin_centers, lower_bound, upper_bound, color=get_randomization_colors(title), alpha=0.2)

        ax.set_ylabel("Avg. Synapse Count")
        ax.set_title(RANDOMIZATION_NAMES.get(title, title))
        ax.grid(True, linestyle='--', alpha=0.3)

    axs2[-1].set_xlabel("Synapse Length (nm)")
    plt.tight_layout()

    return fig1, fig2


# NOTE: `figsize` can now be left as `None` to automatically scale the
# figure height according to the number of subplots so that each histogram
# takes up more vertical space. If `figsize` is provided explicitly it will
# be honoured as before.
def plot_synapse_counts_histogram(conns_dict, bins=30, figsize=None, log_scale=False):
    """
    Plot simple histograms of synapse counts for each network type.

    Parameters:
    -----------
    conns_dict : dict
        Dictionary of network types with their connection DataFrames
    bins : int or list, default=30
        Number of bins or bin edges for histogram
    figsize : tuple, default=(12, 8)
        Figure size (width, height)
    log_scale : bool, default=False
        Whether to use log scale for y-axis

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with histograms
    """

    titles = list(conns_dict.keys())
    n_plots = len(titles)

    # Ensure we have no more than 6 plots
    if n_plots > 6:
        raise ValueError(f"Too many networks to plot ({n_plots}). Maximum supported is 6.")

    # Determine a sensible default figure size if none is provided.
    # Roughly allocate 2.5 vertical inches per subplot so the bars look tall
    # enough while keeping the width fixed at 12 inches (same as other
    # plotting functions in this module).
    if figsize is None:
        figsize = (12, 2.5 * n_plots)

    # --- Compute a common Y-max so every subplot uses the full vertical extent ---
    max_val = 0
    for title in titles:
        # Build histogram purely to get the tallest bar height
        hist_vals, _ = np.histogram(conns_dict[title]["syn_count"].values, bins=bins)
        max_val = max(max_val, hist_vals.max())

    # Add a small margin on top
    max_val *= 1.1

    # Create figure with subplots (one per network)
    fig, axs = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, constrained_layout=True)

    # Ensure axs is always iterable (when n_plots=1, axs is a single Axes object)
    if n_plots == 1:
        axs = [axs]

    for ax, title in zip(axs, titles):
        # Get synapse counts for this network
        syn_counts = conns_dict[title]["syn_count"].values

        # Plot histogram
        ax.hist(syn_counts, bins=bins, color=get_randomization_colors(title), alpha=0.7)

        # Calculate statistics
        mean_count = np.mean(syn_counts)
        median_count = np.median(syn_counts)
        max_count = np.max(syn_counts)
        total_synapses = np.sum(syn_counts)

        # Add statistics as text
        stats_text = (f"Mean: {mean_count:.2f}\n"
                     f"Median: {median_count:.2f}\n"
                     f"Max: {max_count:.2f}\n"
                     f"Total: {total_synapses:,}")

        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Add title and labels
        ax.set_title(RANDOMIZATION_NAMES.get(title, title))
        ax.set_ylabel("Count")

        # Set log or linear scale and unify ylim so the bars occupy the available
        # vertical space consistently across all subplots.
        if log_scale:
            ax.set_yscale('log')
            # In log scale, the lower bound must be > 0.
            ax.set_ylim(1, max_val)
        else:
            ax.set_ylim(0, max_val)

    # Add x-label to bottom subplot only
    axs[-1].set_xlabel("Synapse Count")

    return fig
