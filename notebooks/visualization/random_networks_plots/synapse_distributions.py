from utils.randomizers.randomizers_helpers import compute_individual_synapse_lengths


import matplotlib.pyplot as plt
import numpy as np

from .plot_config import RANDOMIZATION_NAMES, apply_plot_style, get_randomization_colors


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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, ScalarFormatter

def plot_overlay_wiring_distributions(neuron_coords, conns_dict,
                                      biological_key="Biological",
                                      use_density=True,
                                      xmax_percentile=99,
                                      bins=100,
                                      x_unit="nm",
                                      logy=False,
                                      fig_width_mm=90,
                                      font_size=8,
                                      show_mean_lines=True,
                                      mean_label=True,
                                      ax=None):
    """
    Overlay histogram of synapse lengths for multiple network types.
    
    Returns (ax, metrics) dict with totals and means.
    """

    apply_plot_style()

    names = list(conns_dict.keys())
    for name, df in conns_dict.items():
        if "__syn_len_nm" not in df.columns:
            df["__syn_len_nm"] = compute_individual_synapse_lengths(df, neuron_coords)

    d_nm = {name: df["__syn_len_nm"].to_numpy() for name, df in conns_dict.items()}
    w    = {n: np.asarray(conns_dict[n]["syn_count"], float) for n in names}

    totals_km, means_um = {}, {}
    for n in names:
        L = np.asarray(d_nm[n], float)
        ww = w[n]
        totals_km[n] = float(np.sum(L * ww) / 1e12)
        means_um[n]  = float(np.average(L, weights=ww) / 1e3)

    x_scale = 1.0 if x_unit == "nm" else 1e-3
    x_label = f"Synapse length ({'nm' if x_unit=='nm' else 'µm'})"
    d_plot = {n: np.asarray(d_nm[n], float) * x_scale for n in names}

    # Límits i bins comuns
    all_d = np.concatenate([d_plot[n] for n in names])
    xmax  = np.percentile(all_d, xmax_percentile)
    edges = np.linspace(0, xmax, int(bins)) if isinstance(bins, (int, float)) else \
            np.histogram_bin_edges(all_d[all_d <= xmax], bins=bins)

    # Helpers de color/etiqueta
    def label_of(n):
        return RANDOMIZATION_NAMES.get(n, n) if 'RANDOMIZATION_NAMES' in globals() else n

    def color_of(n):
        # Manté la paleta dels altres gràfics
        if n == biological_key:
            return "#000000"
        return get_randomization_colors(n) if 'get_randomization_colors' in globals() else '0.3'

    inches = fig_width_mm / 25.4
    rc = {
        "font.size": font_size,
        "font.family": "Arial",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }

    with mpl.rc_context(rc):
        if ax is None:
            fig, ax = plt.subplots(figsize=(inches, inches*0.6), constrained_layout=True)
        else:
            fig = ax.get_figure()

        # 1) Biològic: àrea farcida en negre
        if biological_key in d_plot:
            ax.hist(d_plot[biological_key],
                    bins=edges, weights=w[biological_key], density=use_density,
                    histtype='stepfilled', color=color_of(biological_key),
                    alpha=0.15, linewidth=0.8, edgecolor='none', zorder=0)

        # 2) Altres ensembles: línia discontínua amb la seva paleta
        for n in names:
            if n == biological_key:
                continue
            ax.hist(d_plot[n],
                    bins=edges, weights=w[n], density=use_density,
                    histtype='step', linewidth=1.0,
                    color=color_of(n), linestyle=(0, (4, 2)), alpha=0.95, zorder=2)

        # Eixos i escala
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density" if use_density else "Count")

        if logy:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            sx = ScalarFormatter(useMathText=True); sx.set_scientific(True); sx.set_powerlimits((0,0)); sx.set_useOffset(False)
            ax.xaxis.set_major_formatter(sx)
        else:
            s = ScalarFormatter(useMathText=True); s.set_scientific(True); s.set_powerlimits((0,0)); s.set_useOffset(False)
            ax.xaxis.set_major_formatter(s)
            sy = ScalarFormatter(useMathText=True); sy.set_scientific(True); sy.set_powerlimits((0,0)); sy.set_useOffset(False)
            ax.yaxis.set_major_formatter(sy)

        if show_mean_lines:
            fig.canvas.draw_idle()
            y0, y1 = ax.get_ylim()
            occupied_spaces = []
            k = 0
            
            def space_is_occupied(space):
                for occupied_space in occupied_spaces:
                    if abs(space[0] - occupied_space[0]) < 2000 and abs(space[1] - occupied_space[1]) < 2000:
                        return True
                return False
                
            for n in names:
                mean_x = (means_um[n] if x_unit == "um" else means_um[n]*1e3)
                ax.vlines(mean_x, y0, y1, colors=color_of(n), linestyles='--',
                          linewidth=0.7, alpha=0.6, zorder=1)

                if mean_label:
                    space = (mean_x + 5500, y1 * 0.98)
                    while space_is_occupied(space):
                        k += 1
                        space = (mean_x + 5500, y1 * (0.98 - k * 0.06))

                    k = 0
                    occupied_spaces.append(space)
                    txt = f"{means_um[n]:.1f} µm" if x_unit == "um" else f"{mean_x:,.0f} nm"
                    ax.text(space[0], space[1], txt,
                            ha='center', va='top', fontsize=font_size-1,
                            color=color_of(n))

        # 4) Llegenda amb totals (km) mantenint el codi de color
        handles, texts = [], []
        for n in names:
            if n == biological_key:
                fc = mpl.colors.to_rgba(color_of(n), 0.25) 
                h = Patch(facecolor=fc, edgecolor='black', linewidth=0.8)
            else:
                h = Line2D([0], [0], color=color_of(n), lw=1.0, linestyle=(0, (4, 2)))
            handles.append(h)
            texts.append(f"{label_of(n)}: {totals_km[n]:.1f} km")
        ax.legend(handles, texts, title="Total synapse lengths",
                  loc="center right", bbox_to_anchor=(1.0, 0.55), 
                  frameon=True, framealpha=0.85,
                  borderpad=0.4, handlelength=1.4, fontsize=font_size,
                  title_fontsize=font_size)

        # Estètica minimalista
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # --- anotació superior: "Means" + dues fletxes a dues mitjanes representatives ---
    if show_mean_lines:
        # (biològica i la més gran)
        mean_x_vals = {n: (means_um[n] if x_unit == "um" else means_um[n]*1e3) for n in names}
        x_left  = mean_x_vals.get(biological_key, min(mean_x_vals.values()))
        x_right = max(mean_x_vals.values())

        x_placement = 0.35
        # text per sobre del gràfic
        ax.text(x_placement, 1.02, "Average synapse lengths", transform=ax.transAxes, ha='center', va='bottom',
                fontsize=font_size, color='0.2', clip_on=False)

        # fletxes cap avall des del text cap a les dues línies de mitjana
        for x_target in (x_left, x_right):
            ax.annotate("", xy=(x_target, y1*0.99), xycoords="data",
                        xytext=(x_placement, 1.02), textcoords=ax.transAxes,
                        arrowprops=dict(linestyle='--', arrowstyle='->', lw=0.6, color='0.2'),
                        annotation_clip=False)


    metrics = {"total_km": totals_km, "mean_um": means_um}
    return ax, metrics
