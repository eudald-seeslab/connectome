import os
import glob
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from connectome.visualization.plot_config import (
    RANDOMIZATIONS,
    apply_plot_style,
)

pd.options.mode.chained_assignment = None


def plot_weber_fraction(results_df: pd.DataFrame) -> plt.Figure:
    # Calculate the percentage of correct answers for each Weber ratio
    results_df["yellow"] = results_df["Image"].apply(
        lambda x: os.path.basename(x).split("_")[1]
    )
    results_df["blue"] = results_df["Image"].apply(
        lambda x: os.path.basename(x).split("_")[2]
    )
    try:
        results_df["weber_ratio"] = results_df.apply(
            lambda row: max(int(row["yellow"]), int(row["blue"]))
            / min(int(row["yellow"]), int(row["blue"])),
            axis=1,
        )
    except ZeroDivisionError:
        results_df["weber_ratio"] = 0
    results_df["equalized"] = results_df["Image"].apply(
        lambda x: "equalized" in os.path.basename(x).lower()
    )

    correct_percentage = (
        results_df.groupby(["weber_ratio", "equalized"])["Is correct"].mean() * 100
    )
    correct_percentage = correct_percentage.reset_index()
    # because matplotlib is very stupid:
    correct_percentage["weber_ratio"] = correct_percentage["weber_ratio"].round(3)

    # Plot
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(
        x="weber_ratio", y="Is correct", hue="equalized", data=correct_percentage
    )
    plt.xlabel("Weber Ratio")
    plt.ylabel("Percentage of Correct Answers")
    plt.title("Correct Classification by Weber Ratio and Image Equalization")
    plt.tight_layout()

    return fig


def plot_accuracy_per_value(df, value):
    if value in ["radius", "point_num", "stripes"]:
        split = 1
    elif value == "distance":
        split = 2
    else:
        raise ValueError(
            "Value must be 'radius', 'distance', 'point_num', or 'stripes'"
        )

    df[value] = df["Image"].apply(lambda x: os.path.basename(x).split("_")[split])
    df[value] = df[value].astype(int)
    df["per_correct"] = df.groupby(value)["Is correct"].transform("mean")
    plt.figure()
    ax = sns.scatterplot(data=df, x=value, y="per_correct")
    if value in ["radius", "distance"]:
        xticks = ax.xaxis.get_major_ticks()
        for i in range(len(xticks)):
            if i % 4 != 0:
                xticks[i].set_visible(False)

    return ax


def plot_accuracy_per_colour(df):
    df["num_points"] = df["Image"].apply(
        lambda x: int(os.path.basename(x).split("_")[1])
        + int(os.path.basename(x).split("_")[2])
    )
    df["colour"] = df["Image"].apply(lambda x: os.path.basename(os.path.dirname(x)))
    df["per_correct"] = df.groupby(["colour", "num_points"])["Is correct"].transform(
        "mean"
    )
    plt.figure()
    ax = sns.barplot(data=df, x="num_points", y="per_correct", hue="colour")

    return ax


def plot_contingency_table(df, classes):
    label_map = dict(enumerate(classes))
    df["Prediction"] = df["Prediction"].map(label_map)
    df["True label"] = df["True label"].map(label_map)

    return (
        df.value_counts(["Prediction", "True label"])
        .unstack()
        .plot(kind="bar", stacked=True)
    )


def plot_results(results_, plot_types, classes=None):
    plots = []
    try:
        for plot_type in plot_types:
            if plot_type == "weber":
                plots.append(plot_weber_fraction(results_.copy()))
            elif plot_type in ["radius", "distance", "point_num", "stripes"]:
                plots.append(plot_accuracy_per_value(results_.copy(), plot_type))
            elif plot_type == "colour":
                plots.append(plot_accuracy_per_colour(results_.copy()))
            elif plot_type == "contingency":
                plots.append(plot_contingency_table(results_.copy(), classes))
    except Exception:
        error = traceback.format_exc()
        print(f"Error plotting results: {error}")

    return plots


def guess_your_plots(config_):
    if config_.plot_types is None:
        # If the user has specified None, don't plot anything
        return []
    if len(config_.plot_types) > 0:
        # If the user has specified the plot types, use them
        return config_.plot_types

    # If the user has left an empty list, it's guessing time
    classes = config_.CLASSES
    # if there is a colour class, it's either weber or colour. One of the plots will be
    #  useless, but it won't crash, just don't look at it
    if any([x in classes for x in ["blue", "yellow", "green", "red"]]):
        return ["weber", "colour"]
    # if there are geometry classes, it's radius, distance and contingency
    if any([x in classes for x in ["circle", "square", "triangle", "star"]]):
        return ["radius", "distance", "contingency"]
    # if there are numbers bigger than 10 in the classes, they will be angles, so it's stripes
    if any([int(x) > 10 for x in classes]):
        return ["stripes"]
    # if there are numbers smaller than 10, it's guess the numbers
    if all([int(x) < 10 for x in classes]):
        # Except for mnist
        if not config_.data_type == "mnist":
            return ["point_num"]
    return []


########################################################
# All of this needs to be moved
########################################################

ORDER = list(RANDOMIZATIONS.labels_in_order)

def _prepare_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse filename, compute Weber ratio, equalized flag, and return tidy df."""
    df = df.copy()

    # Parse counts from "Image" name "..._<yellow>_<blue>_...":
    df["yellow"] = df["Image"].apply(lambda x: os.path.basename(x).split("_")[1])
    df["blue"]   = df["Image"].apply(lambda x: os.path.basename(x).split("_")[2])

    # Compute Weber ratio robustly
    def _weber(row):
        try:
            y = int(row["yellow"])
            b = int(row["blue"])
            lo, hi = sorted((y, b))
            return np.inf if lo == 0 else hi / lo
        except Exception:
            return np.nan

    df["weber_ratio"] = df.apply(_weber, axis=1)

    # remove weber ratios smaller that 1.33
    df = df[df["weber_ratio"] >= 1.33]
    
    df["equalized"] = df["Image"].str.lower().str.contains("equalized")
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy and standard error by Weber ratio (only equalized)."""
    tidy = df[df["equalized"] == True].copy()

    agg = (
        tidy.groupby("weber_ratio")["Is correct"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .dropna(subset=["weber_ratio"])
    )
    agg["mean"] *= 100
    agg["std"]  *= 100
    agg["se"]   = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["weber_ratio"] = agg["weber_ratio"].round(3)

    return agg.sort_values("weber_ratio")


def plot_weber_by_randomization(
    folder: str,
    files_to_keys: dict[str, str] | None = None,
    *,
    min_weber: float | None = None,
    save_path: str | None = None,
    ax=None,
    show_legend=True,
):
    """
    Compare classification accuracy vs. Weber ratio for biological connectome
    and multiple randomizations (surface-equalized only).
    """
    if files_to_keys is None:
        files_to_keys = {
            "biological_results.csv": "biological",
            "unconstrained_results.csv": "unconstrained",
            "connection_pruned_results.csv": "connection_pruned",
            "binned_results.csv": "random_binned",
            "neuron_binned_results.csv": "neuron_binned",
            "random_pruned_results.csv": "random_pruned",
        }

    curves = []
    for filename, key in files_to_keys.items():
        candidates = [
            os.path.join(folder, filename),
            *glob.glob(os.path.join(folder, filename))
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), None)
        if csv_path is None:
            continue

        df = pd.read_csv(csv_path)
        df = _prepare_results_df(df)
        agg = _aggregate(df)
        if min_weber is not None:
            agg = agg[agg["weber_ratio"] >= float(min_weber)]
        if agg.empty:
            continue

        display = RANDOMIZATIONS.label_for(key)
        color = RANDOMIZATIONS.color_for(key)
        curves.append((display, color, agg))

    apply_plot_style()

    if ax is None:
        width_inches = 183 / 25.4
        height_inches = width_inches * 0.75
        fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=300)
    else:
        fig = ax.get_figure()

    for label, color, data in curves:
        ax.errorbar(
            data["weber_ratio"],
            data["mean"],
            yerr=data["se"],
            label=label,
            color=color,
            marker="o",
            linewidth=2,
            capsize=3,
            capthick=1,
        )

    if curves:
        x_values = sorted(
            {
                float(value)
                for _, _, data in curves
                for value in data["weber_ratio"].to_numpy()
            }
        )

        def _format_weber_tick(value: float) -> str:
            fraction_labels = {
                4 / 3: "4/3",
                3 / 2: "3/2",
                5 / 3: "5/3",
                5 / 2: "5/2",
            }
            for fraction_value, label in fraction_labels.items():
                if np.isclose(value, fraction_value, atol=1e-3):
                    return label
            return f"{value:.2f}".rstrip("0").rstrip(".")

        staggered_labels = [
            f"{_format_weber_tick(value)}\n" if idx % 2 == 0 else f"\n{_format_weber_tick(value)}"
            for idx, value in enumerate(x_values)
        ]
        ax.set_xticks(x_values)
        ax.set_xticklabels(staggered_labels)
        if len(x_values) > 1:
            step = min(np.diff(x_values))
            ax.set_xlim(x_values[0] - step * 0.4, x_values[-1] + step * 0.4)

    ax.set_xlabel("Weber ratio", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.set_ylim(40, 105)
    if show_legend:
        ax.legend(frameon=False, loc="lower right", fontsize=14)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def weber_accuracy_table(
    folder: str,
    files_to_keys: dict[str, str] | None = None,
    *,
    min_weber: float | None = None,
    decimals: int = 2,
) -> pd.DataFrame:
    """Return the per-Weber-ratio accuracies shown in the Weber plot."""
    if files_to_keys is None:
        files_to_keys = {
            "biological_results.csv": "biological",
            "unconstrained_results.csv": "unconstrained",
            "connection_pruned_results.csv": "connection_pruned",
            "binned_results.csv": "random_binned",
            "neuron_binned_results.csv": "neuron_binned",
        }

    per_model = {}
    for filename, key in files_to_keys.items():
        candidates = [
            os.path.join(folder, filename),
            *glob.glob(os.path.join(folder, filename))
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), None)
        if csv_path is None:
            continue

        df = pd.read_csv(csv_path)
        df = _prepare_results_df(df)
        agg = _aggregate(df)
        if min_weber is not None:
            agg = agg[agg["weber_ratio"] >= float(min_weber)]
        if agg.empty:
            continue

        per_model[RANDOMIZATIONS.label_for(key)] = agg.set_index("weber_ratio")["mean"]

    table = pd.DataFrame(per_model)
    table.index.name = "Weber Ratio"
    return table.round(decimals)

def plot_ans_accuracies_by_randomization(all_results: pd.DataFrame, ax=None, save=False):
    """Plot Weber fraction by randomization type."""
    apply_plot_style()

    df = all_results.copy()
    df["key"] = df["dataframe"].map(RANDOMIZATIONS.resolve_key)
    df["label"] = df["key"].map(RANDOMIZATIONS.label_for)
    df["color"] = df["key"].map(RANDOMIZATIONS.color_for)

    df = df[df["label"].isin(ORDER)]
    df["label"] = pd.Categorical(df["label"], categories=ORDER, ordered=True)
    df = df.sort_values("label")

    x = np.arange(len(df))
    y = df["w"].to_numpy(float)
    yerr = df["w_se"].to_numpy(float)
    colors = df["color"].to_list()
    labels = df["label"].to_list()

    if ax is None:
        width_inches = 183 / 25.4
        height_inches = width_inches * 0.75
        fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=300)
    else:
        fig = ax.get_figure()

    for i, (yi, sei, ci) in enumerate(zip(y, yerr, colors)):
        ax.errorbar(i, yi, yerr=sei, fmt="o", color=ci, capsize=4, elinewidth=2)

    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xticks(x, labels, rotation=35, ha="right")

    pad = 5 * (np.nanmax(yerr) if len(yerr) else 0.0)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    ax.set_ylabel("Weber fraction")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save:
        plots_dir = os.path.join("../..", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(plots_dir, "ans_accuracy.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(plots_dir, "ans_accuracy.pdf"), dpi=300, bbox_inches="tight")
    
    return ax