import os
import traceback
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.options.mode.chained_assignment = None


def plot_weber_fraction(results_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Create a publication-quality plot showing classification accuracy by Weber ratio.

    Args:
        results_df (pd.DataFrame): DataFrame containing the experimental results
        save_path (str, optional): Path to save the figure

    Returns:
        plt.Figure: The generated figure object
    """

    font_size = 12

    # Data preparation
    results_df["yellow"] = results_df["Image"].apply(
        lambda x: os.path.basename(x).split("_")[1]
    )
    results_df["blue"] = results_df["Image"].apply(
        lambda x: os.path.basename(x).split("_")[2]
    )

    # Calculate Weber ratio with error handling
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

    # Calculate mean and standard error
    correct_percentage = (
        results_df.groupby(["weber_ratio", "equalized"])["Is correct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    correct_percentage["mean"] *= 100
    correct_percentage["std"] *= 100
    correct_percentage["se"] = correct_percentage["std"] / np.sqrt(
        correct_percentage["count"]
    )
    correct_percentage["weber_ratio"] = correct_percentage["weber_ratio"].round(3)

    # Set style for publication
    plt.style.use("seaborn-v0_8-white")

    # Create figure with Nature-compatible dimensions
    # Nature requires figures to be 89 mm or 183 mm wide
    width_mm = 183
    width_inches = width_mm / 25.4
    height_inches = width_inches * 0.75  # Using golden ratio
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=300)

    # Plot data points and error bars
    conditions = [False, True]
    labels = ["Non-equalized", "Surface-equalized"]
    colors = ["#2166AC", "#B2182B"]  # Colorblind-friendly palette

    for condition, label, color in zip(conditions, labels, colors):
        data = correct_percentage[correct_percentage["equalized"] == condition]
        ax.errorbar(
            data["weber_ratio"],
            data["mean"],
            yerr=data["se"],
            label=label,
            color=color,
            marker="o",
            markersize=5,
            capsize=3,
            capthick=1,
            linewidth=1.5,
            linestyle="-",
        )

    # Customize appearance
    ax.set_xlabel("Weber Ratio", fontsize=font_size)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=font_size)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.spines[["right", "top"]].set_visible(False)

    # Add legend
    ax.legend(fontsize=font_size, frameon=False, loc="lower right")

    # Set y-axis limits with some padding
    ax.set_ylim(40, 105)

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

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
