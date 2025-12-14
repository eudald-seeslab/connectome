import os
import traceback
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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

def plot_ans_accuracies_by_randomization(all_results: pd.DataFrame, out="ans_accuracy", dpi=300):
    apply_plot_style()  # usa el teu estil

    df = all_results.copy()
    df["key"] = df["dataframe"].map(lambda k: NAME_ALIAS.get(k, k))
    df["label"] = df["key"].map(RANDOMIZATION_NAMES)
    df["color"] = df["label"].map(get_randomization_colors)

    # manté només els presents i ordena com a la llegenda
    df = df[df["label"].isin(ORDER)]
    df["label"] = pd.Categorical(df["label"], categories=ORDER, ordered=True)
    df = df.sort_values("label")

    x = np.arange(len(df))
    y = df["w"].to_numpy(float)
    yerr = df["w_se"].to_numpy(float)
    colors = df["color"].to_list()
    labels = df["label"].to_list()

    width_inches = 183 / 25.4
    height_inches = width_inches * 0.75
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=300)

    for i, (yi, sei, ci) in enumerate(zip(y, yerr, colors)):
        ax.errorbar(i, yi, yerr=sei, fmt="o", color=ci, capsize=4, elinewidth=2)

    # eix X compacte, sense tallar punts
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xticks(x, labels, rotation=35, ha="right")

    # eix Y compacte (amb marge per a les barres d’error)
    pad = 5 * (np.nanmax(yerr) if len(yerr) else 0.0)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    ax.set_ylabel("Weber fraction")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.2)

    plots_dir = os.path.join("../..", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"{out}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, f"{out}.pdf"), dpi=300, bbox_inches="tight")
    
    return fig, ax