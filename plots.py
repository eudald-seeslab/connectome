import os
import traceback
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import config

pd.options.mode.chained_assignment = None


def plot_weber_fraction(results_df: pd.DataFrame) -> plt.Figure:
    # Calculate the percentage of correct answers for each Weber ratio
    results_df["yellow"] = results_df["Image"].apply(
        lambda x: os.path.basename(x).split("_")[1]
    )
    results_df["blue"] = results_df["Image"].apply(
        lambda x: os.path.basename(x).split("_")[2]
    )
    results_df["weber_ratio"] = results_df.apply(
        lambda row: max(int(row["yellow"]), int(row["blue"]))
        / min(int(row["yellow"]), int(row["blue"])),
        axis=1,
    )
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
        raise ValueError("Value must be 'radius', 'distance', 'point_num', or 'stripes'")

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


def plot_contingency_table(df):
    label_map = dict(enumerate(config.CLASSES))
    df["Prediction"] = df["Prediction"].map(label_map)
    df["True label"] = df["True label"].map(label_map)

    return (
        df.value_counts(["Prediction", "True label"])
        .unstack()
        .plot(kind="bar", stacked=True)
    )


def plot_results(results_, plot_types):
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
                plots.append(plot_contingency_table(results_.copy()))
    except Exception:
        error = traceback.format_exc()
        print(f"Error plotting results: {error}")

    return plots
