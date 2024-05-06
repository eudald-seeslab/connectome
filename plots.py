import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


pd.options.mode.chained_assignment = None


def plot_weber_fraction(results_df: pd.DataFrame) -> plt.Figure:
    # Calculate the percentage of correct answers for each Weber ratio
    results_df["yellow"] = results_df["Image"].apply(lambda x: x.split("_")[2])
    results_df["blue"] = results_df["Image"].apply(lambda x: x.split("_")[3])
    results_df["weber_ratio"] = results_df.apply(
        lambda row: max(int(row["yellow"]), int(row["blue"]))
        / min(int(row["yellow"]), int(row["blue"])),
        axis=1,
    )
    correct_percentage = results_df.groupby("weber_ratio")["Is correct"].mean() * 100
    # Plot
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x=correct_percentage.index, y=correct_percentage.values)
    plt.xlabel("Weber Ratio")
    plt.ylabel("Percentage of Correct Answers")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_accuracy_per_value(df, value):
    if value == "radius":
        split = 1
    elif value == "dist":
        split = 2
    else:
        raise ValueError("Value must be 'radius' or 'distance'")

    df[value] = df["Image"].apply(lambda x: os.path.basename(x).split("_")[split])
    df["per_correct"] = df.groupby(value)["Is correct"].transform("mean")
    plt.figure()
    ax = sns.scatterplot(data=df, x="value", y="per_correct")
    xticks = ax.axis.get_major_ticks()
    for i in range(xticks):
        if i % 4 != 0:
            xticks[i].set_visible(False)

    return ax
