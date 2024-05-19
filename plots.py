import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import create_input_images.data_config as data_config

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

    # Plot
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(
        x="weber_ratio", y="Is correct", hue="equalized", data=correct_percentage
    )
    plt.xlabel("Weber Ratio")
    plt.ylabel("Percentage of Correct Answers")
    plt.title("Correct Classification by Weber Ratio and Image Equalization")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_accuracy_per_value(df, value):
    if value == "radius":
        split = 1
    elif value == "distance":
        split = 2
    else:
        raise ValueError("Value must be 'radius' or 'distance'")

    df[value] = df["Image"].apply(lambda x: os.path.basename(x).split("_")[split])
    df[value] = df[value].astype(int)
    df["per_correct"] = df.groupby(value)["Is correct"].transform("mean")
    plt.figure()
    ax = sns.scatterplot(data=df, x=value, y="per_correct")
    xticks = ax.xaxis.get_major_ticks()
    for i in range(len(xticks)):
        if i % 4 != 0:
            xticks[i].set_visible(False)

    return ax

def plot_contingency_table(df):
    label_map = dict(enumerate(data_config.CLASSES))
    df["Prediction"] = df["Prediction"].map(label_map)
    df["True label"] = df["True label"].map(label_map)

    return df.value_counts(["Prediction", "True label"]).unstack().plot(kind="bar", stacked=True)
