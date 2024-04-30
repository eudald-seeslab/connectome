from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import sys


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


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
