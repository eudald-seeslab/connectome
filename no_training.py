import argparse
import os
from matplotlib import pyplot as plt
import torch
import pandas as pd
import numpy as np
import wandb
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count

from model_inspection_funcs import (
    neuron_data_from_image,
    propagate_neuron_data,
    sample_images,
)
import seaborn as sns

device = torch.device("cpu")
dtype = torch.float32

num_passes = 4
pairs_num = 1000


def process_image(args):
    # Unpack all arguments
    img, neuron_data, connections, all_coords, all_neurons, num_passes = args
    activated_data = neuron_data_from_image(img, neuron_data)
    propagation = propagate_neuron_data(
        activated_data, connections, all_coords, all_neurons, num_passes
    )
    return (
        os.path.basename(img),
        propagation["decision_making"][all_neurons["decision_making"] == 1],
    )


def predict_images(
    sampled_images, neuron_data, connections, all_coords, all_neurons, num_passes
):
    # Prepare the list of arguments for each image processing task
    tasks = [
        (img, neuron_data, connections, all_coords, all_neurons, num_passes)
        for img in sampled_images
    ]

    # Use process_map with the adjusted process_image function
    results = process_map(process_image, tasks, max_workers=cpu_count() - 2, chunksize=1)

    # Convert list of tuples into a DataFrame
    dms = {name: dm for name, dm in results}
    return pd.DataFrame(dms)


def process_points_results(df):
    means = pd.DataFrame(df.mean(axis=0))
    means = means.rename(columns={0: "mean"})
    means["yellow"] = [int(a.split("_")[1]) for a in means.index]
    means["blue"] = [int(a.split("_")[2]) for a in means.index]
    means["color"] = means[["yellow", "blue"]].idxmax(axis=1)
    means["pred"] = np.where(means["mean"] > means["mean"].median(), "yellow", "blue")
    means["correct"] = means["color"] == means["pred"]

    return means


def process_shapes_results(predictions, sampled_images):
    means = pd.DataFrame(predictions.mean(axis=0))
    means = means.rename(columns={0: "mean"})
    names = pd.DataFrame(
        {
            "name": [os.path.basename(a) for a in sampled_images],
            "real": [os.path.basename(os.path.dirname(a)) for a in sampled_images],
        }
    )
    tt = means.merge(names, left_index=True, right_on="name", how="left")
    tt["pred"] = np.where(tt["mean"] > tt["mean"].median(), "triangle", "circle")
    tt["correct"] = tt["real"] == tt["pred"]
    return tt

def log_results(results, type, shuffled=False):
    s_char = "_shuffled" if shuffled else ""

    wandb.log({f"{type}{s_char}_table": wandb.Table(dataframe=results)})

    x_axis = "color" if type == "points" else "real"
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    sns.histplot(data=results, x="mean", hue=x_axis, bins=50, kde=True, alpha=0.5, ax=axes[0])
    axes[0].set_title(f"{type.capitalize()} {s_char.replace('_', '').capitalize()} histogram")
    sns.boxplot(data=results, x=x_axis, y="mean", ax=axes[1])
    axes[1].set_title(f"{type.capitalize()} {s_char.replace('_', '').capitalize()} boxplot")
    plt.tight_layout()
    wandb.log({f"{type}{s_char}_img": wandb.Image(fig)})
    plt.close("all")

    acc = np.mean(results["correct"])
    acc = 1 - acc if acc < 0.5 else acc
    wandb.log({f"{type}{s_char}_acc": acc})

def get_data():
    # horrible data stuff
    connections = (
        pd.read_csv(
            "adult_data/connections.csv",
            dtype={
                "pre_root_id": "string",
                "post_root_id": "string",
                "syn_count": np.int32,
            },
        )
        .groupby(["pre_root_id", "post_root_id"])
        .sum("syn_count")
        .reset_index()
    )
    # set weights to 1 because we are not training
    connections["weight"] = 1
    # reshuffle column post_rood_id of the dataframe connections
    shuffled_connections = connections.copy()
    shuffled_connections["post_root_id"] = np.random.permutation(
        connections["post_root_id"]
    )
    right_root_ids = pd.read_csv("adult_data/root_id_to_index.csv")
    all_neurons = (
        pd.read_csv("adult_data/classification_clean.csv")
        .merge(right_root_ids, on="root_id")
        .fillna("Unknown")
    )
    neuron_data = pd.read_csv(
        "adult_data/right_visual_positions_selected_neurons.csv",
        dtype={"root_id": "string"},
    ).drop(columns=["x", "y", "z", "PC1", "PC2"])
    all_coords = pd.read_csv(
        "adult_data/all_coords_clean.csv", dtype={"root_id": "string"}
    )
    rational_cell_types = pd.read_csv("adult_data/rational_cell_types.csv")
    all_neurons["decision_making"] = np.where(
        all_neurons["cell_type"].isin(
            rational_cell_types["cell_type"].values.tolist()
        ),
        1,
        0,
    )
    all_neurons["root_id"] = all_neurons["root_id"].astype("string")

    return connections, shuffled_connections, all_neurons, neuron_data, all_coords


def main(points, shapes):

    if not points and not shapes:
        print("Please select at least one of the two options.")
        return
    
    connections, shuffled_connections, all_neurons, neuron_data, all_coords = get_data()

    blue_yellow = ["#FFD700", "#0000FF"]
    sns.set_palette(blue_yellow)

    # start
    wandb.init(project="no_training", config={"num_pairs": pairs_num})

    # Points
    if points:
        base_dir = "images/five_to_fifteen/train"
        sub_dirs = ["yellow", "blue"]

        sampled_images = sample_images(base_dir, sub_dirs, pairs_num)

        # Normal
        predictions = predict_images(
            sampled_images, neuron_data, connections, all_coords, all_neurons, num_passes
        )
        results = process_points_results(predictions)
        log_results(results, "points")

        # Reshuffled
        predictions = predict_images(
            sampled_images,
            neuron_data,
            shuffled_connections,
            all_coords,
            all_neurons,
            num_passes,
        )

        results = process_points_results(predictions)
        log_results(results, "points", shuffled=True)

    # Shapes
    if shapes:
        base_dir = "images/blue_80_110_jitter/train"
        sub_dirs = ["circle", "triangle"]

        # Normal
        sampled_images = sample_images(base_dir, sub_dirs, pairs_num)
        predictions = predict_images(
            sampled_images, neuron_data, connections, all_coords, all_neurons, num_passes
        )

        results = process_shapes_results(predictions, sampled_images)
        log_results(results, "shapes")

        predictions = predict_images(
            sampled_images,
            neuron_data,
            shuffled_connections,
            all_coords,
            all_neurons,
            num_passes,
        )

        results = process_shapes_results(predictions, sampled_images)
        log_results(results, "shapes", shuffled=True)

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="No training analysis.")
    parser.add_argument(
        "--points",
        action="store_true",
        help="Are we checking the ANS?",
    )
    parser.add_argument(
    "--shapes",
    action="store_true",
    help="Are we trying to differentiate among two shapes?",
)
    args = parser.parse_args()

    main(args.points, args.shapes)
