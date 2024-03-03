import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree, Voronoi, voronoi_plot_2d

from flyvision.utils.hex_utils import get_hex_coords, get_hextent

np.random.seed(42)


def get_shift_from_direction(direction):
    shift_u = 0
    shift_v = 0

    match direction:
        case "left":
            shift_u = -1
        case "right":
            shift_u = 1
        case "up":
            shift_v = 1
        case "down":
            shift_v = -1
        case _:
            raise ValueError("direction must be one of 'left', 'right', 'up', 'down'")

    return shift_u, shift_v


def interpolate_hexagon(df_orig, direction="left"):
    shift_u, shift_v = get_shift_from_direction(direction)

    dfp = pd.DataFrame(
        {
            "up": df_orig["u"] + shift_u,
            "vp": df_orig["v"] + shift_v,
            "values": df_orig["values"],
        }
    )

    dft = df_orig.merge(
        dfp, left_on=["u", "v"], right_on=["up", "vp"], suffixes=["", "_shifted"]
    )
    dft["u"] = dft["u"] + shift_u / 2
    dft["v"] = dft["v"] + shift_v / 2
    dft["values"] = (dft["values"] + dft["values_shifted"]) / 2
    dfa = pd.concat([df_orig, dft[["u", "v", "values"]]])

    # Make indices integer again
    if shift_u != 0:
        dfa["u"] = dfa["u"] * 2
    if shift_v != 0:
        dfa["v"] = dfa["v"] * 2
    dfa[["u", "v"]] = dfa[["u", "v"]].astype(int)

    return dfa


def full_hexagon_interpolation(df):
    for direction in ["left", "right", "up", "down"]:
        df = interpolate_hexagon(df, direction)
    return df


def plot_voronoi(df, n):
    rand_points = df[["u", "v"]].sample(n).values

    # Create Voronoi Diagram
    vor = Voronoi(rand_points)

    # Plot Voronoi diagram (optional, for visualization)
    voronoi_plot_2d(vor)
    plt.plot(rand_points[:, 0], rand_points[:, 1], "r.")
    plt.show()


def get_vector_of_voronoi_averages(sample_vector, n_centers):
    """Get the Voronoi averages of a single sample vector.
    Parameters
    ----------
    sample_vector: np.ndarray of shape (n_values,)
    n_centers: int with number of centers to use for Voronoi tesselation

    Returns
    -------
    np.ndarray of shape (n_centers,)
    """

    n_values = sample_vector.shape[0]

    u, v = get_hex_coords(get_hextent(n_values))
    df_ = pd.DataFrame({"u": u, "v": v, "values": sample_vector})
    # If n_centers > n_values, we need to interpolate
    while n_centers > n_values:
        df_ = full_hexagon_interpolation(df_)
        n_values = df_.shape[0]

    rand_points = df_[["u", "v"]].sample(n_centers).values

    # Create a KD-tree for fast lookup of nearest Voronoi cell
    tree = cKDTree(rand_points)

    # Find the nearest Voronoi cell for each point in the lattice
    _, indices = tree.query(df_[["u", "v"]].values)

    # Compute the average value for each Voronoi cell
    return np.array([df_["values"][indices == i].mean() for i in range(n_centers)])


def melt_samples(u, v, sample_tensor):
    mdf = pd.DataFrame(np.swapaxes(sample_tensor, 0, 1)).melt()
    mdf["u"] = np.tile(u, sample_tensor.shape[0])
    mdf["v"] = np.tile(v, sample_tensor.shape[0])
    return mdf


def get_voronoi_averages(array_of_values, n_centers):
    """Get the Voronoi averages of an array of values.
    Parameters
    ----------
    array_of_values: np.ndarray of shape (n_samples, n_values)
    n_centers: int with number of centers to use for Voronoi tesselation

    Returns
    -------
    np.ndarray of shape (n_samples, n_centers)
    """
    n_samples, n_values = array_of_values.shape
    u, v = get_hex_coords(get_hextent(n_values))

    melted_values = melt_samples(u, v, array_of_values)

    return pd.DataFrame(
        melted_values.groupby(["variable"])["value"]
        .apply(get_vector_of_voronoi_averages, n_centers=n_centers)
        .tolist()
    )


def interpolate_hexagon_vectorized(df_orig, direction="left"):
    shift_u, shift_v = get_shift_from_direction(direction)

    dfp = pd.DataFrame(
        {
            "variable": df_orig["variable"],
            "up": df_orig["u"] + shift_u,
            "vp": df_orig["v"] + shift_v,
            "value": df_orig["value"],
        }
    )

    dft = df_orig.merge(
        dfp,
        left_on=["variable", "u", "v"],
        right_on=["variable", "up", "vp"],
        suffixes=["", "_shifted"],
    )
    dft["u"] = dft["u"] + shift_u / 2
    dft["v"] = dft["v"] + shift_v / 2
    dft["values"] = (dft["value"] + dft["value_shifted"]) / 2
    dfa = pd.concat([df_orig, dft[["variable", "u", "v", "value"]]])

    # Make indices integer again
    if shift_u != 0:
        dfa["u"] = dfa["u"] * 2
    if shift_v != 0:
        dfa["v"] = dfa["v"] * 2
    dfa[["u", "v"]] = dfa[["u", "v"]].astype(int)

    return dfa


def batch_full_hexagon_interpolation(df):
    for direction in ["left", "right", "up", "down"]:
        df = interpolate_hexagon_vectorized(df, direction)
    return df


def get_batch_voronoi_averages(activation_tensor, n_centers):
    n_samples, n_values = activation_tensor.shape
    u, v = get_hex_coords(get_hextent(n_values))

    melted_values = melt_samples(u, v, activation_tensor)

    while n_values < n_centers:
        melted_values = batch_full_hexagon_interpolation(melted_values)
        n_values = int(melted_values.shape[0] / activation_tensor.shape[0])

    one_sample_df = melted_values[melted_values["variable"] == 0]
    rand_points = one_sample_df[["u", "v"]].sample(n_centers).values
    tree = cKDTree(rand_points)
    _, indices = tree.query(one_sample_df[["u", "v"]].values)

    voronoi_indices = pd.DataFrame(indices, columns=["voronoi_index"])
    return (
        melted_values.pivot(index=["u", "v"], columns="variable", values="value")
        .reset_index()
        .drop(columns=["u", "v"])
        .merge(voronoi_indices, left_index=True, right_index=True)
        .groupby("voronoi_index")
        .mean()
        .transpose()
    )


def get_activation_tensor(activations_, cell_type, last_frame=8):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    activation_tensor = torch.stack(
        [torch.swapaxes(a[cell_type], 1, 2).squeeze(0) for a in activations_]
    ).to(device)
    # Get the activation vector for the last frame with real data, and for a given sample
    return activation_tensor[:, :, -last_frame].cpu().numpy()


def get_synapse_df():
    classification = pd.read_csv("adult_data/classification.csv")
    connections = pd.read_csv("adult_data/connections.csv")
    return pd.merge(
        connections,
        classification[["root_id", "cell_type"]],
        left_on="pre_root_id",
        right_on="root_id",
    )


def voronoi_averages_to_df(dict_with_voronoi_averages):
    dfs = []
    for key, matrix in dict_with_voronoi_averages.items():
        df = pd.DataFrame(matrix.transpose())
        df["index_name"] = key
        dfs.append(df)

    # Concatenate all the DataFrames into one
    return pd.concat(dfs, axis=0, ignore_index=True)


def create_root_id_mapping(classification):
    # Create a dictionary to hold shuffled root_ids for each cell type
    root_id_mapping = {}

    # Populate the dictionary with shuffled root_ids for each cell type
    for cell_type, group in classification.groupby("cell_type"):
        # Shuffle the root_ids within each group
        shuffled_root_ids = group["root_id"].sample(frac=1).values
        root_id_mapping[cell_type] = shuffled_root_ids
    return root_id_mapping
