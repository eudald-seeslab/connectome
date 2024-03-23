import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, load_npz
from torch_geometric.data import Data, Batch

from retina_to_connectome_funcs import (
    get_activation_tensor,
    get_batch_voronoi_averages,
    voronoi_averages_to_df,
    get_synapse_df,
)


def compute_voronoi_averages(
    activations, classification, decoding_cells, last_good_frame=8
):
    """
    Calculate Voronoi averages for each cell type in the classification.
    """
    averages_dict = {}
    for cell_type in decoding_cells:
        number_of_cells = len(classification[classification["cell_type"] == cell_type])
        if number_of_cells > 0:
            activation_tensor = (
                get_activation_tensor(
                    activations, cell_type, last_frame=last_good_frame
                )
                / 255
            )
            averages_dict[cell_type] = get_batch_voronoi_averages(
                activation_tensor, n_centers=number_of_cells
            )
    return voronoi_averages_to_df(averages_dict)


def get_synaptic_matrix(activation_df):

    # not in use unless we have to recreate it, since it's now saved in an external file

    synapse_df = get_synapse_df()

    # Step 1: Identify Common Neurons
    # Unique root_ids in merged_df
    neurons_merged = pd.unique(activation_df["root_id"])

    # Unique root_ids in synapse_df (both pre and post)
    neurons_synapse_pre = pd.unique(synapse_df["pre_root_id"])
    neurons_synapse_post = pd.unique(synapse_df["post_root_id"])
    neurons_synapse = np.unique(
        np.concatenate([neurons_synapse_pre, neurons_synapse_post])
    )

    # Common neurons
    common_neurons = np.intersect1d(neurons_merged, neurons_synapse)

    # Filter synapse_df to include only rows with both pre and post root_ids in common_neurons
    filtered_synapse_df = synapse_df[
        synapse_df["pre_root_id"].isin(common_neurons)
        & synapse_df["post_root_id"].isin(common_neurons)
    ]

    # Map neuron root_ids to matrix indices
    root_id_to_index = {root_id: index for index, root_id in enumerate(common_neurons)}

    # Convert root_ids in filtered_synapse_df to matrix indices
    pre_indices = filtered_synapse_df["pre_root_id"].map(root_id_to_index).values
    post_indices = filtered_synapse_df["post_root_id"].map(root_id_to_index).values

    # Use syn_count as the data for the non-zero elements of the matrix
    data = filtered_synapse_df["syn_count"].values

    # Create a sparse matrix in COO format
    return (
        coo_matrix(
            (data, (pre_indices, post_indices)),
            shape=(len(common_neurons), len(common_neurons)),
            dtype=np.int64,
        ),
        root_id_to_index,
    )


def get_rational_neurons(root_id_to_index, activation_df):

    # not in use at the moment, since it's now saved in an external file

    # get the info from data/cell_type_rational.csv
    cell_type_rational = pd.read_csv("data/cell_type_rational.csv")
    # get the cell types with rational = 1
    rational_cell_types = cell_type_rational[cell_type_rational["rational"] == 1][
        "cell_type"
    ]
    # find, using the root to index dictionary, the indices of the rational cells
    rational_indices = [
        root_id_to_index[root_id]
        for root_id in activation_df[
            activation_df["cell_type"].isin(rational_cell_types)
        ]["root_id"]
    ]
    decision_making_vector = np.zeros(activation_df.shape[0], dtype=int)
    decision_making_vector[rational_indices] = 1

    return decision_making_vector


def from_retina_to_connectome(voronoi_averages_df, classification, root_id_to_index_df):

    # map indices to neuron ids coming from adult_data/root_id_to_index
    mapped_df = voronoi_averages_df.merge(
        root_id_to_index_df, left_index=True, right_on="index_id", how="right"
    ).drop(columns=["index_id", "index_name"])
    # create the full dataframe, with 0 everywhere except the activation value on the specific indices
    return (
        pd.merge(
            classification[["root_id"]],
            mapped_df,
            on="root_id",
            how="right",
        )
        .fillna(0)
        .drop(columns=["root_id"])
    )


def from_connectome_to_model(activation_df_, labels_):
    synaptic_matrix = load_npz("adult_data/synaptic_matrix_sparse.npz")
    edges = torch.tensor(
        np.array([synaptic_matrix.row, synaptic_matrix.col]),
        dtype=torch.long,
    )
    activation_tensor = torch.tensor(activation_df_.values, dtype=torch.float16)
    graph_list_ = []
    for i in range(activation_tensor.shape[1]):
        # Shape [num_nodes, 1], one feature per node
        node_features = activation_tensor[:, i].unsqueeze(1)
        graph = Data(x=node_features, edge_index=edges, y=labels_[i])
        graph_list_.append(graph)

    return graph_list_


def from_retina_to_model(
    activations,
    labels,
    decoding_cells,
    last_good_frame,
    classification,
    root_id_to_index,
):

    voronoi_averages_df = compute_voronoi_averages(
        activations, classification, decoding_cells, last_good_frame=last_good_frame
    )
    activation_df = from_retina_to_connectome(
        voronoi_averages_df, classification, root_id_to_index
    )
    graph_list = from_connectome_to_model(activation_df, labels)
    return (
        Batch.from_data_list(graph_list),
        torch.Tensor(labels),
    )


# todo: move elsewhere
def get_decision_making_neurons():
    # get a dataframe indicating which neurons will be used to classify
    rational_neurons = pd.read_csv("adult_data/rational_neurons.csv", index_col=0)
    return torch.tensor(rational_neurons.values.squeeze(), dtype=torch.float16).detach()


def get_cell_type_indices(classification, root_id_to_index, decoding_cells):
    # Merge classification with root_id_to_index to associate each index_id with a cell_type
    merged_df = root_id_to_index.merge(classification, on="root_id", how="left")

    # Filter only for cells in decoding_cells
    merged_df = merged_df[merged_df["cell_type"].isin(decoding_cells)]

    # Generate a mapping from cell types to unique integer indices
    cell_type_to_index = {
        cell_type: i for i, cell_type in enumerate(merged_df["cell_type"].unique())
    }

    # Apply the mapping to get cell_type indices
    merged_df["cell_type_index"] = merged_df["cell_type"].map(cell_type_to_index)

    # Return a series with one column indicating the cell type index for each node
    return merged_df["cell_type_index"]
