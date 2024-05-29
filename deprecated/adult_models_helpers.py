import numpy as np
import pandas as pd
import torch


def get_synapse_df():
    classification = pd.read_csv("adult_data/classification.csv")
    connections = pd.read_csv("adult_data/connections.csv")
    return pd.merge(
        connections,
        classification[["root_id", "cell_type"]],
        left_on="pre_root_id",
        right_on="root_id",
    )


def scipy_coo_to_torch_compressed_sparse(
    coo_matrix, shared_weights, sparse_layout, dtype, device
):
    """
    Converts a COO formatted matrix from SciPy to a compressed sparse tensor in PyTorch.

    Parameters:
    coo_matrix (scipy.sparse.coo_matrix): The matrix in COO format.
    sparse_layout (torch.sparse): The sparse layout to use.
    dtype (torch.dtype): The data type of the tensor.
    device (torch.device): The device to store the tensor on.

    Returns:
    torch.Tensor: A sparse tensor in CSR format.
    """
    # Extract COO components
    rows = coo_matrix.row
    cols = coo_matrix.col

    # Number of rows in the matrix
    num_rows = coo_matrix.shape[0]

    # Convert to PyTorch tensors
    col_indices = torch.tensor(cols, dtype=torch.int64)

    # Compute crow_indices
    # np.bincount counts the number of occurrences of each value in an array of non-negative ints.
    row_counts = np.bincount(rows, minlength=num_rows)
    crow_indices = np.cumsum(row_counts, dtype=np.int64)
    # Insert 0 at the start for correct CSR format
    crow_indices = np.insert(crow_indices, 0, 0)
    crow_indices = torch.tensor(crow_indices, dtype=torch.int64)

    # Create the sparse tensor in CSR format
    return torch.sparse_compressed_tensor(
        crow_indices,
        col_indices,
        shared_weights,
        layout=sparse_layout,
        dtype=dtype,
        device=device,
    )
