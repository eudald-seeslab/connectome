import pandas as pd

# Upload the adjacency matrix
adj_matrix = pd.read_csv(
    "data/science.add9330_data_s1/Supplementary-Data-S1/all-all_connectivity_matrix.csv",
    index_col=0,
)
nodes = pd.read_csv("node_properties_clean.csv")

# Create a column with info on whether the neuron is visual or not
nodes["visual"] = nodes["additional_annotations"].apply(
    lambda x: True if x == "visual" else False
)
nodes["rational"] = nodes["celltype"].apply(lambda x: True if x == "MBON" else False)

# Merge the adjacency matrix with the nodes dataframe so that the nodes dataframe
#  contains skid's corresponding to the row indices of the adjacency matrix
nodes = nodes.merge(
    adj_matrix.index.to_frame(), left_on="skid", right_index=True, how="right"
)
nodes = nodes.reset_index(drop=True)
nodes = nodes.drop(
    columns=[0, "additional_annotations", "level_7_cluster", "hemisphere"]
)

# Clean nodes
nodes["visual"] = nodes["visual"].fillna(False)
nodes["rational"] = nodes["rational"].fillna(False)
nodes["celltype"] = nodes["celltype"].fillna("Unknown")

# Reset adjacency matrix index and column names to integer
adj_matrix = adj_matrix.reset_index(drop=True)
adj_matrix.columns = range(adj_matrix.shape[1])

# Put adj_matrix to a new dataframe in longitudinal form and set 1 if there is a
#  connection between two neurons and 0 otherwise
adj_matrix_long = pd.melt(adj_matrix.reset_index(), id_vars="index")
