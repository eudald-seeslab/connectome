import pandas as pd


def get_synapse_df():
    classification = pd.read_csv("adult_data/classification.csv")
    connections = pd.read_csv("adult_data/connections.csv")
    return pd.merge(
        connections,
        classification[["root_id", "cell_type"]],
        left_on="pre_root_id",
        right_on="root_id",
    )
