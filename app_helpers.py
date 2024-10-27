import streamlit as st
from config_handler import load_config


# Load configuration and initialize UI
def load_and_display_config():
    config = load_config()

    # Main Settings
    st.header("Main Settings")
    config["data_dir"] = st.text_input(
        "Data Directory", config.get("data_dir", "one_to_ten")
    )
    config["train_edges"] = st.checkbox(
        "Train Edges", value=config.get("train_edges", False)
    )
    config["train_neurons"] = st.checkbox(
        "Train Neurons", value=config.get("train_neurons", True)
    )
    config["refined_synaptic_data"] = st.checkbox(
        "Refined Synaptic Data", value=config.get("refined_synaptic_data", True)
    )
    config["synaptic_limit"] = st.checkbox(
        "Synaptic Limit", value=config.get("synaptic_limit", True)
    )

    # Training Settings
    st.header("Training Settings")
    config["debugging"] = st.checkbox("Debugging", value=config.get("debugging", False))
    config["device_type"] = st.selectbox(
        "Device Type",
        ["cpu", "cuda"],
        index=0 if config.get("device_type") == "cpu" else 1,
    )

    # WandB Project Name
    st.header("WandB Project")
    config["wandb_project"] = st.text_input(
        "WandB Project Name", config.get("wandb_project", "no_synaptic_limit")
    )

    return config
