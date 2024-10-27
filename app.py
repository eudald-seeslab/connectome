# app.py

import io
import time
import logging
import traceback
import streamlit as st
from app_helpers import load_and_display_config
from config_handler import load_config, save_config
from wandb_logger import WandBLogger
from train import main
from multiprocessing import Process, Value

# Initialize a StringIO buffer to capture log output
log_buffer = io.StringIO()


# Custom Streamlit logger handler that writes directly to Streamlit
class StreamlitLoggerHandler(logging.Handler):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self.setLevel(logging.INFO)  # Capture INFO and higher

    def emit(self, record):
        log_entry = self.format(record)
        st.write(log_entry)


# Synchronous model run with log display in Streamlit
def run_model(wandb_logger, stop_flag):
    try:
        wandb_logger.initialize_run(config)
        main(wandb_logger, sweep_config=config, stop_flag=stop_flag)
    except Exception as e:
        error = traceback.format_exc()
        wandb_logger.send_crash(f"Error during training: {error}")
        st.write(f"An error occurred: {error}")
    finally:
        wandb_logger.finish()
        st.write("Model run complete.")


# Load current configuration
config = load_config()

# Set up the stop flag in session state if it doesn't exist
if "stop_flag" not in st.session_state:
    st.session_state["stop_flag"] = Value("i", 0)  # 0 means running, 1 means stopped

# Streamlit UI
st.title("Fruit Fly Connectome Configuration Dashboard")

# Load and display configuration
config = load_and_display_config()

# Stop Model Run Button
if st.button("Stop Model"):
    st.session_state["stop_flag"].value = 1  # Set flag to stop
    st.write("Stop request sent to the model.")

# Start Model Run Button (with auto-save and reload configuration)
if st.button("Run Model"):
    # Auto-save and reload configuration before running
    save_config(config)
    config = load_config()  # Reload the configuration to ensure updated values

    # Display starting message and initialize WandB logger
    st.write("Starting model run ...")
    wandb_logger = WandBLogger(
        config["wandb_project"],
        config.get("wandb_", False),
        config.get("wandb_images_every", 400),
    )

    # Start the model run in a separate process
    process = Process(
        target=run_model, args=(wandb_logger, st.session_state["stop_flag"])
    )
    process.start()

    # Monitor process and log output
    while process.is_alive():
        if st.session_state["stop_flag"].value == 1:
            process.terminate()
            st.write("Model run stopped.")
            break
        time.sleep(5) 

    st.write("Model run complete.")
