import os
import logging
import torch
from pathlib import Path


class ModelManager:
    # For now, we only log stuff on the training run
    logger = logging.getLogger("training_log")
    last_model_filename = None

    def __init__(self, config, save_dir, clean_previous=False):
        self.retina_model = config["RETINA_MODEL"]
        self.connectome_layer_number = config["CONNECTOME_LAYER_NUMBER"]
        self.custom_name = config["SAVED_MODEL_NAME"]
        self.model_dir_name = f"{self.custom_name}_{self.retina_model}_{self.connectome_layer_number}_layers"
        self.model_dir = os.path.join(save_dir, self.model_dir_name)
        self.clean_previous = clean_previous

        # Create a directory for the current model if it doesn't exist
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def save_model(self, model, epoch):
        self.last_model_filename = f"epochs_{epoch + 1}.pth"
        model_path = os.path.join(self.model_dir, self.last_model_filename)

        # Save the model state dictionary
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Saved model after {epoch + 1} runs")

    @staticmethod
    def load_model(model, model_path):
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path), strict=False)
            return model

        raise FileNotFoundError(f"Model '{model_path}' not found.")

    def clean_previous_runs(self):
        if self.clean_previous:
            # Delete all files in the model directory except for the current run
            # Get only files ending with pth
            current_run_files = [
                filename for filename in os.listdir(self.model_dir) if filename.endswith(".pth")
            ]
            for filename in current_run_files:
                file_path = os.path.join(self.model_dir, filename)
                # This fails in some edge cases
                try:
                    if os.path.isfile(file_path) and not filename.startswith(
                        self.last_model_filename
                    ):
                        os.remove(file_path)
                except TypeError:
                    self.logger.warning(f"Failed to delete {file_path}")

    def save_model_plot(self, plot):
        os.makedirs("sweep_plots", exist_ok=True)
        plot_path = os.path.join("sweep_plots", f"weber_fraction_plot_{self.connectome_layer_number}.png")
        plot.savefig(plot_path)
