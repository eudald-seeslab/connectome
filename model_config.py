# model_config.py
import yaml


class ModelConfig:
    def __init__(self, model_config):
        # General config
        self.connectome_layer_number = model_config["CONNECTOME_LAYER_NUMBER"]

        # Config only for custom retina models
        self.model_name = None
        self.num_layers = None
        self.out_channels = None
        self.kernel_size = None
        self.stride = None
        self.padding = None

    def get_data_from_yaml(self, file_path):
        with open(file_path) as f:
            config_data = yaml.safe_load(f)
            self.model_name = config_data["model_name"]
            self.num_layers = config_data["num_layers"]
            self.out_channels = config_data["out_channels"]
            self.kernel_size = config_data["kernel_size"]
            self.stride = config_data["stride"]
            self.padding = config_data["padding"]
        return self
