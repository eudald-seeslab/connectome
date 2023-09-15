# model_config.py
import yaml
from typing import Dict, Union


class ModelConfig:
    def __init__(self, model_config: Dict[str, Union[int, float, str, bool]]) -> None:
        # General config
        self.connectome_layer_number = model_config.get("CONNECTOME_LAYER_NUMBER")
        self.model_name = model_config.get("RETINA_MODEL")

        # Config only for custom retina models
        self.num_layers = None
        self.out_channels = None
        self.kernel_size = None
        self.kernel_stride = None
        self.kernel_padding = None
        self.pool_kernel_size = None
        self.pool_stride = None

    def get_data_from_yaml(self, file_path: str) -> "ModelConfig":
        with open(file_path) as f:
            config_data = yaml.safe_load(f)
            self.num_layers = config_data.get("num_layers")
            self.out_channels = config_data.get("out_channels")
            self.kernel_size = config_data.get("kernel_size")
            self.kernel_stride = config_data.get("kernel_stride")
            self.kernel_padding = config_data.get("kernel_padding")
            self.pool_kernel_size = config_data.get("pool_kernel_size")
            self.pool_stride = config_data.get("pool_stride")
        return self

    def get_model_config(self) -> "ModelConfig":
        return self
