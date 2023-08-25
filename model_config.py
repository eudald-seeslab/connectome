# model_config.py
import yaml


class ModelConfig:
    def __init__(self, model_name, out_channels, kernel_size, stride, padding):
        self.model_name = model_name
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(
            model_name=config_data["model_name"],
            out_channels=config_data["out_channels"],
            kernel_size=config_data["kernel_size"],
            stride=config_data["stride"],
            padding=config_data["padding"],
        )

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            model_name=config_dict["model_name"],
        )
