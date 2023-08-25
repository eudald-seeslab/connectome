# model_config_manager.py
from torchvision import models

from model_config import ModelConfig


PRETRAINED_MODELS = models.list_models()


class ModelConfigManager:
    model_type = None

    def __init__(self):
        self.model_configs = []

    def add_config(self, model_config):
        self.model_configs.append(model_config)

    def load_configs_from_yaml(self, file_paths):
        for file_path in file_paths:
            model_config = ModelConfig.from_yaml(file_path)
            self.add_config(model_config)

    def set_model_config(self, model_name):
        # Pretrained models are not in the config file
        if model_name in PRETRAINED_MODELS:
            self.model_config = ModelConfig.from_dict(dict({"model_name": model_name}))
            self.model_type = "pretrained"
            return

        for config in self.model_configs:
            if config.model_name == model_name:
                self.model_config = config
                self.model_type = "custom"
                return
        raise ValueError(f"Model configuration '{model_name}' not found.")

    def output_model_details(self):
        print("Model configurations:")
        print(f"Model name: {self.model_config.model_name}")

        if self.model_type == "pretrained":
            print("This is a pretrained model")
            return

        print(f"Output channels: {self.model_config.out_channels}")
        print(f"Kernel size: {self.model_config.kernel_size}")
        print(f"Stride: {self.model_config.stride}")
        print(f"Padding: {self.model_config.padding}")
        print("\n")
