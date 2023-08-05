# model_config_manager.py
from model_config import ModelConfig


class ModelConfigManager:
    def __init__(self):
        self.model_configs = []

    def add_config(self, model_config):
        self.model_configs.append(model_config)

    def load_configs_from_yaml(self, file_paths):
        for file_path in file_paths:
            model_config = ModelConfig.from_yaml(file_path)
            self.add_config(model_config)

    def get_config(self, model_name):
        for config in self.model_configs:
            if config.model_name == model_name:
                return config
        raise ValueError(f"Model configuration '{model_name}' not found.")
