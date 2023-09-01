# model_config_manager.py
import logging
import os

from torchvision import models

from model_config import ModelConfig


PRETRAINED_MODELS = models.list_models()
CONFIG_DIRECTORY = "custom_models_config"


logger = logging.getLogger("training_log")


class ModelConfigManager:
    model_type = None
    model_config = None

    def __init__(self):
        self.model_configs = []
        self.load_configs_from_yaml(CONFIG_DIRECTORY)

    def add_config(self, model_config):
        self.model_configs.append(model_config)

    def load_configs_from_yaml(self, config_directory):
        for file_name in os.listdir(config_directory):
            if file_name.endswith(".yml"):
                file_path = os.path.join(config_directory, file_name)
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
        logger.info("Model configurations:")
        logger.info(f"Model name: {self.model_config.model_name}")

        logger.info(f"Number of connectome layers: {self.model_config.num_layers}")

        if self.model_type == "pretrained":
            logger.info("This is a pretrained model")
            return

        logger.info(f"Output channels: {self.model_config.out_channels}")
        logger.info(f"Kernel size: {self.model_config.kernel_size}")
        logger.info(f"Stride: {self.model_config.stride}")
        logger.info(f"Padding: {self.model_config.padding}")
        logger.info("\n")
