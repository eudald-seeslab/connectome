# model_config_manager.py
import logging
import os

from torchvision import models

from model_config import ModelConfig
from typing import Dict, Union


PRETRAINED_MODELS = models.list_models()
CONFIG_DIRECTORY = "custom_models_config"


logger = logging.getLogger("training_log")


class ModelConfigManager:
    model_type = None
    current_model_config = None

    def __init__(
        self, connectome_config: Dict[str, Union[int, float, str, bool]]
    ) -> None:
        self.connectome_config = connectome_config
        self.model_name = connectome_config["RETINA_MODEL"]
        self.connectome_layer_number = connectome_config["CONNECTOME_LAYER_NUMBER"]
        self.retina_model_configs = []
        self.load_configs_from_yaml(CONFIG_DIRECTORY)

    def load_configs_from_yaml(self, config_directory: str) -> None:
        for file_name in os.listdir(config_directory):
            if file_name.endswith(".yml"):
                file_path = os.path.join(config_directory, file_name)
                model_config = ModelConfig(self.connectome_config).get_data_from_yaml(
                    file_path
                )
                self.retina_model_configs.append(model_config)

    def set_model_config(self, model_name: str) -> None:
        # Pretrained models are not in the custom retina models config files
        if model_name in PRETRAINED_MODELS:
            self.current_model_config = ModelConfig(
                self.connectome_config
            ).get_model_config()
            self.model_type = "pretrained"
            return

        for config in self.retina_model_configs:
            if config.model_name == model_name:
                self.current_model_config = config
                self.model_type = "custom"
                return
        raise ValueError(f"Model configuration '{model_name}' not found.")

    def output_model_details(self) -> None:
        logger.info("Model configurations:")
        logger.info(f"Model name: {self.current_model_config.model_name}")

        logger.info(f"Number of connectome layers: {self.connectome_layer_number}")

        if self.model_type == "pretrained":
            logger.info("This is a pretrained model")
            if self.current_model_config.only_first_layer:
                logger.warning("Only the first layer of the pretrained model will be used")
            return

        logger.info(f"Number of retina layers: {self.current_model_config.num_layers}")
        logger.info(f"Output channels: {self.current_model_config.out_channels}")
        logger.info(f"Kernel size: {self.current_model_config.kernel_size}")
        logger.info(f"Kernel stride: {self.current_model_config.kernel_stride}")
        logger.info(f"Kernel padding: {self.current_model_config.kernel_padding}")
        logger.info(f"Pool kernel size: {self.current_model_config.pool_kernel_size}")
        logger.info(f"Pool stride: {self.current_model_config.pool_stride}")
        logger.info(f"Dropout: {self.current_model_config.dropout}")
        logger.info("\n")
