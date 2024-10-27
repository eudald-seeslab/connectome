# app/config_handler.py

import json
import os

CONFIG_FILE_PATH = "config.json"


def load_json_keys():
    # Read the existing keys in config.json
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r") as json_file:
            json_data = json.load(json_file)
            return list(json_data.keys())  # Get keys as list
    return []


def load_config():
    # Start with default settings from config.py
    import config as default_config

    config = {
        key: getattr(default_config, key)
        for key in dir(default_config)
        if not key.startswith("__")
    }

    # Override with active settings from config.json if available
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r") as json_file:
            # Only updates keys present in config.json
            config.update(json.load(json_file))

    return config


def save_config(config):
    json_config = {key: config[key] for key in load_json_keys()}

    with open(CONFIG_FILE_PATH, "w") as json_file:
        json.dump(json_config, json_file, indent=4)


def udpate_config(config):
    save_config(config)
    return load_config()
