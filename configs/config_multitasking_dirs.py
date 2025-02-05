import os


data_dirs = ["one_to_five", "blue_yellow_80_110_jitter", "stripes"]


data_dirs_config = [
    {"TRAINING_DATA_DIR": os.path.join("images", data_dir, "train"), 
     "TESTING_DATA_DIR": os.path.join("images", data_dir, "test"),} 
    for data_dir in data_dirs
    ]
