# check missing values
import torch


def check_for_missing_values(x, step):
    if torch.isnan(x.parameters()).any():
        raise Exception(f"NaN in parameters at step {step}")


def nan_to_unknown(x):
    # Replace the string "nan" with "Unknown"
    return x.replace("nan", "Unknown")
