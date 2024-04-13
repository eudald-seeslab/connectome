# PROBABLY DEPRECATED
import torch
from torch import nn as nn


def check_for_missing_values(x, step):
    if torch.isnan(x.parameters()).any():
        raise Exception(f"NaN in parameters at step {step}")


def nan_to_unknown(x):
    # Replace the string "nan" with "Unknown"
    return x.replace("nan", "Unknown")


def check_model_parameters(model, epoch):
    # NOT IN USE AT THE MOMENT
    check_for_missing_values(model, epoch)
    # Clip gradients to avoid exploding gradients
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    # Clip parameters to avoid exploding parameters
    for p in model.parameters():
        p.data.clamp_(-1, 1)
