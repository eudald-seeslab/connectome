# check missing values
import torch
import wandb
import torch.nn as nn
import logging


logger = logging.getLogger("training_log")


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


def log_training_images(images, labels, outputs):
    num_images_to_log = 5
    for i in range(num_images_to_log):
        # Convert image and label to numpy for visualization
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        label = labels[i].cpu().numpy()
        predicted_label = torch.argmax(outputs[i]).cpu().numpy()

        # There is a bug in wandb that prevents logging images
        try:
            wandb.log(
                {
                    "Training Image": wandb.Image(
                        image,
                        caption=f"True Label: {label}, Predicted Label: {predicted_label}",
                    )
                }
            )
        except FileNotFoundError:
            logger.warning(f"Could not log training image")
