from larvae.utils import log_training_images
import torch
import wandb

import pandas as pd
from models import CombinedModel
from torch.nn.modules.loss import NLLLoss
from torch.optim.adam import Adam


# TODO: all of this needs to be improved


def run_validation_epoch(model, images, labels, total, correct, dev):
    images, labels = images.to(dev), labels.to(dev)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

    # Check and store predictions
    batch_df = check_predictions_and_store_to_df(labels, predicted)
    return correct, total, batch_df


def check_predictions_and_store_to_df(labels, predicted):
    # Check if the prediction is correct
    correct_predictions = (predicted == labels).cpu().numpy().astype(int)
    return pd.DataFrame(
        {
            "Real Label": labels.cpu().numpy(),
            "Predicted Label": predicted.cpu().numpy(),
            "Correct Prediction": correct_predictions,
        }
    )


def run_train_epoch(
    model: CombinedModel,
    criterion: NLLLoss,
    optimizer: Adam,
    images: torch.Tensor,
    labels: torch.Tensor,
    dev: torch.device,
    wb: bool = True,
) -> float:
    # Move images and labels to the device
    images, labels = images.to(dev), labels.to(dev)
    # Forward pass
    outputs = model(images)
    # Compute the loss
    loss = criterion(outputs, labels)
    # Compute the accuracy
    predicted_labels = torch.argmax(outputs, dim=1)
    correct_predictions = (predicted_labels == labels).sum().item()
    accuracy = correct_predictions / len(labels)
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Log images to wandb and tensorboard
    if wb:
        wandb.log({"loss": loss.item(), "accuracy": accuracy})
        log_training_images(images, labels, outputs)

    return loss.item(), accuracy


def calculate_test_accuracy(_test_loader, model, criterion, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in _test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)

    accuracy = correct / total if total > 0 else 0.0
    model.train()  # Set the model back to training mode

    return accuracy, loss.item()
