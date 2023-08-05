from torch import nn, optim, device, cuda
import torch
import yaml
from tqdm import trange

from data_parser import adj_matrix, nodes
from image_parser import train_loader
from utils import check_for_missing_values

from network_models import CombinedModel

config = yaml.safe_load(open("config.yml"))
epochs = config["EPOCHS"]

problems = False

# Initialize the combined model
combined_model = CombinedModel(adj_matrix, neurons=nodes)

# TODO: install cuda
dev = device("cuda" if cuda.is_available() else "cpu")
combined_model = combined_model.to(dev)

# Specify the loss function and the optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=0.0001)

for epoch in trange(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Move images and labels to the device
        images, labels = images.to(dev), labels.to(dev)

        # Checks
        if torch.isnan(images).any():
            raise Exception("NaN in images")
        if torch.isnan(labels).any():
            raise Exception("NaN in labels")

        # Forward pass
        outputs = combined_model(images)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()

        if problems:
            # check for missing values in model parameters
            check_for_missing_values(combined_model, epoch)
            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(combined_model.parameters(), 1)
            # Clip parameters to avoid exploding parameters
            for p in combined_model.parameters():
                p.data.clamp_(-1, 1)

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
