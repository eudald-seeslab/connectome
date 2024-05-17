import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss


def train_model(data_loader, model, device_type, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = BCEWithLogitsLoss()

    model.train()
    for batch_idx, batch in tqdm(enumerate(data_loader)):
        batch = batch.to(device)

        optimizer.zero_grad()

        with autocast(device_type):
            out = model(batch)
            loss = criterion(out, batch.y.unsqueeze(-1).float())
            # Backward pass and optimize
        loss.backward()
        optimizer.step()
