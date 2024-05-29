from torch_scatter import scatter_mean, scatter_std


class TrainingError(Exception):
    pass


def min_max_norm(x):
    return (x - x.min(dim=1, keepdim=True).values) / (
        x.max(dim=1, keepdim=True).values - x.min(dim=1, keepdim=True).values
    )


def normalize_non_zero(x, batch, epsilon=1e-5):
    # x has shape [batch_size * num_neurons, num_features]
    # batch has shape [batch_size * num_neurons]
    batch_size = batch.max().item() + 1
    non_zero_mask = x != 0

    non_zero_entries = x[non_zero_mask]
    non_zero_batch = batch[non_zero_mask.squeeze()]

    mean_per_batch = scatter_mean(
        non_zero_entries, non_zero_batch, dim=0, dim_size=batch_size
    )
    std_per_batch = (
        scatter_std(non_zero_entries, non_zero_batch, dim=0, dim_size=batch_size)
        + epsilon
    )

    x[non_zero_mask] = (
        non_zero_entries - mean_per_batch[non_zero_batch]
    ) / std_per_batch[non_zero_batch]

    return x


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
                            Default: 10
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def _update_stopper(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def should_stop(self, loss):
        self._update_stopper(loss)
        return self.early_stop
