import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Any


class SaveWeights:
    def __init__(self):
        self.weights_history = []

    def on_epoch_end(self, model: torch.nn.Module, epoch: int) -> None:
        self.weights_history.append([param.detach().clone().numpy() for param in model.parameters()])
        
def restore_arma_parameters(
    model_weights: List[np.ndarray], p: int, add_intercept: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

    validate_weight_list_length(model_weights, add_intercept)

    beta_breve = model_weights[0].flatten()
    gamma_breve = model_weights[1].flatten()

    gamma = -gamma_breve
    if len(gamma) < len(beta_breve):
        beta = beta_breve - np.pad(gamma, (0, len(beta_breve) - len(gamma)))
    else:
        beta = beta_breve - gamma[:p]

    if add_intercept:
        alpha = model_weights[-1].flatten()
    else:
        alpha = None
    return beta[:p], gamma, alpha

def validate_weight_list_length(weight_list: list, add_intercept: bool) -> None:
    assert len(weight_list) == 3 if add_intercept else 2

def simulate_arma_process(
    ar: np.ndarray,
    ma: np.ndarray,
    alpha: float,
    n_steps: int = 1000,
    std: float = 1.0,
    burn_in: int = 50,
    ) -> np.ndarray:
    steps_incl_burn_in = n_steps + burn_in
    eps = np.random.normal(0, std, steps_incl_burn_in)
    res = np.zeros(steps_incl_burn_in)
    for i in range(steps_incl_burn_in):
        res[i] = alpha + eps[i]
        for j, ar_par in enumerate(ar):
            if i > j:
                res[i] += res[i - j - 1] * ar_par
        for j, ma_par in enumerate(ma):
            if i > j:
                res[i] += eps[i - j - 1] * ma_par
    return res[burn_in:]

def simulate_varma_process(
    ar: np.ndarray,
    ma: np.ndarray,
    alpha: np.ndarray,
    n_steps: int = 1000,
    std: float = 1.0,
    burn_in: int = 50,
) -> np.ndarray:
    assert ar.ndim == ma.ndim == 3
    assert ar.shape[0] == ar.shape[1] == ma.shape[0] == ma.shape[1]
    assert isinstance(alpha, np.ndarray)

    k = ar.shape[0]

    steps_incl_burn_in = n_steps + burn_in
    eps = np.random.normal(0, std, (steps_incl_burn_in, k))

    res = np.zeros((steps_incl_burn_in, k))

    for i in range(steps_incl_burn_in):
        res[i, :] = alpha + eps[i, :]
        for j in range(ar.shape[2]):
            if i > j:
                res[i, :] += ar[:, :, j] @ res[i - j - 1, :]
        for j in range(ma.shape[2]):
            if i > j:
                res[i, :] += ma[:, :, j] @ eps[i - j - 1, :]
    return res[burn_in:, :]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_arma_input(
    p: int, endog: np.ndarray, sequence_length: int 
) -> Tuple[np.ndarray, np.ndarray]:
    if endog.ndim == 1:
        endog = endog.reshape((-1, 1))
    endog = np.expand_dims(endog, axis=-1)
    endog_rep = np.concatenate(
        [endog[p - 1:, ...]] + [endog[p - i - 1:-i, ...] for i in range(1, p)],
        axis=-1,
    )

    dataset = TimeSeriesDataset(endog_rep, sequence_length)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    X, y = next(iter(dataloader))
    X = X.numpy()
    y = y.numpy()
    y = y[..., 0]

    return X, y


def set_all_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)