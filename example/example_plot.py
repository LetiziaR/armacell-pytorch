import torch
from torch import nn, optim
from armacell.arma_torch import ARMA
from armacell.plotting_torch import plot_convergence
from armacell.helpers_torch import simulate_arma_process, prepare_arma_input, set_all_seeds, SaveWeights
from statsmodels.tsa.arima.model import ARIMA
import numpy as np


# Example usage  
arparams = np.array([0.1, 0.3])
maparams = np.array([-0.4, -0.2])
alpha = 0
set_all_seeds()

# 1. Generate data
y = simulate_arma_process(arparams, maparams, alpha, n_steps=25000, std=2)


# 2. Data pre-processing
p, q = len(arparams), len(maparams)
X_train, y_train = prepare_arma_input(max(p, q), y, sequence_length=10)

# 3. Train the model

def get_trained_ARMA_p_q_model(q, X_train, y_train, units, add_intercept=False, plot_training=False, **kwargs):
    input_dim = (X_train.shape[-2], X_train.shape[-1]) 
    model = ARMA(q=q, input_dim=input_dim, units=units, use_bias=add_intercept, **kwargs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()

    weights_saver = SaveWeights()

    epochs = 80
    batch_size = 100
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            optimizer.zero_grad()
            state = torch.zeros((X_batch.size(0), model.arma_cell.state_size[0], model.arma_cell.state_size[1]))
            outputs, state = model(X_batch, state)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            epoch_loss += loss.item()

        if plot_training:
            weights_saver.on_epoch_end(model, epoch)

    return model, weights_saver.weights_history

model, weights_history = get_trained_ARMA_p_q_model(q, X_train, y_train, units=1, add_intercept=False, plot_training=True)

# 4. Fit a classical ARMA model for comparison
arma_model = ARIMA(endog=y, order=(p, 0, q), trend="n").fit()

# 5. Plot the result
plot_convergence(weights_history, p, add_intercept=False, arima_model=arma_model, path="image.png")