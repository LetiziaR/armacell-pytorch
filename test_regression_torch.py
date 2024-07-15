import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from typing import Tuple, Any
from helpers_torch import (restore_arma_parameters, SaveWeights, simulate_arma_process, simulate_varma_process, prepare_arma_input, set_all_seeds)
from plotting_torch import (plot_convergence)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from arma_torch import ARMA

def get_trained_ARMA_p_q_model(q, X_train, y_train, units, add_intercept=False, plot_training=False, **kwargs):
    input_dim = (X_train.shape[-2], X_train.shape[-1])  # Assuming input_dim should be (batch_size, time_steps)
    model = ARMA(q=q, input_dim=input_dim, units=units, use_bias=add_intercept, **kwargs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()

    weights_saver = SaveWeights()

    epochs = 100
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()

        if plot_training:
            weights_saver.on_epoch_end(model, epoch)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train)}')

    return model, weights_saver.weights_history


def run_p_q_test(
    arparams: np.ndarray,
    maparams: np.ndarray,
    alpha_true: float = 0,
    plot_training: bool = False,
    **kwargs: int
) -> None:
    p = len(arparams)
    q = len(maparams)
    add_intercept = alpha_true != 0

    y = simulate_arma_process(arparams, maparams, alpha_true, n_steps=25000, std=2)

    arima_model = ARIMA(
        endog=y, order=(p, 0, q), trend="c" if add_intercept else "n"
    ).fit()

    X_train, y_train = prepare_arma_input(max(p, q), y, sequence_length=10)
    model, weights_history = get_trained_ARMA_p_q_model(q, X_train, y_train, units=1, add_intercept=add_intercept, plot_training=plot_training, **kwargs)

    if plot_training:
        plot_convergence(weights_history, p, add_intercept, arima_model)

    # Access model weights directly as a list
    weights_list = [
        model.arma_cell.kernel.detach().cpu().numpy(), 
        model.arma_cell.recurrent_kernel.detach().cpu().numpy()
    ]
    if add_intercept:
        weights_list.append(model.arma_cell.bias.detach().cpu().numpy())

    beta, gamma, alpha = restore_arma_parameters(weights_list, p, add_intercept)

    print("Learned AR parameters (beta):", beta)
    print("ARIMA model AR parameters:", arima_model.arparams)
    print("Learned MA parameters (gamma):", gamma)
    print("ARIMA model MA parameters:", arima_model.maparams)

    if add_intercept:
        print("Learned intercept (alpha):", alpha)
        print("ARIMA model intercept:", arima_model.params[0])

    assert np.all(np.abs(beta - arima_model.arparams) < 0.05)
    assert np.all(np.abs(gamma - arima_model.maparams) < 0.05)
    if add_intercept:
        assert np.all(np.abs(alpha - arima_model.params[0]) < 0.05)


def test_ARMA_1_1() -> None:
    set_all_seeds()
    arparams = np.array([0.1])
    maparams = np.array([-0.4])
    run_p_q_test(arparams, maparams)

def test_ARMA_2_1() -> None:
    set_all_seeds()
    arparams = np.array([0.1, 0.3])
    maparams = np.array([-0.4])
    run_p_q_test(arparams, maparams)

def test_ARMA_2_2() -> None:
    set_all_seeds()
    arparams = np.array([0.1, 0.3])
    maparams = np.array([-0.4, -0.2])
    run_p_q_test(arparams, maparams)

def test_ARMA_1_2() -> None:
    set_all_seeds()
    arparams = np.array([0.3])
    maparams = np.array([-0.4, -0.2])
    run_p_q_test(arparams, maparams, p=1)

def test_ARMA_1_1_bias() -> None:
    set_all_seeds()
    arparams = np.array([0.1])
    maparams = np.array([-0.4])
    alpha = 0.2
    run_p_q_test(arparams, maparams, alpha)

def test_ARMA_1_2_bias() -> None:
    set_all_seeds()
    arparams = np.array([0.1])
    maparams = np.array([-0.4, -0.2])
    alpha = 0.2
    run_p_q_test(arparams, maparams, alpha, p=1)

def test_ARMA_2_2_multi_unit() -> None:
    set_all_seeds()
    arparams = np.array([0.1, 0.3])
    maparams = np.array([-0.4, -0.2])
    p = len(arparams)
    q = len(maparams)

    y = simulate_arma_process(arparams, maparams, 0, n_steps=25000, std=2)

    arima_model = ARIMA(endog=y, order=(p, 0, q), trend="n").fit()  # order = (p,d,q)

    X_train, y_train = prepare_arma_input(max(p, q), y, sequence_length=10)
    Y_train = np.stack([y_train, y_train], axis=-1)
    
    model, weights_history = get_trained_ARMA_p_q_model(
        q, X_train, Y_train, units=2, plot_training=True
    )

    # Extract weights for each unit
    raw_ar_weights = model.arma_cell.kernel.detach().cpu().numpy()
    raw_ma_weights = model.arma_cell.recurrent_kernel.detach().cpu().numpy()
    
    weights1 = [raw_ar_weights[:, 0, :, :], raw_ma_weights[:, 0, :, :]]
    weights2 = [raw_ar_weights[:, 1, :, :], raw_ma_weights[:, 1, :, :]]
    
    beta1, gamma1, _ = restore_arma_parameters(weights1, p)
    beta2, gamma2, _ = restore_arma_parameters(weights2, p)

    # Print learned and true parameters for debugging
    print("Learned AR parameters (beta1):", beta1)
    print("ARIMA model AR parameters:", arima_model.arparams)
    print("Learned MA parameters (gamma1):", gamma1)
    print("ARIMA model MA parameters:", arima_model.maparams)

    # Target 1
    assert np.all(np.abs(beta1 - arima_model.arparams) < 0.05), f"AR parameters (beta1) do not match. Difference: {np.abs(beta1 - arima_model.arparams)}"
    assert np.all(np.abs(gamma1 - arima_model.maparams) < 0.05), f"MA parameters (gamma1) do not match. Difference: {np.abs(gamma1 - arima_model.maparams)}"

    # Print learned and true parameters for the second unit
    print("Learned AR parameters (beta2):", beta2)
    print("ARIMA model AR parameters:", arima_model.arparams)
    print("Learned MA parameters (gamma2):", gamma2)
    print("ARIMA model MA parameters:", arima_model.maparams)

    # Target 2
    assert np.all(np.abs(beta2 - arima_model.arparams) < 0.05), f"AR parameters (beta2) do not match. Difference: {np.abs(beta2 - arima_model.arparams)}"
    assert np.all(np.abs(gamma2 - arima_model.maparams) < 0.05), f"MA parameters (gamma2) do not match. Difference: {np.abs(gamma2 - arima_model.maparams)}"
"""
def test_VARMA_1_1_2():
    set_all_seeds()
    VAR = np.array([[0.1, -0.2], [0.0, 0.1]])
    VAR = np.expand_dims(VAR, axis=-1)
    VMA = np.array([[-0.4, 0.2], [0.0, -0.4]])
    VMA = np.expand_dims(VMA, axis=-1)
    alpha = np.zeros(2)
    y = simulate_varma_process(VAR, VMA, alpha, n_steps=10000)
    p = 1
    q = 1

    varma_model = VARMAX(y, order=(1, 1)).fit()

    sequence_length = 10 
    X_train, y_train = prepare_arma_input(max(p, q), y, sequence_length)
    
    
    # Train the ARMA model with 2 units to match the bivariate nature of the data
    model, weights_history = get_trained_ARMA_p_q_model(q, X_train, y_train, units=2)
    
    # Extract AR and MA coefficients
    ar_coef = model.arma_cell.kernel.detach().numpy()
    ma_coef = model.arma_cell.recurrent_kernel.detach().numpy()
    
    # Reshape coefficients to match VARMAX format
    ar_coef_reshaped = ar_coef[0, :, :, :].transpose(1, 2, 0)
    ma_coef_reshaped = ma_coef[0, :, :, :].transpose(1, 2, 0)

    print("ARMA AR coefficients:")
    print(ar_coef_reshaped)
    print("VARMAX AR coefficients:")
    print(varma_model.coefficient_matrices_var[0])
    print("ARMA MA coefficients:")
    print(ma_coef_reshaped)
    print("VARMAX MA coefficients:")
    print(varma_model.coefficient_matrices_vma[0])

    # Compare coefficients
    ar_diff = np.abs(ar_coef_reshaped - varma_model.coefficient_matrices_var[0])
    ma_diff = np.abs(ma_coef_reshaped - varma_model.coefficient_matrices_vma[0])
    
    print("AR coefficient differences:")
    print(ar_diff)
    print("MA coefficient differences:")
    print(ma_diff)

    # Assert that the differences are within a tolerance
    tolerance = 0.1  # You may need to adjust this based on your model's performance
    assert np.all(ar_diff < tolerance), "AR coefficients do not match within tolerance"
    assert np.all(ma_diff < tolerance), "MA coefficients do not match within tolerance"

    print("VARMA test passed successfully!")
"""
test_ARMA_1_1()
test_ARMA_2_1() 
test_ARMA_2_2()
test_ARMA_1_2()
test_ARMA_1_1_bias()
test_ARMA_1_2_bias()
test_ARMA_2_2_multi_unit()
#test_VARMA_1_1_2()
