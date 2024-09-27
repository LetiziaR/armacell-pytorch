import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from armacell.helpers_torch import restore_arma_parameters, SaveWeights, simulate_arma_process, simulate_varma_process, prepare_arma_input, set_all_seeds
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from armacell.arma_torch import ARMA

def get_trained_ARMA_p_q_model(q, X_train, y_train, units, add_intercept=False, plot_training=False, **kwargs):
    input_dim = (X_train.shape[-2], X_train.shape[-1]) 
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            epoch_loss += loss.item()

        if plot_training:
            weights_saver.on_epoch_end(model, epoch)

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

    #if plot_training:
    #    plot_convergence(weights_history, p, add_intercept, arima_model)

    weights_list = [
        model.arma_cell.kernel.detach().cpu().numpy(), 
        model.arma_cell.recurrent_kernel.detach().cpu().numpy()
    ]
    if add_intercept:
        weights_list.append(model.arma_cell.bias.detach().cpu().numpy())

    beta, gamma, alpha = restore_arma_parameters(weights_list, p, add_intercept)


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

    arima_model = ARIMA(endog=y, order=(p, 0, q), trend="n").fit()  

    X_train, y_train = prepare_arma_input(max(p, q), y, sequence_length=10)
    Y_train = np.stack([y_train, y_train], axis=-1)
    
    model, _ = get_trained_ARMA_p_q_model(
        q, X_train, Y_train, units=2, plot_training=True
    )

    raw_ar_weights = model.arma_cell.kernel.detach().cpu().numpy()
    raw_ma_weights = model.arma_cell.recurrent_kernel.detach().cpu().numpy()
    
    weights1 = [raw_ar_weights[:, 0, :, :], raw_ma_weights[:, 0, :, :]]
    weights2 = [raw_ar_weights[:, 1, :, :], raw_ma_weights[:, 1, :, :]]
    
    beta1, gamma1, _ = restore_arma_parameters(weights1, p)
    beta2, gamma2, _ = restore_arma_parameters(weights2, p)

    assert np.all(np.abs(beta1 - arima_model.arparams) < 0.05)
    assert np.all(np.abs(gamma1 - arima_model.maparams) < 0.05)

    assert np.all(np.abs(beta2 - arima_model.arparams) < 0.05)
    assert np.all(np.abs(gamma2 - arima_model.maparams) < 0.05)

def test_VARMA_1_1_2() -> None:
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

    X_train, y_train = prepare_arma_input(max(p, q), y, sequence_length=10)
    model, _ = get_trained_ARMA_p_q_model(q, X_train, y_train, units=1, plot_training=True)

    weights_list = [
        model.arma_cell.kernel.detach().cpu().numpy(),
        model.arma_cell.recurrent_kernel.detach().cpu().numpy()
    ]
    
    gamma = -weights_list[1][0, 0].T
    beta =  weights_list[0][0, 0].T - gamma

    assert np.all(np.abs(beta - varma_model.coefficient_matrices_var[0]) < 0.05)
    assert np.all(np.abs(gamma - varma_model.coefficient_matrices_vma[0]) < 0.05)

