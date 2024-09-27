import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import numpy as np
import torch 
from torch import nn
from armacell.arma_torch import ArmaCell

def test_AR() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p))
    kernel = np.arange(int(np.prod((cell.p, cell.units, cell.k, cell.k)))).reshape((cell.p, cell.units, cell.k, cell.k)).astype(float)
    cell.kernel = nn.Parameter(torch.tensor(kernel, dtype=torch.float32))
    recurrent_kernel = np.zeros((cell.q, cell.units, cell.k, cell.k)).astype(float)
    cell.recurrent_kernel = nn.Parameter(torch.tensor(recurrent_kernel, dtype=torch.float32))
    inputs = torch.tensor(np.ones((batch_size, k, p)).astype(float), dtype=torch.float32)
    state = (
        torch.tensor(np.zeros((batch_size, *cell.state_size)).astype(float), dtype=torch.float32),
    )
    res = cell(inputs, state)
    ar_output = res[0].detach().numpy()
    expected_output = np.apply_over_axes(np.sum, kernel, [0, 1, 2])
    assert np.all(ar_output[:, :, 0] == expected_output.flatten())

def test_AR_multiple_units() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p), units=2)
    kernel = np.arange(int(np.prod((cell.p, 1, cell.k, cell.k)))).reshape((cell.p, 1, cell.k, cell.k)).astype(float)
    kernel = np.tile(kernel, [1, 2, 1, 1])
    kernel[:, 1, :, :] = 2 * kernel[:, 1, :, :]
    cell.kernel = nn.Parameter(torch.tensor(kernel, dtype=torch.float32))
    recurrent_kernel = np.zeros((cell.q, cell.units, cell.k, cell.k)).astype(float)
    cell.recurrent_kernel = nn.Parameter(torch.tensor(recurrent_kernel, dtype=torch.float32))
    inputs = torch.tensor(np.ones((batch_size, k, p)).astype(float), dtype=torch.float32)
    state = (torch.tensor(np.zeros((batch_size, *cell.state_size)).astype(float), dtype=torch.float32),)

    res = cell(inputs, state)
    ar_output = res[0].detach().numpy()
    assert ((ar_output[:, 1::2, 0] / ar_output[:, 0::2, 0]) == 2).all()

def test_different_batches() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p))
    kernel = np.arange(int(np.prod((cell.p, cell.units, cell.k, cell.k)))).reshape((cell.p, cell.units, cell.k, cell.k)).astype(float)
    cell.kernel = nn.Parameter(torch.tensor(kernel, dtype=torch.float32))
    recurrent_kernel = np.zeros((cell.q, cell.units, cell.k, cell.k)).astype(float)
    cell.recurrent_kernel = nn.Parameter(torch.tensor(recurrent_kernel, dtype=torch.float32))
    input_array = np.ones((batch_size, k, p)).astype(float)
    for i in range(batch_size):
        input_array[i,] = (i + 1)
    inputs = torch.tensor(input_array, dtype=torch.float32)
    
    state = (torch.tensor(np.zeros((batch_size, *cell.state_size)).astype(float), dtype=torch.float32),)
    
    res = cell(inputs, state)
    output = res[0].detach().numpy()
    multiples = np.apply_over_axes(np.sum, output, [1, 2]).flatten()
    multiples = multiples / multiples.min()
    assert (multiples == np.array([1, 2, 3, 4, 5])).all()

def test_MA() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p))
    kernel = np.zeros((cell.p, cell.units, cell.k, cell.k)).astype(float)
    cell.kernel = nn.Parameter(torch.tensor(kernel, dtype=torch.float32))
    recurrent_kernel = (
        np.arange(int(np.prod((cell.q, cell.units, cell.k, cell.k))))
        .reshape((cell.q, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.recurrent_kernel = nn.Parameter(torch.tensor(recurrent_kernel, dtype=torch.float32))
    inputs = torch.tensor(np.zeros((batch_size, k, p)).astype(float), dtype=torch.float32)
    state = (torch.tensor(np.ones((batch_size, *cell.state_size)).astype(float), dtype=torch.float32),)
    res = cell(inputs, state)
    ma_output = res[0].detach().numpy()
    expected_output = np.apply_over_axes(np.sum, recurrent_kernel, [0, 1, 2]).flatten()
    assert np.all(ma_output[0, :, 0] == expected_output)

def test_MA_multiple_units() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p), units=2)
    kernel = np.zeros((cell.p, cell.units, cell.k, cell.k)).astype(float)
    cell.kernel = nn.Parameter(torch.tensor(kernel, dtype=torch.float32))
    recurrent_kernel = (
        np.arange(int(np.prod((cell.q, 1, cell.k, cell.k))))
        .reshape((cell.q, 1, cell.k, cell.k))
        .astype(float)
    )
    recurrent_kernel = np.tile(recurrent_kernel, [1, 2, 1, 1])
    recurrent_kernel[:, 1, :, :] = 2 * recurrent_kernel[:, 1, :, :]
    cell.recurrent_kernel = nn.Parameter(torch.tensor(recurrent_kernel, dtype=torch.float32))
    inputs = torch.tensor(np.zeros((batch_size, k, p)).astype(float), dtype=torch.float32)
    state = (torch.tensor(np.ones((batch_size, *cell.state_size)).astype(float), dtype=torch.float32),)
    res = cell(inputs, state)
    ma_output = res[0].detach().numpy()
    assert ((ma_output[0, 1::2, 0] / ma_output[0, 0::2, 0]) == 2).all()

def test_states() -> None:
    k = 4
    p = 3
    q = 3
    batch_size = 5
    cell = ArmaCell(q, (k, p))
    kernel = (
        np.arange(int(np.prod((cell.p, cell.units, cell.k, cell.k))))
        .reshape((cell.p, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.kernel = nn.Parameter(torch.tensor(kernel, dtype=torch.float32))
    recurrent_kernel = (
        np.arange(int(np.prod((cell.q, cell.units, cell.k, cell.k))))
        .reshape((cell.q, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.recurrent_kernel = nn.Parameter(torch.tensor(recurrent_kernel, dtype=torch.float32))
    inputs = torch.tensor(np.ones((batch_size, k, p)).astype(float), dtype=torch.float32)
    state = (torch.tensor(np.zeros((batch_size, *cell.state_size)).astype(float), dtype=torch.float32),)
    
    _, output_state = cell(inputs, state)
    _, output_state2 = cell(inputs, (output_state,))
    _, output_state3 = cell(inputs, (output_state2,))
    assert (output_state[0, :, 0].detach().numpy() == output_state2[0, :, 1].detach().numpy()).all()
    assert (output_state[0, :, 0].detach().numpy() == output_state3[0, :, 2].detach().numpy()).all() 
    assert (
        output_state3[0, :, 0].detach().numpy()
        == np.array([263688.0, 300756.0, 337824.0, 374892.0])
    ).all()

def test_return_state() -> None:
    k = 1
    p = 3
    q = 3
    batch_size = 5
    cell = ArmaCell(q, (k, p), return_lags=True)
    kernel = (
        np.arange(int(np.prod((cell.p, cell.units, cell.k, cell.k))))
        .reshape((cell.p, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.kernel = nn.Parameter(torch.tensor(kernel, dtype=torch.float32))
    recurrent_kernel = (
        np.arange(int(np.prod((cell.q, cell.units, cell.k, cell.k))))
        .reshape((cell.q, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.recurrent_kernel = nn.Parameter(torch.tensor(recurrent_kernel, dtype=torch.float32))
    inputs = torch.tensor(np.ones((batch_size, k, p)).astype(float), dtype=torch.float32)
    state = (torch.tensor(np.zeros((batch_size, *cell.state_size)).astype(float), dtype=torch.float32),)
    output, output_state = cell(inputs, state)
    assert (output.detach().numpy() == output_state.detach().numpy()).all()

def test_nonlinear_activation() -> None:
    k = 4
    p = 3
    batch_size = 5
    cell = ArmaCell(2, (k, p), activation="tanh")
    kernel = (
        np.arange(int(np.prod((cell.p, cell.units, cell.k, cell.k))))
        .reshape((cell.p, cell.units, cell.k, cell.k))
        .astype(float)
    )
    cell.kernel = nn.Parameter(torch.tensor(kernel, dtype=torch.float32))
    recurrent_kernel = np.zeros((cell.q, cell.units, cell.k, cell.k)).astype(float)
    cell.recurrent_kernel = nn.Parameter(torch.tensor(recurrent_kernel, dtype=torch.float32))
    inputs = torch.tensor(np.ones((batch_size, k, p)).astype(float), dtype=torch.float32)
    state = (torch.tensor(np.zeros((batch_size, *cell.state_size)).astype(float), dtype=torch.float32),)
    res = cell(inputs, state)
    ar_output = res[0].detach().numpy()
    expected_output = np.tanh(np.apply_over_axes(np.sum, kernel, [0, 1, 2]).flatten())
    assert np.all(ar_output[0, :, 0] == expected_output)


test_AR()
test_AR_multiple_units()
test_different_batches()
test_MA()
test_MA_multiple_units()
test_states()
test_return_state()
test_nonlinear_activation()
print("All tests passed!")