# ARMAcell in PyTorch 

This repo contains the PyTorch implementation of the ARMA cell, a neural network cell designed to model time series using a modular and simpler approach compared to traditional Recurrent Neural Network (RNN) cells like Long Short-Term Memory (LSTM) cells. It builds on the autoregressive moving average (ARMA) model, a classical statistical tool for time series analysis.     
The methodology is described in detail in the paper Schiele, P., Berninger, C., & Rügamer, D. (2024). [ARMA Cell: A Modular and Effective Approach for Neural Autoregressive Modeling](https://arxiv.org/abs/2208.14919). The official implementation of the ARMA cell in TensorFlow can be found at the [armacell](https://github.com/phschiele/armacell_paper) repository. Additionally, data and code to reproduce the experiments in the original paper can be found at the [armacell_paper](https://github.com/phschiele/armacell_paper) repository.



## Getting started

The ARMA cell can be used similarly to other PyTorch modules.The syntax of the PyTorch implementation of the ARMA cell is similiar to the Tensorflow  one. Where `q` is the number of MA lags, wheras the number of AR lags is already represented in the preprocessed data, which is handled by `prepare_arma_input`.

Below is an example using the PyTorch function.

```python
x = ARMA(q, input_dim=(n_features, p), units=1, activation="relu", use_bias=True)(x)
```

 To use our repository, first clone it and install the required packages. 
 
```bash
conda create -n arma_pytorch python=3.10
conda activate arma_pytorch
pip install -r requirements.txt
```

## Test
Unit and regression tests are handled through `pytest`, which can be installed via `pip install pytest`.
To run all tests, simply execute
```shell
pytest
```
from the root of the repository. To check all the test files please refer to the test folder.


## Comparison between TensorFlow and PyTorch Implementations

The main differences between the orignal repository's TensorFlow implementation and the PyTorch implementation are in how they structure classes, set up parameters, and handle activations and tensor operations. For class structure, TensorFlow uses special RNN classes like `AbstractRNNCell` and `RNN`. PyTorch, however, uses the basic `nn.Module` class for both `ArmaCell` and `ARMA`.  this allows for easy integration into existing PyTorch models and facilitates the use of standard training and evaluation procedures. 

When it comes to setting up parameters, TensorFlow uses `self.add_weight()` in the `build()` method. PyTorch does this differently, using `nn.Parameter()` in the `__init__()` method.

For activation functions and tensor operations,the TensorFlow implementation uses `tf.keras.activations.deserialize()` for activations, and operations like `tf.concat` and `tf.expand_dims` for tensors. PyTorch gets its activation functions from `torch.nn.functional` and uses operations such as `torch.cat` and `unsqueeze` for tensors. which clearly shows the frameworks' different APIs.

These differences show how TensorFlow and PyTorch approach things differently. TensorFlow has more built-in tools for RNNs, which can make some tasks easier. PyTorch gives you more control over the details, which can be more flexible. Both ways have their good points, and the choice between them often depends on what you need for your project and what you prefer to work with.


## Minimum working example
```python
import torch
from torch import nn, optim
from arma_torch import ARMA
from plotting_torch import plot_convergence
from helpers_torch import (simulate_arma_process, prepare_arma_input, set_all_seeds, SaveWeights)
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# 1. Obtain time series data. Simulating an ARMA process here for simplicity

arparams = np.array([0.1, 0.3])
maparams = np.array([-0.4, -0.2])
alpha = 0
set_all_seeds()

#  Generate data
y = simulate_arma_process(arparams, maparams, alpha, n_steps=25000, std=2)

# 2. Data pre-processing.
# In practice, p and q are hyperparameters. Here, we use the true values.
p, q = len(arparams), len(maparams)
X_train, y_train = prepare_arma_input(max(p, q), y)

# 3. Train and Fitthe model
def get_trained_ARMA_p_q_model(q, X_train, y_train, units, add_intercept=False, plot_training=False, **kwargs):
    input_dim = (X_train.shape[-2], X_train.shape[-1])
    # Assuming input_dim should be (batch_size, time_steps)
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()

        if plot_training:
            weights_saver.on_epoch_end(model, epoch)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train)}')

    return model, weights_saver.weights_history

model, weights_history = get_trained_ARMA_p_q_model(q, X_train, y_train, units=1, add_intercept=False, plot_training=True)


# 4. Fit a classical ARMA model for comparison
arma_model = ARIMA(endog=y, order=(p, 0, q), trend="n").fit()

# 5. Plot the result
plot_convergence(weights_history, p, add_intercept=False, arima_model=arma_model, path="image.png")
```

Looking at the convergence plot, similiarly to the TensorFlow implementation, the ARMA cell converged to the true parameters at least as good as a classical ARIMA model.

![convergence plot](example/image.png)




## Acknowledgments
We wish to express our gratitude to Dr. Schiele and Prof. Rügamer for providing guidance and support. The code base is based on the orignal implementation of the ARMA cell in TensorFlow, which can be found at the [armacell](https://github.com/phschiele/armacell_paper) repository. We also thank the authors for their open-sourced code.


## License
See [`LICENSE.md`](LICENSE.md) for details.


