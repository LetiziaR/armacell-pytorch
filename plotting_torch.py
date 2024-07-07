from typing import List
from statsmodels.tsa.arima.model import ARIMA
from helpers_torch import restore_arma_parameters
from matplotlib import pyplot as plt
import numpy as np  

def plot_convergence(weights_history: List[List[np.ndarray]], p: int, add_intercept: bool, arima_model: ARIMA, path: str = "") -> None:
    
    transformed_parameters = []
    for weights in weights_history:
        params = restore_arma_parameters(weights, p, add_intercept)
        transformed_parameters.append(params)
    
    beta = np.stack([params[0] for params in transformed_parameters])
    gamma = np.stack([params[1] for params in transformed_parameters])

    plt.figure(figsize=(15, 5))
    plt.axhline(y=0, color="#909090", linestyle="-")

    for i in range(beta.shape[1]):
        plt.axhline(arima_model.arparams[i], c="g", linestyle="--")
        if i > 0:
            plt.plot(beta[:, i], c="g")
        else:
            plt.plot(beta[:, i], c="g", label="AR")

    for i in range(gamma.shape[1]):
        plt.axhline(arima_model.maparams[i], c="r", linestyle="--")
        if i > 0:
            plt.plot(gamma[:, i], c="r")
        else:
            plt.plot(gamma[:, i], c="r", label="MA")

    if add_intercept:
        alpha = np.stack([params[2] for params in transformed_parameters if params[2] is not None])
        plt.plot(alpha, c="b", label="Intercept")
        plt.axhline(arima_model.params[0], c="b", linestyle="--")

    plt.xlim(0, len(transformed_parameters) - 1)
    plt.xlabel("Epochs")
    plt.ylabel("Coefficient Value")
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if path:
        plt.savefig(path)
    plt.show()
