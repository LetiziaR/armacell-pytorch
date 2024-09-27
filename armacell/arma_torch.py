import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from typing import Tuple, Optional, Any

class ArmaCell(nn.Module):
    def __init__(
        self,
        q: int,
        input_dim: Tuple[int, int],
        p: Optional[int] = None,
        units: int = 1,
        activation: str = "linear",
        use_bias: bool = False,
        return_lags: bool = False,
        **kwargs: Any
    ):
        super(ArmaCell, self).__init__(**kwargs)
        self.units = units
        self.activation = getattr(F, activation) if activation != "linear" else lambda x: x
        self.q = q
        self.p = p if p is not None else input_dim[1]
        self.k = input_dim[0]
        assert self.p <= input_dim[1]
        assert self.p > 0
        assert self.q > 0
        assert self.k > 0
        self.q_overhang = self.q > self.p
        self.use_bias = use_bias
        self.return_lags = return_lags

        self.kernel = nn.Parameter(torch.Tensor(self.p, self.units, self.k, self.k))
        self.recurrent_kernel = nn.Parameter(torch.Tensor(self.q, self.units, self.k, self.k))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.k * self.units))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    @property
    def state_size(self) -> torch.Size:
        return torch.Size((self.k * self.units, self.q))

    @property
    def output_size(self) -> torch.Size:
        return (
            torch.Size((self.k * self.units, self.q))
            if self.return_lags
            else torch.Size((self.k * self.units, 1))
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel)
        nn.init.xavier_uniform_(self.recurrent_kernel)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, states):
        input_state = states[0]
        input_state = input_state.unsqueeze(-2)
        input_state = input_state.reshape(-1, self.k, self.units, self.q)

        ar_out = []
        for i in range(self.p):
            ar_out.append(torch.matmul(inputs[:, :, i], self.kernel[i, :, :, :]))
        ar = sum(ar_out)
        ar = ar.permute(1, 2, 0)
        ar = ar.reshape(-1, self.k * self.units)

        ma_out = []
        for i in range(self.q):
            ma_unit = []
            if i + 1 > self.p:
                lhs = input_state - inputs.unsqueeze(dim=-2)
            else:
                lhs = input_state
            for j in range(self.units):
                ma_unit.append(torch.matmul(lhs[:, :, j, i], self.recurrent_kernel[i, j, :, :]))  
            ma_out.append(torch.stack(ma_unit, dim=-1))
        ma = sum(ma_out)
        ma = ma.reshape(-1, self.k * self.units)

        output = ar + ma

        if self.use_bias:
            output = output + self.bias

        output = self.activation(output)
        output = output.unsqueeze(dim=-1)
        output_state = torch.cat([output, states[0][:, :, :-1]], dim=-1)

        if self.return_lags:
            return output_state, output_state
        else:
            return output, output_state

class ARMA(nn.Module):
    def __init__(self, q, input_dim, use_bias=False, **kwargs):
        super(ARMA, self).__init__()
        self.arma_cell = ArmaCell(q=q, input_dim=input_dim, use_bias=use_bias, **kwargs)
        
    def forward(self, x, state):
        outputs = []
        for i in range(x.size(1)):
            out, state = self.arma_cell(x[:, i, :], (state,))
            outputs.append(out)
        outputs = torch.cat(outputs, dim=2)
        outputs = outputs[:, :, -1]
        return outputs, state