import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple

class ConvARMACell(nn.Module):
    def __init__(
        self,
        q: int,
        image_shape: Tuple[Any, ...],
        units: int = 1,
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = "relu",
        use_bias: bool = True,
        return_lags: bool = True,
        **kwargs: dict
    ) -> None:
        super(ConvARMACell, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.activation = getattr(F, activation) if activation != "linear" else lambda x: x
        self.image_shape = image_shape
        self.q = q
        self.use_bias = use_bias
        self.return_lags = return_lags

        # Set during build()
        self.kernel = None
        self.recurrent_kernel = None
        self.ar_parameters = None
        self.ma_parameters = None
        self.p = None
        self.arma_bias = None
        self.ar_conv_bias = None
        self.ma_conv_bias = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        input_dim = input_shape[-1] 
        self.p = input_shape[-4]
        assert self.image_shape == tuple(input_shape[-4:]), (
            self.image_shape,
            input_shape[-4:],
        )
        kernel_shape = (self.p, self.units) + self.kernel_size + (input_dim,)
        self.kernel = nn.Parameter(torch.Tensor(*kernel_shape))
        
        recurrent_kernel_shape = (self.q, self.units) + self.kernel_size + (self.units,)
        self.recurrent_kernel = nn.Parameter(torch.Tensor(*recurrent_kernel_shape))

        self.ar_parameters = nn.Parameter(torch.Tensor(self.units, self.p))
        self.ma_parameters = nn.Parameter(torch.Tensor(self.units, self.q))

        if self.use_bias:
            self.arma_bias = nn.Parameter(torch.Tensor(self.units))
            self.ar_conv_bias = nn.Parameter(torch.Tensor(self.p, self.units))
            self.ma_conv_bias = nn.Parameter(torch.Tensor(self.q, self.units))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel)
        nn.init.xavier_uniform_(self.recurrent_kernel)
        nn.init.xavier_uniform_(self.ar_parameters)
        nn.init.xavier_uniform_(self.ma_parameters)
        if self.use_bias:
            nn.init.zeros_(self.arma_bias)
            nn.init.zeros_(self.ar_conv_bias)
            nn.init.zeros_(self.ma_conv_bias)

    def forward(self, inputs: torch.Tensor, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_states = states[0]
        
        # AR part
        ar_convs = []
        for i in range(self.p):
            conv_out = F.conv2d(
                inputs[:, i],
                self.kernel[i].permute(0, 3, 1, 2),
                bias=self.ar_conv_bias[i] if self.use_bias else None,
                padding='same'
            )
            ar_convs.append(conv_out)
        ar = torch.stack(ar_convs, dim=-1) * self.ar_parameters.unsqueeze(1).unsqueeze(2)

        # MA part
        ma_convs = []
        for i in range(self.q):
            conv_out = F.conv2d(
                input_states[:, i],
                self.recurrent_kernel[i].permute(0, 3, 1, 2),
                bias=self.ma_conv_bias[i] if self.use_bias else None,
                padding='same'
            )
            ma_convs.append(conv_out)
        ma = torch.stack(ma_convs, dim=-1) * self.ma_parameters.unsqueeze(1).unsqueeze(2)

        output = torch.sum(ar, dim=-1) + torch.sum(ma, dim=-1)
        if self.use_bias:
            output = output + self.arma_bias.view(1, -1, 1, 1)
        output = self.activation(output)

        output_states = torch.cat([output.unsqueeze(1), input_states[:, :-1]], dim=1)

        if self.return_lags:
            return output_states, output_states
        else:
            return output, output_states

class ConvARMA(nn.Module):
    def __init__(
        self,
        q: int,
        image_shape: Tuple[Any, ...],
        units: int = 1,
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = "relu",
        use_bias: bool = True,
        return_lags: bool = True,
        return_sequences: bool = False,
        **kwargs: dict
    ) -> None:
        super(ConvARMA, self).__init__()
        self.cell = ConvARMACell(
            q, image_shape, units, kernel_size, activation, use_bias, return_lags, **kwargs
        )
        self.return_sequences = return_sequences

    def forward(self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape[:2]
        if initial_state is None:
            initial_state = torch.zeros(batch_size, self.cell.q, *self.cell.image_shape[1:-1], self.cell.units, device=x.device)

        outputs = []
        state = initial_state

        for t in range(seq_len):
            output, state = self.cell(x[:, t], state)
            outputs.append(output)

        if self.return_sequences:
            return torch.stack(outputs, dim=1), state
        else:
            return outputs[-1], state

class SpatialDiffs(nn.Module):
    def __init__(self, shift_val: int):
        super(SpatialDiffs, self).__init__()
        self.shift_val = shift_val

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shift_val = self.shift_val
        
        diff_1 = torch.roll(inputs, shifts=shift_val, dims=-3) - inputs
        diff_1[..., :shift_val, :, :] = 0

        diff_2 = torch.roll(inputs, shifts=-shift_val, dims=-3) - inputs
        diff_2[..., -shift_val:, :, :] = 0

        diff_3 = torch.roll(inputs, shifts=shift_val, dims=-2) - inputs
        diff_3[..., :, :shift_val, :] = 0

        diff_4 = torch.roll(inputs, shifts=-shift_val, dims=-2) - inputs
        diff_4[..., :, -shift_val:, :] = 0

        inputs_and_diffs = torch.cat([inputs, diff_1, diff_2, diff_3, diff_4], dim=-1)
        return inputs_and_diffs
    

