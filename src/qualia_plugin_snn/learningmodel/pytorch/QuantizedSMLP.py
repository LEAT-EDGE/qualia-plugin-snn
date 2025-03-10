"""Contains the template for a quantized spiking multi-layer perceptron."""

from __future__ import annotations

import math
import sys
from collections import OrderedDict

from qualia_core.typing import TYPE_CHECKING
from torch import nn

from .SNN import SNN

if TYPE_CHECKING:
    import torch
    from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig  # noqa: TC002
    from qualia_core.typing import RecursiveConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class QuantizedSMLP(SNN):
    """Quantized spiking multi-layer perceptron template.

    Should have topology identical to :class:`qualia_plugin_snn.learningmodel.pytorch.SMLP.SMLP` but with layers replaced with
    their quantized equivalent.
    """

    def __init__(self,  # noqa: PLR0913
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 units: list[int],
                 quant_params: QuantizationConfig,
                 timesteps: int,
                 neuron: RecursiveConfigDict | None = None) -> None:
        """Construct :class:`QuantizedSMLP`.

        :param input_shape: Input shape
        :param output_shape: Output shape
        :param units: List of :class:`torch.nn.Linear` layer ``out_features`` to add in the network
        :param quant_params: Quantization configuration dict, see :class:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer`
        :param neuron: Spiking neuron configuration, see :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.__init__`
        :param timesteps: Number of timesteps
        """
        # Prepend Quantized to the neuron kind to instantiate quantized spiking neurons
        if neuron is not None and 'kind' in neuron and isinstance(neuron['kind'], str):
            neuron['kind'] = 'Quantized' + neuron['kind']

        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         timesteps=timesteps,
                         neuron=neuron)

        from spikingjelly.activation_based.layer import Flatten  # type: ignore[import-untyped]

        from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly.quantized_layers import QuantizedLinear

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        layers['flatten1'] = Flatten(step_mode=self.step_mode)

        i = 1

        for in_units, out_units in zip([math.prod(input_shape), *units[:-1]], units):
            layers[f'fc{i}'] = QuantizedLinear(in_units,
                                               out_units,
                                               quant_params=quant_params,
                                               step_mode=self.step_mode)
            layers[f'neuron{i}'] = self.create_neuron(quant_params=quant_params)
            i += 1

        layers[f'fc{i}'] = QuantizedLinear(units[-1] if len(units) > 1 else math.prod(input_shape),
                                           output_shape[0],
                                           quant_params=quant_params,
                                           step_mode=self.step_mode)

        self.layers = nn.ModuleDict(layers)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward calls each of the MLP :attr:`layers` sequentially.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input
        for layer in self.layers:
            x = self.layers[layer](x)

        return x
