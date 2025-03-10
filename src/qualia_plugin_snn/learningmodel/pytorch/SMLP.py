"""Contains the template for a spiking multi-layer perceptron."""

from __future__ import annotations

import math
import sys
from collections import OrderedDict

from qualia_core.typing import TYPE_CHECKING
from torch import nn

from .SNN import SNN

if TYPE_CHECKING:
    import torch
    from qualia_core.typing import RecursiveConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class SMLP(SNN):
    """Spiking multi-layer perceptron template.

    Similar to :class:`qualia_core.learningmodel.pytorch.MLP.MLP` but with spiking neuron activation layers (e.g., IF) instead of
    :class:`torch.nn.ReLU`.

    Last :class:`torch.nn.Linear` layer matching number of output classes is implicitely added.

    Example TOML configuration for a 3-layer spiking MLP over 4 timesteps with soft-reset multi-step IF based on the SMLP template:

    .. code-block:: toml

        [[model]]
        kind = "SMLP"
        name = "smlp_128-128-10"
        params.units        = [128, 128]
        params.timesteps    = 4
        params.neuron.kind                = 'IFNode'
        params.neuron.params.v_reset      = false # Soft reset
        params.neuron.params.v_threshold  = 1.0
        params.neuron.params.detach_reset = true
        params.neuron.params.step_mode    = 'm' # Multi-step mode, make sure to use SpikingJellyMultiStep learningframework
        params.neuron.params.backend      = 'torch'
    """

    def __init__(self,
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 units: list[int],
                 timesteps: int,
                 neuron: RecursiveConfigDict | None = None) -> None:
        """Construct :class:`SMLP`.

        :param input_shape: Input shape
        :param output_shape: Output shape
        :param units: List of :class:`torch.nn.Linear` layer ``out_features`` to add in the network
        :param neuron: Spiking neuron configuration, see :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.__init__`
        :param timesteps: Number of timesteps
        """
        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         timesteps=timesteps,
                         neuron=neuron)

        from spikingjelly.activation_based.layer import Flatten, Linear  # type: ignore[import-untyped]

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        layers['flatten1'] = Flatten(step_mode=self.step_mode)

        i = 1

        for in_units, out_units in zip([math.prod(input_shape), *units[:-1]], units):
            layers[f'fc{i}'] = Linear(in_units, out_units, step_mode=self.step_mode)
            layers[f'neuron{i}'] = self.create_neuron()
            i += 1

        layers[f'fc{i}'] = Linear(units[-1] if len(units) > 1 else math.prod(input_shape),
                                  output_shape[0],
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
