"""Contains the template for a quantized convolutional spiking neural network."""

from __future__ import annotations

import logging
import math
import sys
from collections import OrderedDict

import numpy as np
from qualia_core.learningmodel.pytorch import layers1d, layers2d
from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedIdentity
from qualia_core.typing import TYPE_CHECKING
from torch import nn

from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly import layers1d as sjlayers1d
from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly import layers2d as sjlayers2d

from .SNN import SNN

if TYPE_CHECKING:
    from types import ModuleType  # noqa: TCH003

    import torch
    from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig  # noqa: TCH002
    from qualia_core.typing import RecursiveConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class QuantizedSCNN(SNN):
    """Quantized convolutional spiking neural network template.

    Should have topology identical to :class:`qualia_plugin_snn.learningmodel.pytorch.SCNN.SCNN` but with layers replaced with
    their quantized equivalent.
    """

    layers: nn.ModuleDict #: List of sequential layers of the SCNN model


    def __init__(self,  # noqa: PLR0913, PLR0915, PLR0912, C901
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],

                 filters: list[int],
                 kernel_sizes: list[int],
                 paddings: list[int],
                 strides: list[int],
                 dropouts: float | list[float],
                 pool_sizes: list[int],
                 fc_units: list[int],

                 quant_params: QuantizationConfig,

                 batch_norm: bool = False,  # noqa: FBT001, FBT002
                 prepool: int | list[int] = 1,
                 postpool: int | list[int] = 1,

                 neuron: RecursiveConfigDict | None = None,
                 timesteps: int = 4,

                 gsp: bool = False,  # noqa: FBT001, FBT002

                 dims: int=1) -> None:
        """Construct :class:`QuantizedSCNN`.

        :param input_shape: Input shape
        :param output_shape: Output shape
        :param filters: List of ``out_channels`` for each QuantizedConv layer, also defines the number of QuantizedConv layers
        :param kernel_sizes: List of ``kernel_size`` for each QuantizedConv layer, must of the same size as ``filters``
        :param paddings: List of ``padding`` for each QuantizedConv layer, must of the same size as ``filters``
        :param strides: List of ``stride`` for each QuantizedConv layer, must of the same size as ``filters``
        :param dropouts: List of Dropout layer ``p`` to apply after each QuantizedConv or QuantizedLinear layer, must be of the
                         same size as ``filters`` + ``fc_units``, no layer added if element is 0
        :param pool_sizes: List of QuantizedMaxPool layer ``kernel_size`` to apply after each QuantizedConv layer, must be of the
                           same size as ``filters``, no layer added if element is 0
        :param fc_units: List of QuantizedLinear layer ``out_features`` to add at the end of the network, no layer added
                         if empty
        :param batch_norm: If ``True``, add a QuantizedBatchNorm layer after each QuantizedConv layer, otherwise no layer added
        :param prepool: QuantizedAvgPool layer ``kernel_size`` to add at the beginning of the network, no layer added if 0
        :param postpool: QuantizedAvgPool layer ``kernel_size`` to add after all QuantizedConv layers, no layer added if 0
        :param neuron: Spiking neuron configuration, see :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.__init__`,
                       ``'Quantized'`` is automatically prepended to ``neuron.kind``
        :param timesteps: Number of timesteps
        :param gsp: If ``True``, a single QuantizedGlobalSumPool layer is added instead of QuantizedLinear layers
        :param dims: Either 1 or 2 for 1D or 2D convolutional network.
        :param quant_params: Quantization configuration dict, see :class:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer`
        """
        # Prepend Quantized to the neuron kind to instantiate quantized spiking neurons
        if neuron is not None and 'kind' in neuron and isinstance(neuron['kind'], str):
            neuron['kind'] = 'Quantized' + neuron['kind']

        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         timesteps=timesteps,
                         neuron=neuron)

        from spikingjelly.activation_based.layer import Dropout, Flatten  # type: ignore[import-untyped]

        from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly.quantized_layers import QuantizedLinear

        layers_t: ModuleType
        sjlayers_t: ModuleType

        if dims == 1:
            layers_t = layers1d
            sjlayers_t = sjlayers1d
        elif dims == 2:  # noqa: PLR2004
            layers_t = layers2d
            sjlayers_t = sjlayers2d
        else:
            logger.error('Only dims=1 or dims=2 supported, got: %s', dims)
            raise ValueError

        # Backward compatibility for config not defining dropout as a list
        dropout_list = [dropouts] * (len(filters) + len(fc_units)) if not isinstance(dropouts, list) else dropouts

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        layers['identity1'] = QuantizedIdentity(quant_params=quant_params)

        if (math.prod(prepool) if isinstance(prepool, list) else prepool) > 1:
            layers['prepool'] = sjlayers_t.QuantizedAvgPool(tuple(prepool) if isinstance(prepool, list) else prepool,
                                                   step_mode=self.step_mode,
                                                   quant_params=quant_params)

        layers['conv1'] = sjlayers_t.QuantizedConv(in_channels=input_shape[-1],
                                                   out_channels=filters[0],
                                                   kernel_size=kernel_sizes[0],
                                                   padding=paddings[0],
                                                   stride=strides[0],
                                                   bias=not batch_norm,
                                                   quant_params=quant_params,
                                                   step_mode=self.step_mode)

        if batch_norm:
            layers['bn1'] = sjlayers_t.QuantizedBatchNorm(filters[0], quant_params=quant_params, step_mode=self.step_mode)

        layers['neuron1'] = self.create_neuron(quant_params=quant_params)

        if dropout_list[0]:
            layers['dropout1'] = Dropout(dropout_list[0], step_mode=self.step_mode)
        if pool_sizes[0]:
            layers['maxpool1'] = sjlayers_t.QuantizedMaxPool(pool_sizes[0], quant_params=quant_params, step_mode=self.step_mode)

        i = 2
        for in_filters, out_filters, kernel, pool_size, padding, stride, dropout in zip(filters,
                                                                               filters[1:],
                                                                               kernel_sizes[1:],
                                                                               pool_sizes[1:],
                                                                               paddings[1:],
                                                                               strides[1:],
                                                                               dropout_list[1:]):
            layers[f'conv{i}'] = sjlayers_t.QuantizedConv(in_channels=in_filters,
                                                          out_channels=out_filters,
                                                          kernel_size=kernel,
                                                          padding=padding,
                                                          stride=stride,
                                                          bias=not batch_norm,
                                                          quant_params=quant_params,
                                                          step_mode=self.step_mode)

            if batch_norm:
                layers[f'bn{i}'] = sjlayers_t.QuantizedBatchNorm(out_filters,
                                                                 quant_params=quant_params,
                                                                 step_mode=self.step_mode)

            layers[f'neuron{i}'] = self.create_neuron(quant_params=quant_params)

            if dropout:
                layers[f'dropout{i}'] = Dropout(dropout, step_mode=self.step_mode)
            if pool_size:
                layers[f'maxpool{i}'] = sjlayers_t.QuantizedMaxPool(pool_size, quant_params=quant_params, step_mode=self.step_mode)

            i += 1

        if (math.prod(postpool) if  isinstance(postpool, list) else postpool) > 1:
            layers['postpool'] = sjlayers_t.QuantizedAvgPool(tuple(postpool) if isinstance(postpool, list) else postpool,
                                                    step_mode=self.step_mode,
                                                    quant_params=quant_params)

        if gsp:
            layers[f'conv{i}'] = sjlayers_t.QuantizedConv(in_channels=filters[-1],
                                                          out_channels=output_shape[0],
                                                          kernel_size=1,
                                                          padding=0,
                                                          stride=1,
                                                          bias=True,
                                                          quant_params=quant_params,
                                                          step_mode=self.step_mode)
            layers['gsp'] = layers_t.QuantizedGlobalSumPool(quant_params=quant_params)
        else:
            layers['flatten'] = Flatten(step_mode=self.step_mode)

            in_features = np.array(input_shape[:-1]) // np.array(prepool)
            for _, kernel, pool, padding, stride in zip(filters, kernel_sizes, pool_sizes, paddings, strides):
                in_features += np.array(padding) * 2
                in_features -= (kernel - 1)
                in_features = np.ceil(in_features / stride).astype(int)
                if pool:
                    in_features = in_features // pool
            in_features = in_features // np.array(postpool)
            in_features = in_features.prod()
            in_features *= filters[-1]

            j = 1
            for in_units, out_units, dropout in zip((in_features, *fc_units), fc_units, dropout_list[len(filters):]):
                layers[f'fc{j}'] = QuantizedLinear(in_units, out_units, quant_params=quant_params, step_mode=self.step_mode)
                layers[f'neuron{i}'] = self.create_neuron(quant_params=quant_params)
                if dropout:
                    layers[f'dropout{i}'] = Dropout(dropout, step_mode=self.step_mode)
                i += 1
                j += 1

            layers[f'fc{j}'] = QuantizedLinear(fc_units[-1] if len(fc_units) > 0 else in_features,
                                               output_shape[0],
                                               quant_params=quant_params,
                                               step_mode=self.step_mode)
        self.layers = nn.ModuleDict(layers)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward calls each of the SCNN :attr:`layers` sequentially.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input
        for layer in self.layers:
            x = self.layers[layer](x)

        return x
