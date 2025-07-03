"""Contains the template for a convolutional spiking neural network."""

from __future__ import annotations

import logging
import math
import sys
from collections import OrderedDict

import numpy as np
import torch
from qualia_core.learningmodel.pytorch.layers import layers1d, layers2d
from qualia_core.typing import TYPE_CHECKING
from torch import nn

from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly import layers1d as sjlayers1d
from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly import layers2d as sjlayers2d

from .SNN import SNN

if TYPE_CHECKING:
    from types import ModuleType

    from qualia_core.typing import RecursiveConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class SCNN(SNN):
    """Convolutional spiking neural network template.

    Similar to :class:`qualia_core.learningmodel.pytorch.CNN.CNN` but with spiking neuron activation layers (e.g., IF) instead of
    :class:`torch.nn.ReLU`.


    Example TOML configuration for a 2D SVGG16 over 4 timesteps with soft-reset multi-step IF based on the SCNN template:

    .. code-block:: toml

        [[model]]
        name = "VGG16"
        params.filters      = [ 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        params.kernel_sizes = [  3,  3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3]
        params.paddings     = [  1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
        params.strides      = [  1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
        params.pool_sizes   = [  0,  2,   0,   2,   0,   0,   2,   0,   0,   2,   0,   0,   2]
        params.dropouts     = [  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0]
        params.separables   = [false, false, false, false, false, false, false, false, false, false, false, false, false]
        params.fc_units     = [4096, 4096]
        params.batch_norm   = true
        params.timesteps    = 4
        params.dims         = 2
        params.neuron.kind                = 'IFNode'
        params.neuron.params.v_reset      = false # Soft reset
        params.neuron.params.v_threshold  = 1.0
        params.neuron.params.detach_reset = true
        params.neuron.params.step_mode    = 'm' # Multi-step mode, make sure to use SpikingJellyMultiStep learningframework
        params.neuron.params.backend      = 'cupy'
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
                 separables: list[bool] | None = None,
                 batch_norm: bool = False,  # noqa: FBT001, FBT002
                 prepool: int | list[int] = 1,
                 postpool: int | list[int] = 1,

                 neuron: RecursiveConfigDict | None = None,
                 timesteps: int = 4,

                 gsp: bool = False,  # noqa: FBT001, FBT002

                 dims: int = 1) -> None:
        """Construct :class:`SCNN`.

        :param input_shape: Input shape
        :param output_shape: Output shape
        :param filters: List of ``out_channels`` for each Conv layer, also defines the number of Conv layers
        :param kernel_sizes: List of ``kernel_size`` for each Conv layer, must of the same size as ``filters``
        :param paddings: List of ``padding`` for each Conv layer, must of the same size as ``filters``
        :param strides: List of ``stride`` for each Conv layer, must of the same size as ``filters``
        :param dropouts: List of Dropout layer ``p`` to apply after each Conv or Linear layer, must be of the same size as
                         ``filters`` + ``fc_units``, no layer added if element is 0
        :param pool_sizes: List of MaxPool layer ``kernel_size`` to apply after each Conv layer, must be of the same size as
                           ``filters``, no layer added if element is 0
        :param fc_units: List of :class:`torch.nn.Linear` layer ``out_features`` to add at the end of the network, no layer added
                         if empty
        :param separables: Whether a given Conv layer is implemented as a depthwise-pointwise pair (true)
                         or as a standard Conv (false), must be of the same size as ``filters``
        :param batch_norm: If ``True``, add a BatchNorm layer after each Conv layer, otherwise no layer added
        :param prepool: AvgPool layer ``kernel_size`` to add at the beginning of the network, no layer added if 0
        :param postpool: AvgPool layer ``kernel_size`` to add after all Conv layers, no layer added if 0
        :param neuron: Spiking neuron configuration, see :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.__init__`
        :param timesteps: Number of timesteps
        :param gsp: If ``True``, a single GlobalSumPool layer is added instead of Linear layers
        :param dims: Either 1 or 2 for 1D or 2D convolutional network.
        """
        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         timesteps=timesteps,
                         neuron=neuron)

        from spikingjelly.activation_based.layer import Dropout, Flatten, Linear  # type: ignore[import-untyped]

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

        if separables is None:
            separables = [False] * len(filters)

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        if (math.prod(prepool) if isinstance(prepool, list) else prepool) > 1:
            layers['prepool'] = sjlayers_t.AvgPool(tuple(prepool) if isinstance(prepool, list) else prepool,
                                                   step_mode=self.step_mode)

        i = 1
        for (in_filters,
             out_filters,
             kernel,
             pool_size,
             padding,
             stride,
             dropout,
             separable) in zip([input_shape[-1], *filters],
                               filters,
                               kernel_sizes,
                               pool_sizes,
                               paddings,
                               strides,
                               dropout_list,
                               separables):
            if separable:
                layers[f'conv{i}_dw'] = sjlayers_t.Conv(in_channels=in_filters,
                                                        out_channels=in_filters,
                                                        kernel_size=kernel,
                                                        padding=padding,
                                                        stride=stride,
                                                        groups=in_filters,
                                                        bias=not batch_norm,
                                                        step_mode=self.step_mode)

                if batch_norm:
                    layers[f'bn{i}_dw'] = sjlayers_t.BatchNorm(out_filters, step_mode=self.step_mode)

                layers[f'neuron{i}_dw'] = self.create_neuron()

                layers[f'conv{i}_pw'] = sjlayers_t.Conv(in_channels=in_filters,
                                                          out_channels=out_filters,
                                                          kernel_size=1,
                                                          padding=0,
                                                          stride=1,
                                                          bias=not batch_norm,
                                                          step_mode=self.step_mode)

                if batch_norm:
                    layers[f'bn{i}_pw'] = sjlayers_t.BatchNorm(out_filters, step_mode=self.step_mode)

                layers[f'neuron{i}_pw'] = self.create_neuron()
            else:
                layers[f'conv{i}'] = sjlayers_t.Conv(in_channels=in_filters,
                                                          out_channels=out_filters,
                                                          kernel_size=kernel,
                                                          padding=padding,
                                                          stride=stride,
                                                          bias=not batch_norm,
                                                          step_mode=self.step_mode)

                if batch_norm:
                    layers[f'bn{i}'] = sjlayers_t.BatchNorm(out_filters, step_mode=self.step_mode)

                layers[f'neuron{i}'] = self.create_neuron()

            if dropout:
                layers[f'dropout{i}'] = Dropout(dropout, step_mode=self.step_mode)
            if pool_size:
                layers[f'maxpool{i}'] = sjlayers_t.MaxPool(pool_size, step_mode=self.step_mode)

            i += 1

        if (math.prod(postpool) if isinstance(postpool, list) else postpool) > 1:
            layers['postpool'] = sjlayers_t.AvgPool(tuple(postpool) if isinstance(postpool, list) else postpool,
                                                    step_mode=self.step_mode)

        if gsp:
            layers[f'conv{i}'] = sjlayers_t.Conv(in_channels=filters[-1],
                                                 out_channels=output_shape[0],
                                                 kernel_size=1,
                                                 padding=0,
                                                 stride=1,
                                                 bias=True,
                                                 step_mode=self.step_mode)
            layers[f'neuron{i}'] = self.create_neuron()
            layers['gsp'] = layers_t.GlobalSumPool()
        else:
            layers['flatten'] = Flatten(step_mode=self.step_mode)

            in_features = np.array(input_shape[:-1]) // np.array(prepool)
            for _, kernel, pool, padding, stride in zip(filters, kernel_sizes, pool_sizes, paddings, strides):
                in_features += np.array(padding) * 2
                in_features -= (np.array(kernel) - 1)
                in_features = np.ceil(in_features / stride).astype(int)
                if pool:
                    in_features = in_features // pool
            in_features = in_features // np.array(postpool)
            in_features = in_features.prod()
            in_features *= filters[-1]

            j = 1
            for in_units, out_units, dropout in zip((in_features, *fc_units), fc_units, dropout_list[len(filters):]):
                layers[f'fc{j}'] = Linear(in_units, out_units, step_mode=self.step_mode)
                layers[f'neuron{i}'] = self.create_neuron()
                if dropout:
                    layers[f'dropout{i}'] = Dropout(dropout, step_mode=self.step_mode)
                i += 1
                j += 1

            layers[f'fc{j}'] = Linear(fc_units[-1] if len(fc_units) > 0 else in_features,
                                           output_shape[0],
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
