"""Contains the template for a quantized residual spiking neural network."""

from __future__ import annotations

import logging
import sys
from typing import Protocol, cast

import numpy as np
import torch
from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedIdentity
from qualia_core.typing import TYPE_CHECKING
from torch import nn

from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly import layers1d as sjlayers1d
from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly import layers2d as sjlayers2d
from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly.QuantizedAdd import QuantizedAdd

from .SNN import SNN

if TYPE_CHECKING:
    from types import ModuleType

    from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig
    from qualia_core.typing import RecursiveConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class CreateNeuron(Protocol):
    """Signature for create_neuron from :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.create_neuron`.

    Used to pass neuron builder to BasicBlock.
    """

    def __call__(self, quant_params: QuantizationConfig) -> nn.Module:
        """Instanciate a spiking neuron.

        :param quant_params: Optional quantization configuration dict in case of quantized network, see
                             :class:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer`
        :return: A spiking neuron instance
        """
        ...

class QuantizedBasicBlockBuilder(Protocol):
    """Signature for basicblockbuilder.

    Used to bind hyperparameters constant across all the ResNet blocks.
    """

    def __call__(self,
                 in_planes: int,
                 planes: int,
                 kernel_size: int,
                 stride: int,
                 padding: int) -> QuantizedBasicBlock:
        """Build a :class:`QuantizedBasicBlock`.

        :param in_planes: Number of input channels
        :param planes: Number of filters (i.e., output channels) in the main branch Conv layers
        :param kernel_size: ``kernel_size`` for the main branch Conv layers
        :param stride: ``kernel_size`` for the MaxPool layers, no MaxPool layer added if `1`
        :param padding: Padding for the main branch Conv layers
        :return: A :class:`QuantizedBasicBlock`
        """
        ...

class QuantizedBasicBlock(nn.Module):
    r"""A single quantized ResNetv1 block.

    Should have topology identical to :class:`qualia_plugin_snn.learningmodel.pytorch.SResNet.BasicBlock` but with layers replaced
    with their quantized equivalent.

    Structure is:

    .. code-block::

                         |
                     /        \
                 |                 |
           QuantizedConv           |
                 |                 |
        QuantizedBatchNorm         |
                 |                 |
         QuantizedMaxPool   QuantizedConv
                 |                 |
            QuantizedIF  QuantizedBatchNorm
                 |                 |
           QuantizedConv  QuantizedMaxPool
                 |                 |
        QuantizedBatchNorm   QuantizedIF
                 |                 |
            QuantizedIF            |
                 |                 |
                     \       /
                         |
                    QuantizedAdd

    Main (left) branch QuantizedConv use ``kernel_size=kernel_size``, while residual (right) branch QuantizedConv
    use ``kernel_size=1``.

    QuantizedBatchNorm layers will be absent if ``batch_norm == False``

    QuantizedMaxPool layer will be absent if ``stride == 1``.

    Residual (right) branch QuantizedConv layer will be asbent if ``in_planes==planes``, except if
    ``force_projection_with_stride==True`` and ``stride != 1``.
    """

    expansion: int = 1 #: Unused

    def __init__(self,  # noqa: PLR0913
                 sjlayers_t: ModuleType,
                 in_planes: int,
                 planes: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,  # noqa: FBT001
                 bn_momentum: float,
                 force_projection_with_stride: bool,  # noqa: FBT001
                 create_neuron: CreateNeuron,
                 step_mode: str,
                 quant_params: QuantizationConfig) -> None:
        """Construct :class:`QuantizedBasicBlock`.

        :param sjlayers_t: Module containing the aliased quantized layers to use (1D or 2D)
        :param in_planes: Number of input channels
        :param planes: Number of filters (i.e., output channels) in the main branch QuantizedConv layers
        :param kernel_size: ``kernel_size`` for the main branch QuantizedConv layers
        :param stride: ``kernel_size`` for the QuantizedMaxPool layers, no QuantizedMaxPool layer added if `1`
        :param padding: Padding for the main branch QuantizedConv layers
        :param batch_norm: If ``True``, add BatchNorm layer after each QuantizedConv layer
        :param bn_momentum: QuantizedBatchNorm layer ``momentum``
        :param force_projection_with_stride: If ``True``, residual QuantizedConv layer is kept when ``stride != 1`` even if
                                             ``in_planes == planes``
        :param create_neuron: :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.create_neuron` method to instantiate a spiking
                              neuron
        :param step_mode: SpikingJelly ``step_mode`` from :attr:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.step_mode`
        :param quant_params: Quantization configuration dict, see :class:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer`
        """
        super().__init__()
        self.expansion = 1
        self.batch_norm = batch_norm
        self.force_projection_with_stride = force_projection_with_stride
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = sjlayers_t.QuantizedConv(in_planes,
                                              planes,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              padding=padding,
                                              bias=not batch_norm,
                                              step_mode=step_mode,
                                              quant_params=quant_params,
                                              activation=None)
        if batch_norm:
            self.bn1 = sjlayers_t.QuantizedBatchNorm(planes, momentum=bn_momentum, step_mode=step_mode, quant_params=quant_params)

        if self.stride != 1:
            self.pool1 = sjlayers_t.QuantizedMaxPool(stride, step_mode=step_mode, quant_params=quant_params, activation=None)

        self.neuron1 = create_neuron(quant_params=quant_params)

        self.conv2 = sjlayers_t.QuantizedConv(planes,
                                              planes,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              padding=padding,
                                              bias=not batch_norm,
                                              step_mode=step_mode,
                                              quant_params=quant_params)

        if batch_norm:
            self.bn2 = sjlayers_t.QuantizedBatchNorm(planes, momentum=bn_momentum, step_mode=step_mode, quant_params=quant_params)

        self.neuron2 = create_neuron(quant_params=quant_params)

        if self.stride != 1:
            self.smax = sjlayers_t.QuantizedMaxPool(stride, step_mode=step_mode, quant_params=quant_params)
        if self.in_planes != self.expansion*self.planes or (force_projection_with_stride and self.stride != 1):
            self.sconv = sjlayers_t.QuantizedConv(in_planes,
                                                  self.expansion*planes,
                                                  kernel_size=1,
                                                  stride=1,
                                                  bias=not batch_norm,
                                                  step_mode=step_mode,
                                                  quant_params=quant_params)
            self.neuronr = create_neuron(quant_params=quant_params)

            if batch_norm:
                self.sbn = sjlayers_t.QuantizedBatchNorm(self.expansion*planes,
                                                         momentum=bn_momentum,
                                                         step_mode=step_mode,
                                                         quant_params=quant_params)
        self.add = QuantizedAdd(quant_params=quant_params)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward of quantized ResNet block.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input

        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)
        if self.stride != 1:
            out = self.pool1(out)
        out = self.neuron1(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.neuron2(out)

        # shortcut
        tmp = x

        if self.in_planes != self.expansion*self.planes or (self.force_projection_with_stride and self.stride != 1):
            tmp = self.sconv(tmp)

            if self.batch_norm:
                tmp = self.sbn(tmp)

        if self.stride != 1:
            tmp = self.smax(tmp)

        if self.in_planes != self.expansion*self.planes or (self.force_projection_with_stride and self.stride != 1):
            tmp = self.neuronr(tmp)

        return self.add(out, tmp)

class QuantizedSResNet(SNN):
    """Quantized residual spiking neural network template.

    Should have topology identical to :class:`qualia_plugin_snn.learningmodel.pytorch.SResNet.SResNet` but with layers replaced
    with their quantized equivalent.
    """

    def __init__(self,  # noqa: PLR0913, C901
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 filters: list[int],
                 kernel_sizes: list[int],

                 num_blocks: list[int],
                 strides: list[int],
                 paddings: list[int],

                 quant_params: QuantizationConfig,

                 prepool: int = 1,
                 postpool: str = 'max',
                 batch_norm: bool = False,  # noqa: FBT001, FBT002
                 bn_momentum: float = 0.1,
                 force_projection_with_stride: bool = True,  # noqa: FBT001, FBT002

                 neuron: RecursiveConfigDict | None = None,
                 timesteps: int = 2,

                 dims: int = 1,
                 basicblockbuilder: QuantizedBasicBlockBuilder | None = None) -> None:
        """Construct :class:`QuantizedSResNet`.

        Structure is:

        .. code-block::

              QuantizedInput
                     |
             QuantizedAvgPool
                     |
               QuantizedConv
                     |
            QuantizedBatchNorm
                     |
                QuantizedIF
                     |
            QuantizedBasicBlock
                     |
                     …
                     |
            QuantizedBasicBlock
                     |
            QuantizedGlobalPool
                     |
                  Flatten
                     |
              QuantizedLinear

        :param input_shape: Input shape
        :param output_shape: Output shape
        :param filters: List of ``out_channels`` for QuantizedConv layers inside each :class:`QuantizedBasicBlock` group, must be
                        of the same size as ``num_blocks``, first element is for the first QuantizedConv layer at the beginning
                        of the network
        :param kernel_sizes: List of ``kernel_size`` for QuantizedConv layers inside each :class:`QuantizedBasicBlock` group, must
                             of the same size as ``num_blocks``, first element is for the first QuantizedConv layer at the
                             beginning of the network
        :param num_blocks: List of number of :class:`QuantizedBasicBlock` in each group, also defines the number
                           of :class:`QuantizedBasicBlock` groups inside the network
        :param strides: List of ``kernel_size`` for QuantizedMaxPool layers inside each :class:`QuantizedBasicBlock` group, must
                        of the same size as ``num_blocks``, ``stride`` is applied only to the first :class:`QuantizedBasicBlock`
                        of the group, next :class:`QuantizedBasicBlock` in the group use a ``stride`` of ``1``, first element is
                        the stride of the first QuantizedConv layer at the beginning of the network
        :param paddings: List of ``padding`` for QuantizedConv layer inside each :class:`QuantizedBasicBlock` group, must of the
                         same size as ``num_blocks``, first element is for the first QuantizedConv layer at the beginning
                         of the network
        :param prepool: QuantizedAvgPool layer ``kernel_size`` to add at the beginning of the network, no layer added if 0
        :param postpool: Quantized global pooling layer type after all :class:`QuantizedBasicBlock`, either `max` for
                         QuantizedMaxPool or `avg` for QuantizedAvgPool
        :param batch_norm: If ``True``, add a QuantizedBatchNorm layer after each QuantizedConv layer, otherwise no layer added
        :param bn_momentum: QuantizedBatchNorm ``momentum``
        :param force_projection_with_stride: If ``True``, residual QuantizedConv layer is kept when ``stride != 1`` even if
                                             ``in_planes == planes`` inside a :class:`QuantizedBasicBlock`
        :param neuron: Spiking neuron configuration, see :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.__init__`
        :param timesteps: Number of timesteps
        :param dims: Either 1 or 2 for 1D or 2D convolutional network.
        :param basicblockbuilder: Optional function with :meth:`QuantizedBasicBlockBuilder.__call__` signature to build a basic
                                  block after binding constants common across all basic blocks
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


        sjlayers_t: ModuleType
        if dims == 1:
            sjlayers_t = sjlayers1d
        elif dims == 2:  # noqa: PLR2004
            sjlayers_t = sjlayers2d
        else:
            logger.error('Only dims=1 or dims=2 supported, got: %s', dims)
            raise ValueError

        if basicblockbuilder is None:
            def builder(in_planes: int,
                        planes: int,
                        kernel_size: int,
                        stride: int,
                        padding: int) -> QuantizedBasicBlock:
                return QuantizedBasicBlock(
                        sjlayers_t=sjlayers_t,
                        in_planes=in_planes,
                        planes=planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        batch_norm=batch_norm,
                        bn_momentum=bn_momentum,
                        force_projection_with_stride=force_projection_with_stride,
                        create_neuron=self.create_neuron,
                        step_mode=self.step_mode,
                        quant_params=quant_params)
            basicblockbuilder = builder

        self.in_planes: int = filters[0]
        self.batch_norm = batch_norm
        self.num_blocks = num_blocks

        self.identity1 = QuantizedIdentity(quant_params=quant_params)

        if prepool > 1:
            self.prepool = sjlayers_t.QuantizedAvgPool(prepool, step_mode=self.step_mode, quant_params=quant_params)

        self.conv1 = sjlayers_t.QuantizedConv(input_shape[-1],
                                              filters[0],
                                              kernel_size=kernel_sizes[0],
                                              stride=strides[0],
                                              padding=paddings[0],
                                              bias=not batch_norm,
                                              step_mode=self.step_mode,
                                              quant_params=quant_params)
        if self.batch_norm:
            self.bn1 = sjlayers_t.QuantizedBatchNorm(self.in_planes,
                                                     momentum=bn_momentum,
                                                     step_mode=self.step_mode,
                                                     quant_params=quant_params)

        self.neuron1: nn.Module = self.create_neuron(quant_params=quant_params)

        blocks: list[nn.Sequential] = []
        for planes, kernel_size, stride, padding, num_block in zip(filters[1:],
                                                                   kernel_sizes[1:],
                                                                   strides[1:],
                                                                   paddings[1:],
                                                                   num_blocks):
            blocks.append(self._make_layer(basicblockbuilder, num_block, planes, kernel_size, stride, padding))
        self.layers = nn.ModuleList(blocks)

        # GlobalMaxPool kernel_size computation
        self._fm_dims = np.array(input_shape[:-1]) // np.array(prepool)
        for _, kernel, stride, padding in zip(filters, kernel_sizes, strides, paddings):
            self._fm_dims += np.array(padding) * 2
            self._fm_dims -= (kernel - 1)
            self._fm_dims = np.floor(self._fm_dims / stride).astype(int)


        if postpool == 'avg':
            self.postpool = sjlayers_t.QuantizedAvgPool(tuple(self._fm_dims), step_mode=self.step_mode, quant_params=quant_params)
        elif postpool == 'max':
            self.postpool = sjlayers_t.QuantizedMaxPool(tuple(self._fm_dims), step_mode=self.step_mode, quant_params=quant_params)

        self.flatten = Flatten(step_mode=self.step_mode)
        # Replace Linear with the quantized variant
        self.linear = QuantizedLinear(self.in_planes*QuantizedBasicBlock.expansion,
                                      output_shape[0],
                                      step_mode=self.step_mode,
                                      quant_params=quant_params)

    def _make_layer(self,  # noqa: PLR0913
                    basicblockbuilder: QuantizedBasicBlockBuilder,
                    num_blocks: int,
                    planes: int,
                    kernel_size: int,
                    stride: int,
                    padding: int) -> nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers: list[QuantizedBasicBlock] = []
        for s in strides:
            block: QuantizedBasicBlock = basicblockbuilder(in_planes=self.in_planes,
                                      planes=planes,
                                      kernel_size=kernel_size,
                                      stride=s,
                                      padding=padding)
            layers.append(block)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        out = self.identity1(x)

        if hasattr(self, 'prepool'):
            out = self.prepool(out)

        out = self.conv1(out)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.neuron1(out)

        for i in range(len(self.layers)):
            out = self.layers[i](out)

        if hasattr(self, 'postpool'):
            out = self.postpool(out)

        out = self.flatten(out)
        return cast('torch.Tensor', self.linear(out))
