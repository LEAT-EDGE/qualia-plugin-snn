"""Contains the template for a residual spiking neural network."""

from __future__ import annotations

import logging
import sys
from typing import Callable, Protocol, cast

import numpy as np
import torch
from qualia_core.learningmodel.pytorch import layers1d, layers2d
from qualia_core.typing import TYPE_CHECKING
from torch import nn

from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly import layers1d as sjlayers1d
from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly import layers2d as sjlayers2d
from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly.Add import Add

from .SNN import SNN

if TYPE_CHECKING:
    from types import ModuleType

    from qualia_core.typing import RecursiveConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class BasicBlockBuilder(Protocol):
    """Signature for basicblockbuilder.

    Used to bind hyperparameters constant across all the ResNet blocks.
    """

    def __call__(self,  # noqa: PLR0913
                 in_planes: int,
                 planes: int,
                 kernel_size: int,
                 pool_size: int,
                 stride: int,
                 padding: int) -> BasicBlock:
        """Build a :class:`BasicBlock`.

        :param in_planes: Number of input channels
        :param planes: Number of filters (i.e., output channels) in the main branch Conv layers
        :param kernel_size: ``kernel_size`` for the main branch Conv layers
        :param pool_size: ``kernel_size`` for the MaxPool layers, no MaxPool layer added if `1`
        :param stride: ``kernel_size`` for the MaxPool layers, no MaxPool layer added if `1`
        :param padding: Padding for the main branch Conv layers
        :return: A :class:`BasicBlock`
        """
        ...


class BasicBlock(nn.Module):
    r"""A single ResNetv1 block.

    Structure is:

    .. code-block::

                |
              /   \
            |       |
           Conv     |
            |       |
        BatchNorm   |
            |       |
         MaxPool   Conv
            |       |
            IF  BatchNorm
            |       |
           Conv  MaxPool
            |       |
        BatchNorm   IF
            |       |
            IF      |
            |       |
              \   /
                |
               Add

    Main (left) branch Conv use ``kernel_size=kernel_size``, while residual (right) branch Conv use ``kernel_size=1``.

    BatchNorm layers will be absent if ``batch_norm == False``

    MaxPool layer will be absent if ``pool_size == 1``.

    Residual (right) branch Conv layer will be asbent if ``in_planes==planes``, except if ``force_projection_with_pooling==True``
    and ``pool_size != 1``.
    """

    expansion: int = 1  #: Unused

    def __init__(self,  # noqa: PLR0913
                 sjlayers_t: ModuleType,
                 in_planes: int,
                 planes: int,
                 kernel_size: int,
                 pool_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,  # noqa: FBT001
                 bn_momentum: float,
                 force_projection_with_pooling: bool,  # noqa: FBT001
                 create_neuron: Callable[[], nn.Module],
                 step_mode: str) -> None:
        """Construct :class:`BasicBlock`.

        :param sjlayers_t: Module containing the aliased layers to use (1D or 2D)
        :param in_planes: Number of input channels
        :param planes: Number of filters (i.e., output channels) in the main branch Conv layers
        :param kernel_size: ``kernel_size`` for the main branch Conv layers
        :param pool_size: ``kernel_size`` for the MaxPool layers, no MaxPool layer added if `1`
        :param stride: ``stride`` for the first Conv layer
        :param padding: Padding for the main branch Conv layers
        :param batch_norm: If ``True``, add BatchNorm layer after each Conv layer
        :param bn_momentum: BatchNorm layer ``momentum``
        :param force_projection_with_pooling: If ``True``, residual Conv layer is kept when ``stride != 1`` even if
                                             ``in_planes == planes``
        :param create_neuron: :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.create_neuron` method to instantiate a spiking
                              neuron
        :param step_mode: SpikingJelly ``step_mode`` from :attr:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.step_mode`
        """
        super().__init__()
        self.expansion = 1
        self.batch_norm = batch_norm
        self.force_projection_with_pooling = force_projection_with_pooling
        self.in_planes = in_planes
        self.planes = planes
        self.pool_size = pool_size
        self.stride = stride

        self.conv1 = sjlayers_t.Conv(in_planes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=not batch_norm,
                                     step_mode=step_mode)
        if batch_norm:
            self.bn1 = sjlayers_t.BatchNorm(planes, momentum=bn_momentum, step_mode=step_mode)

        if self.pool_size != 1:
            self.pool1 = sjlayers_t.MaxPool(pool_size, step_mode=step_mode)

        # QUANTIZED NEURON
        self.neuron1 = create_neuron()

        self.conv2 = sjlayers_t.Conv(planes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     bias=not batch_norm,
                                     step_mode=step_mode)

        if batch_norm:
            self.bn2 = sjlayers_t.BatchNorm(planes, momentum=bn_momentum, step_mode=step_mode)

        self.neuron2 = create_neuron()
        # residual path
        if self.pool_size != 1:
            self.spool = sjlayers_t.MaxPool(pool_size, step_mode=step_mode)
        if (self.in_planes != self.expansion * self.planes
            or (self.force_projection_with_pooling and self.pool_size != 1)
            or self.stride != 1):
            self.sconv = sjlayers_t.Conv(in_planes,
                                         self.expansion * planes,
                                         kernel_size=1,
                                         stride=stride,
                                         bias=not batch_norm,
                                         step_mode=step_mode)
            self.neuronr = create_neuron()
            if batch_norm:
                self.sbn = sjlayers_t.BatchNorm(self.expansion * planes, momentum=bn_momentum, step_mode=step_mode)
        self.add = Add()

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward of ResNet block.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input

        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)
        if self.pool_size != 1:
            out = self.pool1(out)
        out = self.neuron1(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.neuron2(out)

        # shortcut
        tmp = x

        if (self.in_planes != self.expansion * self.planes
            or (self.force_projection_with_pooling and self.pool_size != 1)
            or self.stride != 1):
            tmp = self.sconv(tmp)

            if self.batch_norm:
                tmp = self.sbn(tmp)

        if self.pool_size != 1:
            tmp = self.spool(tmp)

        if (self.in_planes != self.expansion * self.planes
            or (self.force_projection_with_pooling and self.pool_size != 1)
            or self.stride != 1):
            tmp = self.neuronr(tmp)

        return self.add(out, tmp)


class SResNetStride(SNN):
    """Residual spiking neural network template.

    Similar to :class:`qualia_core.learningmodel.pytorch.ResNetStride.ResNetStride` but with spiking neuron activation
    layers (e.g., IF) instead of :class:`torch.nn.ReLU`.


    Example TOML configuration for a 2D ResNetv1-18 over 4 timesteps with soft-reset multi-step IF based on the SResNet template:

    .. code-block:: toml

        [[model]]
        name = "SResNetv1-18"
        params.filters      = [64, 64, 128, 256, 512]
        params.kernel_sizes = [ 7,  3,   3,   3,   3]
        params.paddings     = [ 3,  1,   1,   1,   1]
        params.strides      = [ 2,  1,   1,   1,   1]
        params.pool_sizes   = [ 0,  0,   0,   0,   0]
        params.num_blocks   = [     2,   2,   2,   2]
        params.prepool      = 1
        params.postpool     = 'max'
        params.batch_norm   = true
        params.dims         = 2
        params.timesteps    = 4
        params.neuron.kind  = 'IFNode'
        params.neuron.params.v_reset = false # Soft reset
        params.neuron.params.v_threshold = 1.0
        params.neuron.params.detach_reset = true
        params.neuron.params.step_mode = 'm' # Multi-step mode, make sure to use SpikingJellyMultiStep learningframework
        params.neuron.params.backend = 'torch'
    """

    def __init__(self,  # noqa: C901, PLR0912, PLR0913
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 filters: list[int],
                 kernel_sizes: list[int],
                 pool_sizes: list[int],

                 num_blocks: list[int],
                 strides: list[int],
                 paddings: list[int],
                 prepool: int = 1,
                 postpool: str = 'max',
                 batch_norm: bool = False,  # noqa: FBT001, FBT002
                 bn_momentum: float = 0.1,
                 force_projection_with_pooling: bool = True,  # noqa: FBT001, FBT002

                 neuron: RecursiveConfigDict | None = None,
                 timesteps: int = 2,

                 dims: int = 1,
                 basicblockbuilder: BasicBlockBuilder | None = None) -> None:
        """Construct :class:`SResNet`.

        Structure is:

        .. code-block::

              Input
                |
             AvgPool
                |
               Conv
                |
            BatchNorm
                |
                IF
                |
            BasicBlock
                |
                …
                |
            BasicBlock
                |
            GlobalPool
                |
             Flatten
                |
              Linear

        :param input_shape: Input shape
        :param output_shape: Output shape
        :param filters: List of ``out_channels`` for Conv layers inside each :class:`BasicBlock` group, must be of the same size
                        as ``num_blocks``, first element is for the first Conv layer at the beginning of the network
        :param kernel_sizes: List of ``kernel_size`` for Conv layers inside each :class:`BasicBlock` group, must be of the same
                             size as ``num_blocks``, first element is for the first Conv layer at the beginning of the network
        :param pool_size: List of ``kernel_size`` for MaxPool layers inside each :class:`BasicBlock` group, must be of the same
                          size as ``num_blocks``, ``stride`` is applied only to the first :class:`BasicBlock` of the group, next
                          :class:`BasicBlock` in the group use a ``stride`` of ``1``, first element is the stride of the first Conv
                          layer at the beginning of the network
        :param num_blocks: List of number of :class:`BasicBlock` in each group, also defines the number of :class:`BasicBlock`
                           groups inside the network
        :param strides: List of ``stride`` for first Conv layers inside each :class:`BasicBlock` group, must be of the same size
                        as ``num_blocks``, first element is for the first Conv layer at the beginning of the network
        :param paddings: List of ``padding`` for Conv layer inside each :class:`BasicBlock` group, must be of the same size
                         as ``num_blocks``, first element is for the first Conv layer at the beginning of the network
        :param prepool: AvgPool layer ``kernel_size`` to add at the beginning of the network, no layer added if 0
        :param postpool: Global pooling layer type after all :class:`BasicBlock`, either `max` for MaxPool or `avg` for AvgPool
        :param batch_norm: If ``True``, add a BatchNorm layer after each Conv layer, otherwise no layer added
        :param bn_momentum: BatchNorm ``momentum``
        :param force_projection_with_pooling: If ``True``, residual Conv layer is kept when ``stride != 1`` even if
                                             ``in_planes == planes`` inside a :class:`BasicBlock`
        :param neuron: Spiking neuron configuration, see :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.__init__`
        :param timesteps: Number of timesteps
        :param dims: Either 1 or 2 for 1D or 2D convolutional network.
        :param basicblockbuilder: Optional function with :meth:`BasicBlockBuilder.__call__` signature to build a basic block after
                                  binding constants common across all basic blocks
        """
        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         timesteps=timesteps,
                         neuron=neuron)

        from spikingjelly.activation_based.layer import Flatten, Linear  # type: ignore[import-untyped]

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

        if basicblockbuilder is None:
            def builder(in_planes: int,  # noqa: PLR0913
                        planes: int,
                        kernel_size: int,
                        pool_size: int,
                        stride: int,
                        padding: int) -> BasicBlock:
                return BasicBlock(
                        sjlayers_t=sjlayers_t,
                        in_planes=in_planes,
                        planes=planes,
                        kernel_size=kernel_size,
                        pool_size=pool_size,
                        stride=stride,
                        padding=padding,
                        batch_norm=batch_norm,
                        bn_momentum=bn_momentum,
                        force_projection_with_pooling=force_projection_with_pooling,
                        create_neuron=self.create_neuron,
                        step_mode=self.step_mode)
            basicblockbuilder = builder

        self.in_planes: int = filters[0]
        self.batch_norm = batch_norm
        self.num_blocks = num_blocks

        if prepool > 1:
            self.prepool = sjlayers_t.AvgPool(prepool, step_mode=self.step_mode)

        self.conv1 = sjlayers_t.Conv(input_shape[-1],
                                     filters[0],
                                     kernel_size=kernel_sizes[0],
                                     stride=strides[0],
                                     padding=paddings[0],
                                     bias=not batch_norm,
                                     step_mode=self.step_mode)
        if self.batch_norm:
            self.bn1 = sjlayers_t.BatchNorm(self.in_planes,
                                            momentum=bn_momentum,
                                            step_mode=self.step_mode)
        if pool_sizes[0] != 1:
            self.pool1 = sjlayers_t.MaxPool(pool_sizes[0], step_mode=self.step_mode)

        self.neuron1 = self.create_neuron()

        blocks: list[nn.Sequential] = []
        for planes, kernel_size, pool_size, stride, padding, num_block in zip(filters[1:],
                                                                              kernel_sizes[1:],
                                                                              pool_sizes[1:],
                                                                              strides[1:],
                                                                              paddings[1:],
                                                                              num_blocks):
            blocks.append(self._make_layer(basicblockbuilder, num_block, planes, kernel_size, pool_size, stride, padding))
        self.layers = nn.ModuleList(blocks)

        if postpool == 'sum':
            self.conv2 = sjlayers_t.Conv(in_channels=filters[-1],
                                         out_channels=output_shape[0],
                                         kernel_size=1,
                                         padding=0,
                                         stride=1,
                                         bias=True,
                                         step_mode=self.step_mode)
            self.neuron2 = self.create_neuron()
            self.gsp = layers_t.GlobalSumPool()
        else:
            # GlobalMaxPool kernel_size computation
            self._fm_dims = np.array(input_shape[:-1]) // np.array(prepool)
            for _, kernel, pool_size, stride, padding in zip(filters, kernel_sizes, pool_sizes, strides, paddings):
                self._fm_dims += np.array(padding) * 2
                self._fm_dims -= (kernel - 1)
                self._fm_dims = np.floor(self._fm_dims / pool_size).astype(int)
                self._fm_dims = np.ceil(self._fm_dims / stride).astype(int)

            if postpool == 'avg':
                self.postpool = sjlayers_t.AvgPool(tuple(self._fm_dims), step_mode=self.step_mode)
            elif postpool == 'max':
                self.postpool = sjlayers_t.MaxPool(tuple(self._fm_dims), step_mode=self.step_mode)

            self.flatten = Flatten(step_mode=self.step_mode)
            self.linear = Linear(self.in_planes * BasicBlock.expansion, output_shape[0], step_mode=self.step_mode)

    def _make_layer(self,  # noqa: PLR0913
                    basicblockbuilder: BasicBlockBuilder,
                    num_blocks: int,
                    planes: int,
                    kernel_size: int,
                    pool_size: int,
                    stride: int,
                    padding: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[BasicBlock] = []
        for s in strides:
            block: BasicBlock = basicblockbuilder(in_planes=self.in_planes,
                                      planes=planes,
                                      kernel_size=kernel_size,
                                      pool_size=pool_size,
                                      stride=s,
                                      padding=padding)
            layers.append(block)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward of residual spiking neural network.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input
        if hasattr(self, 'prepool'):
            x = self.prepool(x)

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        if hasattr(self, 'pool1'):
            out = self.pool1(out)
        out = self.neuron1(out)

        for i in range(len(self.layers)):
            out = self.layers[i](out)

        if hasattr(self, 'gsp'):
            out = self.conv2(out)
            out = self.neuron2(out)
            out = self.gsp(out)
        else:
            if hasattr(self, 'postpool'):
                out = self.postpool(out)

            out = self.flatten(out)
            out = self.linear(out)

        return cast('torch.Tensor', out)
