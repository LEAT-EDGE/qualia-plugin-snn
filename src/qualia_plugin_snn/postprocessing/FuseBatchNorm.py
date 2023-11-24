"""Provide a postprocessing module to fuse BatchNorm layers to the previous convolution with SpikingJelly layers support."""

from __future__ import annotations

import sys

import spikingjelly.activation_based.base as sjb  # type: ignore[import-untyped]
import spikingjelly.activation_based.layer as sjl  # type: ignore[import-untyped]
import spikingjelly.activation_based.neuron as sjn  # type: ignore[import-untyped]
from qualia_core.postprocessing.FuseBatchNorm import FuseBatchNorm as FuseBatchNormQualiaCore
from qualia_core.typing import TYPE_CHECKING

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from torch import nn  # noqa: TCH002
    from torch.fx.graph_module import GraphModule  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class FuseBatchNorm(FuseBatchNormQualiaCore):
    """Extend :class:`qualia_core.postprocessing.FuseBatchNorm.FuseBatchNorm` with support for Spiking Neural Networks.

    :class:`spikingjelly.activation_based.neuron.BaseNode` and :class:`spikingjelly.activation_based.base.StepModule`
    are added to :attr:`qualia_core.postprocessing.FuseBatchNorm.FuseBatchNorm.custom_layers` to avoid tracing inside.

    SpikingJelly-wrapped layers are added to lookup patterns:

    * (:class:`spikingjelly.activation_based.layer.Conv1d`, :class:`spikingjelly.activation_based.layer.BatchNorm1d`
    * (:class:`spikingjelly.activation_based.layer.Conv2d`, :class:`spikingjelly.activation_based.layer.BatchNorm2d`
    * (:class:`spikingjelly.activation_based.layer.Conv3d`, :class:`spikingjelly.activation_based.layer.BatchNorm3d`

    Extended attributes ``timesteps`` and ``is_snn`` are copied into target model.
    """

    custom_layers: tuple[type[nn.Module | sjb.StepModule], ...] = (
            sjn.BaseNode,
            sjb.StepModule, # StepModule not a subclass of nn.Module but still required to avoid parsing sj-wrapped nn layers
            *FuseBatchNormQualiaCore.custom_layers,
            )


    def __init__(self) -> None:
        """Construct :class:`qualia_plugin_snn.postprocessing.FuseBatchNorm.FuseBatchNorm`.

        Patterns are extended to include SpikingJelly-wrapped layers.
        """
        super().__init__()

        self.patterns += [
                (sjl.Conv1d, sjl.BatchNorm1d),
                (sjl.Conv2d, sjl.BatchNorm2d),
                (sjl.Conv3d, sjl.BatchNorm3d),
                ]

    @override
    def fuse(self,
             model: nn.Module,
             inplace: bool = False) -> GraphModule:  # noqa: FBT001, FBT002 PyTorch signature
        """Fuse BatchNorm to Conv and copy source model ``timesteps`` and `is_snn` attributes to target model.

        :param mode: PyTorch model with Conv-BatchNorm layers to fuse
        :param inplace: Modify model in place instead of deep-copying
        :return: Resulting model with BatchNorm fused to Conv
        """
        fused_model = super().fuse(model, inplace=inplace)
        if hasattr(model, 'timesteps'):
            fused_model.timesteps = model.timesteps
        if hasattr(model, 'is_snn'):
            fused_model.is_snn = model.is_snn
        return fused_model
