"""Provide a postprocessing module to fuse BatchNorm layers to the previous convolution with SpikingJelly layers support."""

from __future__ import annotations

import sys

import spikingjelly.activation_based.base as sjb  # type: ignore[import-untyped]
import spikingjelly.activation_based.layer as sjl  # type: ignore[import-untyped]
import spikingjelly.activation_based.neuron as sjn  # type: ignore[import-untyped]
from qualia_core.postprocessing.FuseBatchNorm import FuseBatchNorm as FuseBatchNormQualiaCore
from qualia_core.typing import TYPE_CHECKING
from torch.fx.graph_module import GraphModule

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from qualia_core.learningframework.PyTorch import PyTorch  # noqa: TC002
    from torch import nn  # noqa: TC002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class GraphModuleStepModule(GraphModule, sjb.StepModule):  # type: ignore[misc]
    """Mixin of :class:`torch.fx.graph_module.GraphModule` and :class:`spikingjelly.activation_based.base.StepModule`.

    Used by :class:`FuseBatchNorm` to return a module that still inherits
    from class:`spikingjelly.activation_based.base.StepModule`.

    This is required so that SpikingJelly's network-wise operations such as
    :meth:`spikingjelly.activation_based.functional.set_step_mode` work as expected.
    """

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

    extra_custom_layers: tuple[type[nn.Module | sjb.StepModule], ...] = (
            sjn.BaseNode,
            sjb.StepModule, # StepModule not a subclass of nn.Module but still required to avoid parsing sj-wrapped nn layers
            )

    def __init__(self, evaluate: bool = True) -> None:  # noqa: FBT001, FBT002
        """Construct :class:`qualia_plugin_snn.postprocessing.FuseBatchNorm.FuseBatchNorm`.

        Patterns are extended to include SpikingJelly-wrapped layers.
        """
        super().__init__(evaluate=evaluate)

        self.patterns += [
                (sjl.Conv1d, sjl.BatchNorm1d),
                (sjl.Conv2d, sjl.BatchNorm2d),
                (sjl.Conv3d, sjl.BatchNorm3d),
                ]

    @override
    def fuse(self,
             model: nn.Module,
             graphmodule_cls: type[GraphModule],
             framework: PyTorch,
             inplace: bool = False) -> GraphModule:
        """Fuse BatchNorm to Conv and copy source model ``timesteps`` and `is_snn` attributes to target model.

        :param mode: PyTorch model with Conv-BatchNorm layers to fuse
        :param inplace: Modify model in place instead of deep-copying
        :return: Resulting model with BatchNorm fused to Conv
        """
        if isinstance(model, sjb.StepModule):
            graphmodule_cls = GraphModuleStepModule

        fused_model = super().fuse(model,
                                   graphmodule_cls=graphmodule_cls,
                                   framework=framework,
                                   inplace=inplace)
        if hasattr(model, 'timesteps'):
            fused_model.timesteps = model.timesteps
        if hasattr(model, 'step_mode'):
            fused_model.step_mode = model.step_mode
        if hasattr(model, 'is_snn'):
            fused_model.is_snn = model.is_snn
        return fused_model
