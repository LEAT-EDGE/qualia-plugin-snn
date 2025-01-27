"""Provide the SpikingJelly single-step learningframework module."""

from __future__ import annotations

import sys

import spikingjelly.activation_based.base as sjb  # type: ignore[import-untyped]
import spikingjelly.activation_based.neuron as sjn  # type: ignore[import-untyped]
import torch
import torch.fx
from qualia_core.learningframework import PyTorch
from qualia_core.typing import TYPE_CHECKING
from spikingjelly.activation_based import functional

import qualia_plugin_snn
import qualia_plugin_snn.learningmodel.pytorch

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from types import ModuleType  # noqa: TC003

    from torch import nn

    from qualia_plugin_snn.learningmodel.pytorch.SNN import SNN  # noqa: TC001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class SpikingJelly(PyTorch):
    """SpikingJelly single-step LearningFramework implementation extending PyTorch.

    :attr:`qualia_core.learningframework.PyTorch.PyTorch.learningmodels` are replaced by the Spiking Neural Networks
    from :mod:`qualia_plugin_snn.learningmodel.pytorch`
    """

    learningmodels: ModuleType = qualia_plugin_snn.learningmodel.pytorch
    """:mod:`qualia_plugin_snn.learningmodel.pytorch` additional learningmodels for Spiking Neural Networks.

    :meta hide-value:
    """

    class TrainerModule(PyTorch.TrainerModule):
        """SpikingJelly single-step TrainerModule implementation extending PyTorch TrainerModule."""

        #: Spiking Neural Network model used by this TrainerModule
        model: SNN

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass for a Spiking Neural Network model with duplicated timesteps.

            First calls SpikingJelly's reset on the model to reset neurons potentials.
            Then duplicate the input to generate the number of timesteps given by
            :attr:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.timesteps`
            Call :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.forward` for each timestep.
            Finally, average the output of the model over the timesteps.

            :param x: Input data
            :return: Output predictions
            """
            functional.reset_net(self.model)
            x_list = [self.model(x) for _ in range(self.model.timesteps)]

            return torch.stack(x_list).sum(0) / self.model.timesteps

    @override
    def trace_model(self,
                    model: nn.Module,
                    extra_custom_layers: tuple[type[nn.Module], ...] = ()) -> tuple[torch.fx.Graph, torch.fx.GraphModule]:
        # StepModule not a subclass of nn.Module but still required to avoid parsing sj-wrapped nn layers
        sj_extra_custom_layers: tuple[type[nn.Module | sjb.StepModule], ...] = (
                sjn.BaseNode,
                sjb.StepModule,
                )
        return super().trace_model(model=model,
                                   extra_custom_layers=(*extra_custom_layers, *sj_extra_custom_layers))
