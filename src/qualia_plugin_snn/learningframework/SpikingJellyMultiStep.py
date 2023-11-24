"""Provide the SpikingJellyMultiStep learningframework module."""

import sys

import torch
from spikingjelly.activation_based import functional  # type: ignore[import-untyped]

from .SpikingJelly import SpikingJelly

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class SpikingJellyMultiStep(SpikingJelly):
    """SpikingJelly multi-step LearningFramework implementation extending SpikingJelly single-step."""

    class TrainerModule(SpikingJelly.TrainerModule):
        """SpikingJelly multi-step TrainerModule implementation extending SpikingJelly single-step TrainerModule."""

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass for a Spiking Neural Network model with duplicated timesteps in multi-step mode.

            First calls SpikingJelly's reset on the model to reset neurons potentials.
            Then duplicate the input to generate the number of timesteps given by
            :attr:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.timesteps`.
            Timesteps are generated as a new dimension of the input tensor: [N, C, …] → [T, N, C, …]
            Call :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.forward` for the whole tensor.
            Finally, average the output of the model over the timestep dimension.

            :param x: Input data
            :return: Output predictions
            """
            functional.reset_net(self.model)

            # Duplicate input over timesteps
            x = x.unsqueeze(0).repeat(self.model.timesteps, *([1] * len(x.shape)))

            return self.model(x).sum(0) / self.model.timesteps
