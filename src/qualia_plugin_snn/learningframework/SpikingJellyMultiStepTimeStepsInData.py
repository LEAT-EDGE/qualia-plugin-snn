"""Provide the SpikingJelly multi-step with timesteps in input data learningframework module."""

from __future__ import annotations

import logging
import sys

from qualia_core.typing import TYPE_CHECKING
from spikingjelly.activation_based import functional  # type: ignore[import-untyped]

from .SpikingJellyTimeStepsInData import SpikingJellyTimeStepsInData

if TYPE_CHECKING:
    import torch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class SpikingJellyMultiStepTimeStepsInData(SpikingJellyTimeStepsInData):
    """SpikingJelly multi-step with timesteps in data LearningFramework implementation extending SpikingJelly single-step."""

    class TrainerModule(SpikingJellyTimeStepsInData.TrainerModule):
        """SpikingJelly multi-step with timesteps in data TrainerModule extending SpikingJelly single-step TrainerModule."""

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass for a Spiking Neural Network model with timesteps in input data in multi-step mode.

            First calls SpikingJelly's reset on the model to reset neurons potentials.
            Call :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.forward` for each timestep of the input data.
            Finally, average the output of the model over the timesteps.

            :param x: Input data with timestep dimension in [N, T, C, S] or [N, T, C, H, W] order
            :return: Output predictions
            :raise ValueError: when the input data does not have the correct number of dimenions or the timestep dimension does not
                match :attr:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.timesteps`
            """
            functional.reset_net(self.model)

            # Switch timestep dim from 2nd to 1st place, [N, T, C, H, W] → [T, N, C, H, W]
            x = x.swapaxes(0, 1)

            if x.shape[0] != self.model.timesteps:
                logger.error('Model.timesteps differs from timesteps dimension in data: %s != %s',
                             self.model.timesteps,
                             x.shape[0])
                raise ValueError

            return self.model(x).sum(0) / self.model.timesteps
