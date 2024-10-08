"""Provide the SpikingJelly single-step with timesteps in input data learningframework module."""

import logging
import sys

import torch
from spikingjelly.activation_based import functional  # type: ignore[import-untyped]

from .SpikingJelly import SpikingJelly

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class SpikingJellyMultiStepTimeStepsInData(SpikingJelly):
    """SpikingJelly single-step with timesteps in data LearningFramework implementation extending SpikingJelly single-step."""

    class TrainerModule(SpikingJelly.TrainerModule):
        """SpikingJelly single-step with timesteps in data TrainerModule extending SpikingJelly single-step TrainerModule."""

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass for a Spiking Neural Network model with timesteps in input data in single-step mode.

            First calls SpikingJelly's reset on the model to reset neurons potentials.
            Call :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.forward` for each timestep of the input data.
            Finally, average the output of the model over the timesteps.

            :param x: Input data with timestep dimension in [N, C, T, S] or [N, C, T, H, W] order
            :return: Output predictions
            :raise ValueError: when the input data does not have the correct number of dimenions or the timestep dimension does not
                match :attr:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.timesteps`
            """
            functional.reset_net(self.model)

            # Switch timestep dim from 3rd to 1st place
            if len(x.shape) == 5: # noqa: PLR2004 2D data, [N, C, T, H, W] → [T, N, C, H, W]
                x = x.permute(2, 0, 1, 3, 4)
            elif len(x.shape) == 4: # noqa: PLR2004 1D data [N, C, T, S] → [T, N, C, S]
                x = x.permute(2, 0, 1, 3)
            else:
                logger.error('Unsupported number of axes in dataset: %s, must be 4 or 5', len(x.shape))
                raise ValueError

            if x.shape[0] != self.model.timesteps:
                logger.error('Model.timesteps differs from timesteps dimension in data: %s != %s',
                             self.model.timesteps,
                             x.shape[0])
                raise ValueError

            return self.model(x).sum(0) / self.model.timesteps
