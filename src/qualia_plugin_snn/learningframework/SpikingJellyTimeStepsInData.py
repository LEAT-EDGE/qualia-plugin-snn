"""Provide the SpikingJelly single-step with timesteps in input data learningframework module."""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy.typing
import torch
from spikingjelly.activation_based import functional  # type: ignore[import-untyped]

from qualia_core.datamodel.RawDataModel import RawData

from .SpikingJelly import SpikingJelly

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class SpikingJellyTimeStepsInData(SpikingJelly):
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

            # Switch timestep dim from 2nd to 1st place
            x = x.swapaxes(0, 1)

            input_timesteps = list(x) # Split timestep dims into multiple arrays
            if len(input_timesteps) != self.model.timesteps:
                logger.error('Model.timesteps differs from timesteps dimension in data: %s != %s',
                             self.model.timesteps,
                             len(input_timesteps))
                raise ValueError

            x_list = [self.model(t) for t in input_timesteps]
            x = torch.stack(x_list).sum(0)

            return x / len(input_timesteps)

    @staticmethod
    def channels_last_to_channels_first(x: numpy.typing.NDArray[Any]) -> numpy.typing.NDArray[Any]:
        if len(x.shape) == 5: # N, T, H, W, C → N, T, C, H, W
            x = x.transpose(0, 1, 4, 2, 3)
        elif len(x.shape) == 4: # N, T, S, C → N, T, C, S
            x = x.swapaxes(2, 3)
        else:
            logger.error('Unsupported number of axes in dataset: %s, must be 4 or 5', len(x.shape))
            raise ValueError
        return x

    @staticmethod
    def channels_first_to_channels_last(x: numpy.typing.NDArray[Any]) -> numpy.typing.NDArray[Any]:
        if len(x.shape) == 5: # N, T, C, H, W → N, T, H, W, C
            x = x.transpose(0, 1, 3, 4, 2)
        elif len(x.shape) == 4: # N, T, C, S → N, T, S, C
            x = x.swapaxes(3, 2)
        else:
            logger.error('Unsupported number of axes in dataset: %s, must be 4 or 5', len(x.shape))
            raise ValueError
        return x

    class DatasetFromArray(SpikingJelly.DatasetFromArray):
        def __init__(self, dataset: RawData) -> None:
            self.x = SpikingJellyTimeStepsInData.channels_last_to_channels_first(dataset.x)
            self.y = dataset.y
