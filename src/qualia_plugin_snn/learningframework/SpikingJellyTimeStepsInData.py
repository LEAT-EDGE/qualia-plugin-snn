"""Provide the SpikingJelly single-step with timesteps in input data learningframework module."""

from __future__ import annotations

import logging
import sys
from typing import Any

import torch
from qualia_core.typing import TYPE_CHECKING
from spikingjelly.activation_based import functional  # type: ignore[import-untyped]

from qualia_plugin_snn.dataaugmentation.pytorch.DataAugmentationPyTorchTimeStepsInData import (
    DataAugmentationPyTorchTimeStepsInData,
)

from .SpikingJelly import SpikingJelly

if TYPE_CHECKING:
    import numpy as np  # noqa: TC002
    from qualia_core.dataaugmentation.pytorch.DataAugmentationPyTorch import DataAugmentationPyTorch  # noqa: TC002
    from qualia_core.datamodel.RawDataModel import RawData  # noqa: TC002

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
        def apply_dataaugmentation(self,
                                   batch: tuple[torch.Tensor, torch.Tensor],
                                   dataaugmentation: DataAugmentationPyTorch) -> tuple[torch.Tensor, torch.Tensor]:
            """Call a dataaugmentation module on  the current batch.

            If the dataaugmentation module is not a
            :class:`qualia_plugin_snn.dataaugmentation.pytorch.DataAugmentationPyTorchTimeStepsInData`,
            or if :attr:`qualia_plugin_snn.dataaugmentation.pytorch.DataAugmentationPyTorchTimeStepsInData.collapse_timesteps`
            is ``True``, then the timestep dimension is automatically merged with the batch dimension in order to support
            existing non-timestep-aware Qualia-Core dataaugmentation modules transparently.

            :param batch: Current batch of data and targets
            :param dataaugmentation: dataaugmentation module to apply
            :return: Augmented batch
            """
            if isinstance(dataaugmentation, DataAugmentationPyTorchTimeStepsInData) and not dataaugmentation.collapse_timesteps:
                return super().apply_dataaugmentation(batch, dataaugmentation)

            # Reshape to merge timestep and batch dimensions
            x, y = batch
            shape = x.shape
            x = x.reshape(*(shape[0] * shape[1], *shape[2:]))

            x, y = super().apply_dataaugmentation((x, y), dataaugmentation)

            x = x.reshape(*(shape[0], shape[1], *x.shape[1:]))
            return x, y

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass for a Spiking Neural Network model with timesteps in input data in single-step mode.

            First calls SpikingJelly's reset on the model to reset neurons potentials.
            Call :meth:`qualia_plugin_snn.learningmodel.pytorch.SNN.SNN.forward` for each timestep of the input data.
            Finally, average the output of the model over the timesteps.

            :param x: Input data with timestep dimension in [N, T, C, S] or [N, T, C, H, W] order
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

    @override
    @staticmethod
    def channels_last_to_channels_first(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Channels last to channels first conversion with consideration of timestep dimension.

        For 2D data with timestep: [N, T, H, W, C] → [N, T, C, H, W]

        For 1D data with timestep: [N, T, S, C] → [N, T, C, S]

        :param x: NumPy array in channels last format
        :return: NumPy array reordered to channels first format
        """
        if len(x.shape) == 5: # N, T, H, W, C → N, T, C, H, W  # noqa: PLR2004
            x = x.transpose(0, 1, 4, 2, 3)
        elif len(x.shape) == 4: # N, T, S, C → N, T, C, S  # noqa: PLR2004
            x = x.swapaxes(2, 3)
        else:
            logger.error('Unsupported number of axes in dataset: %s, must be 4 or 5', len(x.shape))
            raise ValueError
        return x

    @override
    @staticmethod
    def channels_first_to_channels_last(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Channels first to channels last conversion with consideration of timestep dimension.

        For 2D data with timestep: [N, T, C, H, W] → [N, T, H, W, C]

        For 1D data with timestep: [N, T, C, S] → [N, T, S, C]

        :param x: NumPy array in channels first format
        :return: NumPy array reordered to channels last format
        """
        if len(x.shape) == 5: # N, T, C, H, W → N, T, H, W, C  # noqa: PLR2004
            x = x.transpose(0, 1, 3, 4, 2)
        elif len(x.shape) == 4: # N, T, C, S → N, T, S, C  # noqa: PLR2004
            x = x.swapaxes(3, 2)
        else:
            logger.error('Unsupported number of axes in dataset: %s, must be 4 or 5', len(x.shape))
            raise ValueError
        return x

    class DatasetFromArray(SpikingJelly.DatasetFromArray):
        """Override :class:`qualia_core.learningframework.PyTorch.PyTorch.DatasetFromArray` to reorder data with timestep dim."""

        def __init__(self, dataset: RawData) -> None:
            """Load a Qualia dataset partition as a PyTorch dataset.

            Reorders the input data from channels_last to channels_first, with consideration of timestep dimension.

            :param dataset: Dataset partition to load
            """
            self.x = SpikingJellyTimeStepsInData.channels_last_to_channels_first(dataset.x)
            self.y = dataset.y
