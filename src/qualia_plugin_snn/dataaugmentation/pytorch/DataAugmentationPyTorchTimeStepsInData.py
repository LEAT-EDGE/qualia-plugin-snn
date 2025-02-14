"""Provide a parent class for timestep-aware dataaugmentation modules."""

from __future__ import annotations

from qualia_core.dataaugmentation.pytorch.DataAugmentationPyTorch import DataAugmentationPyTorch


class DataAugmentationPyTorchTimeStepsInData(DataAugmentationPyTorch):
    """Timestep-aware dataaugmentation modules."""

    def __init__(self,
                 evaluate: bool = False,  # noqa: FBT001, FBT002
                 before: bool = False,  # noqa: FBT001, FBT002
                 after: bool = True,  # noqa: FBT001, FBT002
                 collapse_timesteps: bool = False) -> None:  # noqa: FBT002, FBT001
        """Construct :class:`DataAugmentationPyTorchTimeStepsInData`.

        :param evaluate: Also execute during evaluation
        :param before: Execute before transferring batch to GPU
        :param after: Execute after transferring batch to GPU
        :param collapse_timesteps: Mark this dataaugmentation as needing timestep dimension merged with batch dimension, see
            :meth:`qualia_plugin_snn.learningframework.SpikingJellyTimeStepsInData.SpikingJellyTimeStepsInData.TrainerModule.apply_dataaugmentation`
        """
        super().__init__(evaluate=evaluate, before=before, after=after)
        self.collapse_timesteps = collapse_timesteps
