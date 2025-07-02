"""Provide a preprocessing module to split data into timesteps."""

import sys

from qualia_core.datamodel import RawDataModel
from qualia_core.preprocessing.Preprocessing import Preprocessing

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class Split2TimeSteps(Preprocessing[RawDataModel, RawDataModel]):
    """Preprocessing module to split 1D input dataset into multiple timesteps."""

    def __init__(self, chunks: int) -> None:
        """Construct :class:`qualia_plugin_snn.preprocessing.Split2TimeSteps.Split2TimeSteps`.

        :param chunks: Number of chunks to split the data into
        """
        super().__init__()

        self.__chunks = chunks

    @override
    def __call__(self, datamodel: RawDataModel) -> RawDataModel:
        """Split the given :class:`qualia_core.datamodel.RawDataModel.RawDataModel` into multiple timesteps.

        Input data should be 1D (+ channel) with [N, S, C] order (channels_last).
        Output data has [N, T, S // T, C] dimensions
        Extra data that do not fit in a chunk is truncated.

        :param datamodel: The input dataset
        :return: The dataset with additional timestep dimension
        """
        for _, s in datamodel:
            truncated_dim = (s.x.shape[1] // self.__chunks) * self.__chunks
            s.x = s.x[:,:truncated_dim,:]
            s.x = s.x.reshape((s.x.shape[0], self.__chunks, s.x.shape[1] // self.__chunks, *s.x.shape[2:]))

        return datamodel
