"""Provide a preprocessing module to group frame data from same sample into timesteps."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import numpy as np
from qualia_core.datamodel import RawDataModel
from qualia_core.preprocessing.Preprocessing import Preprocessing

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class Group2TimeStepsBySample(Preprocessing[RawDataModel, RawDataModel]):
    """Preprocessing module to group frame data from same sample into timesteps."""

    def __init__(self, timesteps: int) -> None:
        """Construct :class:`qualia_plugin_snn.preprocessing.Group2TimeStepsBySample.Group2TimeStepsBySample`.

        :param timesteps: Number of timesteps to group frames from a sample
        """
        super().__init__()

        self.__timesteps = timesteps

    @override
    def __call__(self, datamodel: RawDataModel) -> RawDataModel:
        """Group frames from the same samples of from a :class:`qualia_core.datamodel.RawDataModel.RawDataModel` into timesteps.

        Relies on sample indices (begin, end) from the info array to only group frames from the same sample.

        Input data should be 2D or 1D (+ channel) with [N, H, W, C] order (channels_last).
        Output data has [N // T, T, H, W, C] dimensions
        Extra data from a sample that do not fit in a timestep group is truncated.

        :param datamodel: The input dataset
        :return: The dataset with additional timestep dimension
        """
        start = time.time()
        for name, s in datamodel:

            if s.info is None:
                logger.error('Info for %s must be defined', name)
                raise ValueError

            data_list: list[np.ndarray[Any, np.dtype[np.float32]]] = []
            labels_list: list[np.ndarray[Any, Any]] = []
            info_list: list[tuple[int, int]] = []

            first = 0
            last = 0
            for sample in s.info: # For each input sample
                data = s.x[sample.begin:sample.end]
                labels = s.y[sample.begin:sample.end]

                frame_chunks: int = data.shape[0] // self.__timesteps
                data = data[:frame_chunks * self.__timesteps] # Truncate excessive frames
                data = data.reshape((frame_chunks, self.__timesteps, *data.shape[1:])) # N, T, H, W, C

                labels = labels[:frame_chunks * self.__timesteps] # Truncate excessive labels
                labels = labels.reshape((frame_chunks, self.__timesteps, *labels.shape[1:])) # Group labels, [N, T, …]

                # One label per group of timesteps
                new_labels: np.ndarray[Any, Any] = np.ndarray((frame_chunks, *labels.shape[2:]), dtype=labels.dtype)
                # Find most common label in each group of timesteps to reduce over timesteps
                for i, frame_labels in enumerate(labels):
                    values, counts = np.unique(frame_labels, return_counts=True)
                    new_labels[i] = values[counts.argmax()]

                data_list.append(data)
                labels_list.append(new_labels)

                last += len(data)
                info_list.append((first, last))
                first = last

            data_array = np.concatenate(data_list)
            labels_array = np.concatenate(labels_list)
            info_array = np.rec.array(info_list, dtype=np.dtype([('begin', np.int64), ('end', np.int64)]))

            logger.info('Shapes: %s_x=%s, %s_y=%s, %s_info=%s',
                        name, data_array.shape,
                        name, labels_array.shape,
                        name, info_array.shape)

            s.x = data_array
            s.y = labels_array
            s.info = info_array

        logger.info('Grouping to timesteps finished in %s s.', time.time() - start)
        return datamodel
