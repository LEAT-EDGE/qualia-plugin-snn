"""Provide a preprocessing module to construct fixed number of frames from a sample of event data."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import numpy as np
from qualia_core.datamodel.RawDataModel import RawData, RawDataModel, RawDataSets
from qualia_core.learningframework.PyTorch import PyTorch
from qualia_core.typing import TYPE_CHECKING
from spikingjelly.datasets import cal_fixed_frames_number_segment_index  # type: ignore[import-untyped]

from .IntegrateEventsByFixedDuration import IntegrateEventsByFixedDuration

if TYPE_CHECKING:
    from qualia_core.dataset.Dataset import Dataset

    from qualia_plugin_snn.datamodel.EventDataModel import EventDataModel

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class IntegrateEventsByFixedFramesNumber(IntegrateEventsByFixedDuration):
    """Preprocessing module to construct fixed number of frames from a sample of event data."""

    def __init__(self,
                 split_by: str,
                 frames_num: int) -> None:
        """Construct :class:`qualia_plugin_snn.preprocessing.IntegrateEventsByFixedFramesNumber.IntegrateEventsByFixedFramesNumber`.

        :param split_by: ``'time'`` or ``'number'``, see :meth:`spikingjelly.datasets.cal_fixed_frames_number_segment_index`
        :param frames_num: Number of frames to produce per sample
        """  # noqa: E501
        super().__init__(0)

        self.__split_by = split_by
        self.__frames_num = frames_num

    # Adapted from SpikingJelly
    def __integrate_events_by_fixed_frames_number(self,
                                                  events: np.recarray[Any, Any],
                                                  labels: np.ndarray[Any, Any],
                                                  h: int,
                                                  w: int) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]:
        """Integrate events into a fixed number of frames.

        Adapted from SpikingJelly :meth:`spikingjelly.datasets.integrate_events_by_fixed_frames_number`

        Additionally reduces labels of each event over the integration window.

        :param events: Collection of events with (t, x, y, p) columns
        :param labels: Labels for each event
        :param h: Height of the frame
        :param w: Width of the frame
        :return: Frame after integration and labels after reduction
        """
        j_l, j_r = cal_fixed_frames_number_segment_index(events.t, self.__split_by, self.__frames_num)

        if not hasattr(events, 'y'): # 1D Data
            frames = np.zeros([self.__frames_num, 2, w], dtype=np.float32)
        else:
            frames = np.zeros([self.__frames_num, 2, h, w], dtype=np.float32)

        reduced_labels: np.ndarray[Any, Any] = np.ndarray((self.__frames_num,), dtype=labels.dtype)

        for i in range(self.__frames_num):
            frames[i] = self._integrate_events_segment_to_frame(events, h, w, j_l[i], j_r[i])

            # Find most common label in frame
            values, counts = np.unique(labels[j_l[i]:j_r[i]], return_counts=True)
            reduced_labels[i] = values[counts.argmax()]

        return frames, reduced_labels

    @override
    def __call__(self, datamodel: EventDataModel) -> RawDataModel:
        """Construct frames from events of the same sample of a :class:`qualia_plugin_snn.datamodel.EventDataModel.EventDataModel`.

        Relies on sample indices (begin, end) from the info array to only collect events from the same sample.

        Input data should be 2D event data with (t, x, y, p) columns or 1D event data with (t, x, p) columns.
        Output data has [N, H, W, C] or [N, W, C] dimensions

        Uses a derivative of SpikingJelly's implementation of :meth:`spikingjelly.datasets.integrate_events_by_fixed_frames_number`

        :param datamodel: The input event-based dataset
        :return: The new frame dataset
        """
        start = time.time()

        sets: dict[str, RawData] = {}
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
                data, labels = self.__integrate_events_by_fixed_frames_number(
                        events=s.x[sample.begin:sample.end],
                        labels=s.y[sample.begin:sample.end],
                        h=datamodel.h,
                        w=datamodel.w)

                # channels_first to channels_last: N, C, H, W → N, H, W, C or N, C, S → N, S, C
                data = PyTorch.channels_first_to_channels_last(data)

                data_list.append(data)
                labels_list.append(labels)

                # Record new begin and end indices of sample in data array to info array
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

            sets[name] = RawData(x=data_array,
                                 y=labels_array,
                                 info=info_array)

        logger.info('Event integration finished in %s s.', time.time() - start)
        return RawDataModel(sets=RawDataSets(**sets), name=datamodel.name)

    @override
    def import_data(self, dataset: Dataset[Any]) -> Dataset[Any]:
        def func() -> RawDataModel:
            rdm = RawDataModel(name=dataset.name)
            rdm.import_sets(set_names=dataset.sets)
            return rdm
        dataset.import_data = func  # type: ignore[method-assign]
        return dataset
