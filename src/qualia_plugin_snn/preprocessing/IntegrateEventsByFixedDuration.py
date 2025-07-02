"""Provide a preprocessing module to construct fixed-duration frames from event data."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, cast

import numpy as np
from qualia_core.datamodel.RawDataModel import RawData, RawDataModel, RawDataSets
from qualia_core.learningframework.PyTorch import PyTorch
from qualia_core.preprocessing.Preprocessing import Preprocessing
from qualia_core.typing import TYPE_CHECKING
from spikingjelly.datasets import integrate_events_segment_to_frame  # type: ignore[import-untyped]

from qualia_plugin_snn.datamodel.EventDataModel import EventDataModel

if TYPE_CHECKING:
    from qualia_core.dataset.Dataset import Dataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class IntegrateEventsByFixedDuration(Preprocessing[EventDataModel, RawDataModel]):
    """Preprocessing module to construct fixed-duration frames from event data."""

    def __init__(self, duration: int) -> None:
        """Construct :class:`qualia_plugin_snn.preprocessing.IntegrateEventsByFixedDuration.IntegrateEventsByFixedDuration`.

        :param duration: Duration of frame
        """
        super().__init__()

        self.__duration = duration

    def __integrate_events_segment_to_frame_1d(self,
                                               x: np.ndarray[Any, Any],
                                               p: np.ndarray[Any, Any],
                                               w: int,
                                               j_l: np.intp,
                                               j_r: np.intp) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Like spikingjelly.datasets.integrate_events_segment_to_frame but without y for 1D data."""
        frame: np.ndarray[Any, np.dtype[np.float32]] = np.zeros(shape=[2, w], dtype=np.dtype(np.float32))
        x = x[j_l: j_r].astype(int)  # avoid overflow
        p = p[j_l: j_r]

        mask: list[bool] = []
        mask.append(p == 0)
        mask.append(np.logical_not(mask[0]))

        for c in range(2):
            position = x[mask[c]]
            events_number_per_pos = np.bincount(position)
            frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos

        return frame

    def _integrate_events_segment_to_frame(self,
                                           events: np.recarray[Any, Any],
                                           h: int,
                                           w: int,
                                           left: np.intp,
                                           right: np.intp) -> np.ndarray[Any, np.dtype[np.float32]]:
        if not hasattr(events, 'y'): # No y means 1D data
            return self.__integrate_events_segment_to_frame_1d(events.x, events.p, w, left, right)
        return cast('np.ndarray[Any, np.dtype[np.float32]]',
                    integrate_events_segment_to_frame(events.x, events.y, events.p, h, w, left, right))


    # Adapted from SpikingJelly
    def __integrate_events_by_fixed_duration(self,
                                             events: np.recarray[Any, Any],
                                             labels: np.ndarray[Any, Any],
                                             h: int,
                                             w: int) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]]:
        """Integrate events from a fixed-duration window to a frame.

        Adapted from SpikingJelly :meth:`spikingjelly.datasets.integrate_events_by_fixed_duration`

        Additionally reduces labels of each event over the integration window.

        :param events: Collection of events with (t, x, y, p) columns
        :param labels: Labels for each event
        :param h: Height of the frame
        :param w: Width of the frame
        :return: Frame after integration and labels after reduction
        """
        n = events.t.size

        # Origin at first event timestamp
        events.t -= events.t.min()

        frames_num = int(np.ceil(events.t[-1] / self.__duration))
        if not hasattr(events, 'y'): # 1D Data
            frames = np.zeros([frames_num, 2, w], dtype=np.float32)
        else:
            frames = np.zeros([frames_num, 2, h, w], dtype=np.float32)
        frame_index = events.t // self.__duration
        left = np.intp(0)

        reduced_labels: np.ndarray[Any, Any] = np.ndarray((frames_num,), dtype=labels.dtype)

        for i in range(frames_num - 1):
            right = np.searchsorted(frame_index, i + 1, side='left')
            frames[i] = self._integrate_events_segment_to_frame(events, h, w, left, right)

            # Find most common label in frame
            values, counts = np.unique(labels[left:right], return_counts=True)
            reduced_labels[i] = values[counts.argmax()]

            left = right

        frames[-1] = self._integrate_events_segment_to_frame(events, h, w, left, n)

        # Find most common label in frame
        values, counts = np.unique(labels[left:n], return_counts=True)
        reduced_labels[-1] = values[counts.argmax()]

        return frames, reduced_labels


    @override
    def __call__(self, datamodel: EventDataModel) -> RawDataModel:
        """Construct frames from events of the same sample of a :class:`qualia_plugin_snn.datamodel.EventDataModel.EventDataModel`.

        Relies on sample indices (begin, end) from the info array to only collect events from the same sample.

        Input data should be 2D event data with (t, x, y, p) columns or 1D event data with (t, x, p) columns.
        Output data has [N, H, W, C] or [N, W, C] dimensions

        Uses SpikingJelly's implementation of :meth:`spikingjelly.datasets.integrate_events_segment_to_frame`

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
            for sample in s.info:  # For each input sample
                # recarray.__getitem__ should return a recarray bus is not type-hinted as such and inherits ndarray type hint
                events = cast('np.recarray[Any, Any]', s.x[sample.begin:sample.end])

                data, labels = self.__integrate_events_by_fixed_duration(
                        events=events,
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
