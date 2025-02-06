"""DVS128-Gesture event-based dataset import module based on SpikingJelly."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture  # type: ignore[import-untyped]

from qualia_plugin_snn.datamodel.EventDataModel import EventData, EventDataModel, EventDataSets

from .EventDataset import EventDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class DVSGesture(EventDataset):
    """DVS128 Gesture event-based data loading based on SpikingJelly."""

    h: int = 128
    w: int = 128

    def __init__(self,
                 path: str='',
                 data_type: str = 'frame') -> None:
        """Instantiate the DVS128 Gesture dataset loader.

        :param path: Dataset source path
        :param data_type: Only ``'frame'`` is supported
        """
        super().__init__()
        self.__path = Path(path)
        self.__data_type = data_type
        self.sets.remove('valid')

    def __load_dvs128gesture(self, *, train: bool) -> DVS128Gesture:
        """Call SpikingJelly loader implementation for DVS128 Gesture.

        :param train: Load train data if ``True``, otherwise load test data
        """
        self.__path.mkdir(parents=True, exist_ok=True)

        return DVS128Gesture(str(self.__path),
                             train=train,
                             data_type='event')

    def __dvs128gesture_to_event_data(self, dvs128gesture: DVS128Gesture) -> EventData:
        """Load events using SpikingJelly loader and fill event-based data structure.

        :param dvs128gesture: SpikingJelly DVS128Gesture loader
        :return: Event data with timestamps, x and y coordinates, polarity, label, and sample indices
        """
        start = time.time()

        t: list[np.ndarray[Any, np.dtype[np.int64]]] = []
        y: list[np.ndarray[Any, np.dtype[np.int8]]] = []
        x: list[np.ndarray[Any, np.dtype[np.int8]]] = []
        p: list[np.ndarray[Any, np.dtype[np.bool_]]] = []
        labels: list[np.ndarray[Any, np.dtype[np.int8]]] = []

        # Couple of begin and end indices for each sample in concatenated array
        sample_indices = np.recarray((len(dvs128gesture),), dtype=np.dtype([('begin', np.int64), ('end', np.int64)]))

        first = 0
        last = 0
        for i, sample in enumerate(dvs128gesture):
            t.append(sample[0]['t'].astype(np.int64))
            y.append(sample[0]['y'].astype(np.int8))
            x.append(sample[0]['x'].astype(np.int8))
            p.append(sample[0]['p'].astype(np.bool_))
            labels.append(np.full(sample[0]['t'].shape[0], sample[1], dtype=np.int8))

            last += len(t[-1])
            sample_indices[i].begin = first
            sample_indices[i].end = last
            first = last

        t_array = np.concatenate(t)
        y_array = np.concatenate(y)
        x_array = np.concatenate(x)
        p_array = np.concatenate(p)
        labels_array = np.concatenate(labels)


        data = np.rec.fromarrays([t_array, y_array, x_array, p_array],
                dtype=np.dtype([('t', np.int64), ('y', np.int8), ('x', np.int8), ('p', np.bool_)]))

        logger.info('Loading finished in %s s.', time.time() - start)
        return EventData(data, labels_array, info=sample_indices)

    @override
    def __call__(self) -> EventDataModel:
        """Load DVS128 Gesture data as events.

        :return: Data model structure with train and test sets containing events and labels
        """
        if self.__data_type != 'frame':
            logger.error('Unsupported data_type %s', self.__data_type)
            raise ValueError

        train_dvs128gesture = self.__load_dvs128gesture(train=True)
        test_dvs128gesture = self.__load_dvs128gesture(train=False)

        trainset = self.__dvs128gesture_to_event_data(train_dvs128gesture)
        testset = self.__dvs128gesture_to_event_data(test_dvs128gesture)


        logger.info('Shapes: train_x=%s, train_y=%s, test_x=%s, test_y=%s',
                    trainset.x.shape if trainset.x is not None else None,
                    trainset.y.shape if trainset.y is not None else None,
                    testset.x.shape if testset.x is not None else None,
                    testset.y.shape if testset.y is not None else None)

        return EventDataModel(sets=EventDataSets(train=trainset, test=testset),
                              name=self.name,
                              h=self.h,
                              w=self.w)

    @property
    @override
    def name(self) -> str:
        return f'{self.__class__.__name__}_{self.__data_type}'
