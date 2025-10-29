"""DVS128-Gesture event-based dataset import module based on SpikingJelly."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
from qualia_core.datamodel.RawDataModel import RawDataDType, RawDataShape
from qualia_core.typing import TYPE_CHECKING
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture  # type: ignore[import-untyped]

from qualia_plugin_snn.datamodel.EventDataModel import (
    EventData,
    EventDataChunks,
    EventDataChunksModel,
    EventDataChunksSets,
    EventDataInfo,
    EventDataInfoRecord,
)

from .EventDataset import EventDatasetChunks

if TYPE_CHECKING:
    from collections.abc import Generator

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class DVSGesture(EventDatasetChunks):
    """DVS128 Gesture event-based data loading based on SpikingJelly."""

    h: int = 128
    w: int = 128
    dtype = np.dtype([('t', np.int64), ('y', np.int8), ('x', np.int8), ('p', np.bool_)])

    def __init__(self,
                 path: str = '',
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

    def __dvs128gesture_to_event_data(self, dvs128gesture: DVS128Gesture) -> Generator[EventData]:
        """Load events using SpikingJelly loader and fill event-based data structure.

        :param dvs128gesture: SpikingJelly DVS128Gesture loader
        :yield: Event data with timestamps, x and y coordinates, polarity, label, and sample indices
        """
        start = time.time()

        # Couple of begin and end indices for each sample in concatenated array
        sample_indices = EventDataInfo((len(dvs128gesture),))

        for sample in dvs128gesture:
            t = sample[0]['t'].astype(np.int64)
            y = sample[0]['y'].astype(np.int8)
            x = sample[0]['x'].astype(np.int8)
            p = sample[0]['p'].astype(np.bool_)
            labels = np.full(sample[0]['t'].shape[0], sample[1], dtype=np.int8)

            data = np.rec.fromarrays([t, y, x, p],
                    dtype=self.dtype)

            # Each chunk corresponds to a single sample
            sample_indices = EventDataInfo((1,))
            cast('EventDataInfoRecord', sample_indices[0]).begin = np.int64(0)
            cast('EventDataInfoRecord', sample_indices[0]).end = t.shape[0]

            yield EventData(data, labels, sample_indices)

        logger.info('Loading finished in %s s.', time.time() - start)

    @override
    def __call__(self) -> EventDataChunksModel:
        """Load DVS128 Gesture data as events.

        :return: Data model structure with train and test sets containing events and labels
        """
        if self.__data_type != 'frame':
            logger.error('Unsupported data_type %s', self.__data_type)
            raise ValueError

        train_dvs128gesture = self.__load_dvs128gesture(train=True)
        test_dvs128gesture = self.__load_dvs128gesture(train=False)

        shapes = RawDataShape(x=(None,), y=(None,))
        dtypes = RawDataDType(x=self.dtype, y=np.dtype(np.uint8))

        trainset = EventDataChunks(chunks=self.__dvs128gesture_to_event_data(train_dvs128gesture),
                                   shapes=shapes,
                                   dtypes=dtypes)
        testset = EventDataChunks(chunks=self.__dvs128gesture_to_event_data(test_dvs128gesture),
                                  shapes=shapes,
                                  dtypes=dtypes)

        return EventDataChunksModel(sets=EventDataChunksSets(train=trainset, test=testset),
                              name=self.name,
                              h=self.h,
                              w=self.w)

    @property
    @override
    def name(self) -> str:
        return f'{self.__class__.__name__}_{self.__data_type}'
