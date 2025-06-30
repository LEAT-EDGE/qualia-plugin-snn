"""Spiking Heidelberg Digits dataset import module."""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from qualia_plugin_snn.datamodel.EventDataModel import EventData, EventDataModel, EventDataSets

from .EventDataset import EventDataset

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class SHD(EventDataset):
    """Spiking Heidelberg Digits dataset data loading."""

    h: int = 1
    w: int = 700

    def __init__(self, path: str = '', prefix: str = 'shd') -> None:
        """Instantiate the Spiking Heidelberg Digits dataset loader.

        :param path: Dataset source path
        :param prefix: source file name prefix, default to ``shd``
        """
        super().__init__()
        self.__path = Path(path)
        self.__prefix = prefix
        self.sets.remove('valid')

    def __load_shd(self, *, path: Path, part: str) -> EventData:
        import gzip

        import h5py  # type: ignore[import-untyped]

        start = time.time()

        t: list[np.ndarray[Any, np.dtype[np.float16]]] = []
        x: list[np.ndarray[tuple[int, ...], np.dtype[np.uint16]]] = []
        p: list[np.ndarray[Any, np.dtype[np.bool_]]] = []
        labels: list[np.ndarray[Any, np.dtype[np.uint8]]] = []

        logger.info('Reading %s', path/f'{self.__prefix}_{part}.h5.gz')

        with gzip.open(path/f'{self.__prefix}_{part}.h5.gz') as f:
            buf = f.read()

        logger.info('Decompressed in %s s.', time.time() - start)
        fbuf = io.BytesIO(buf)
        dataset = h5py.File(fbuf, 'r')

        spikes = dataset['spikes']
        if not isinstance(spikes, h5py.Group):
            logger.error('Expected "spikes" to be a Group, got: %s', type(spikes))
            raise TypeError

        times = spikes['times']
        if not isinstance(times, h5py.Dataset):
            logger.error('Expected "spikes.times" to be a Dataset, got: %s', type(times))
            raise TypeError

        units = spikes['units']
        if not isinstance(units, h5py.Dataset):
            logger.error('Expected "spikes.units" to be a Dataset, got: %s', type(units))
            raise TypeError

        # Assume lists of ndarray of correct dtype
        t = cast('list[np.ndarray[Any, np.dtype[np.float16]]]', times[...])
        x = cast('list[np.ndarray[tuple[int, ...], np.dtype[np.uint16]]]', units[...])

        source_labels = np.array(dataset['labels'], dtype=np.uint8)

        sample_indices = np.recarray((len(x),), dtype=np.dtype([('begin', np.int64), ('end', np.int64)]))

        first = 0
        last = 0
        for i, sample in enumerate(x):
            labels.append(np.full(sample.shape, source_labels[i], dtype=np.uint8))  # Duplicate labels for all events

            p.append(np.ones_like(sample, dtype=np.bool_))  # Generate only positive spikes

            # Record sample start and end indices
            last += len(labels[-1])
            sample_indices[i].begin = first
            sample_indices[i].end = last
            first = last

        t_array = np.concatenate(t)
        t_array = (t_array.astype(np.float64) * 1000000).astype(np.int64)  # Convert from s to µs
        x_array = np.concatenate(x)
        p_array = np.concatenate(p)
        labels_array = np.concatenate(labels)

        data = np.rec.fromarrays([t_array, x_array, p_array], dtype=np.dtype([('t', np.int64), ('x', np.uint16), ('p', np.bool_)]))

        logger.info('Loading finished in %s s.', time.time() - start)
        return EventData(data, labels_array, info=sample_indices)

    @override
    def __call__(self) -> EventDataModel:
        """Load Spiking Heidelberge Digits data as events.

        :return: Data model structure with train and test sets containing events and labels
        """
        trainset = self.__load_shd(path=self.__path, part='train')
        testset = self.__load_shd(path=self.__path, part='test')


        logger.info('Shapes: train_x=%s, train_y=%s, train_info=%s, test_x=%s, test_y=%s, test_info=%s',
                    trainset.x.shape if trainset.x is not None else None,
                    trainset.y.shape if trainset.y is not None else None,
                    trainset.info.shape if trainset.info is not None else None,
                    testset.x.shape if testset.x is not None else None,
                    testset.y.shape if testset.y is not None else None,
                    testset.info.shape if testset.info is not None else None)

        return EventDataModel(sets=EventDataSets(train=trainset, test=testset),
                              name=self.name,
                              h=self.h,
                              w=self.w)
