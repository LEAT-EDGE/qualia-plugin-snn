"""Spiking Speech Commands dataset import module."""

from __future__ import annotations

import logging
import sys

from qualia_plugin_snn.datamodel.EventDataModel import EventDataModel, EventDataSets

from .SHD import SHD

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class SSC(SHD):
    """Spiking Speech Commands dataset data loading."""

    def __init__(self, path: str = '', prefix: str = 'ssc') -> None:
        """Instantiate the Spiking Speech Commands dataset loader.

        :param path: Dataset source path
        :param prefix: source file name prefix, default to ``ssc``
        """
        super().__init__(path=path, prefix=prefix)
        self.sets.append('valid')

    @override
    def __call__(self) -> EventDataModel:
        """Load Spiking Heidelberge Digits data as events.

        :return: Data model structure with train and test sets containing events and labels
        """
        trainset = self._load_shd(path=self._path, part='train')
        validset = self._load_shd(path=self._path, part='valid')
        testset = self._load_shd(path=self._path, part='test')

        logger.info('Shapes: train_x=%s, train_y=%s, train_info=%s',
                    trainset.x.shape if trainset.x is not None else None,
                    trainset.y.shape if trainset.y is not None else None,
                    trainset.info.shape if trainset.info is not None else None)

        logger.info('Shapes: valid_x=%s, valid_y=%s, valid_info=%s',
                    validset.x.shape if validset.x is not None else None,
                    validset.y.shape if validset.y is not None else None,
                    validset.info.shape if validset.info is not None else None)

        logger.info('Shapes: test_x=%s, test_y=%s, test_info=%s',
                    testset.x.shape if testset.x is not None else None,
                    testset.y.shape if testset.y is not None else None,
                    testset.info.shape if testset.info is not None else None)

        return EventDataModel(sets=EventDataSets(train=trainset, valid=validset, test=testset),
                              name=self.name,
                              h=self.h,
                              w=self.w)
