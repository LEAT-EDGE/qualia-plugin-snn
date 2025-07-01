"""DVS128-Gesture event-based dataset import module based on SpikingJelly including preprocessing to frames and timesteps."""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
from qualia_core.datamodel import RawDataModel
from qualia_core.datamodel.RawDataModel import RawData
from qualia_core.dataset.RawDataset import RawDataset
from qualia_core.utils.process.init_process import init_process
from qualia_core.utils.process.SharedMemoryManager import SharedMemoryManager
from spikingjelly.datasets import integrate_events_by_fixed_duration  # type: ignore[import-untyped]
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture  # type: ignore[import-untyped]

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

LoadFramesReturnT = tuple[tuple[str, tuple[int, ...], npt.DTypeLike], tuple[str, tuple[int, ...], npt.DTypeLike]]
SharedMemoryArrayReturnT = tuple[str, tuple[int, ...], npt.DTypeLike]

class DVSGestureWithPreprocessing(RawDataset):
    """DVS128 Gesture event-based data loading based on SpikingJelly including preprocessing to frames and timesteps."""

    def __init__(self,
                 path: str='',
                 data_type: str = 'frame',
                 duration: int = 0,
                 timesteps: int = 0) -> None:
        """Instantiate the DVS128 Gesture dataset loader with preprocessing.

        :param path: Dataset source path
        :param data_type: Only ``'frame'`` is supported
        :param duration: Frame integration duration
        :param timesteps: Number of timesteps to groupe frames by
        """
        super().__init__()
        self.__path = Path(path)
        self.__data_type = data_type
        self.__duration = duration
        self.__timesteps = timesteps
        self.sets.remove('valid')

    def __load_dvs128gesture(self, *, train: bool) -> DVS128Gesture:
        """Call SpikingJelly loader implementation for DVS128 Gesture.

        :param train: Load train data if ``True``, otherwise load test data
        """
        self.__path.mkdir(parents=True, exist_ok=True)

        return DVS128Gesture(str(self.__path),
                             train=train,
                             data_type='event')

    def _shared_memory_array(self,
                             smm: SharedMemoryManager,
                             data_array: npt.NDArray[np.float32] |
                                         npt.NDArray[np.int32]) -> SharedMemoryArrayReturnT:
        data_buffer = smm.SharedMemory(size=data_array.nbytes)
        data_shared = np.frombuffer(data_buffer.buf, count=data_array.size, dtype=data_array.dtype).reshape(data_array.shape)

        np.copyto(data_shared, data_array)

        del data_shared

        ret = (data_buffer.name, data_array.shape, data_array.dtype)

        data_buffer.close()

        return ret

    def _load_frames(self,
                     smm_address: str | tuple[str, int],
                     i: int,
                     dvs128gesture: DVS128Gesture,
                     chunks: npt.NDArray[np.int32]) -> LoadFramesReturnT:
        """Subprocess entry point to load and process data for a set of samples.

        :param i: Process number
        :param dvs128gesture: SpikingJelly DVS128Gesture loader
        :param chunks: List of samples to load
        :return: Frames over timesteps and labels for selected samples
        """
        start = time.time()

        smm = SharedMemoryManager(address=smm_address)
        smm.connect()

        logger.info('Process %s loading frames for chunks %s...', i, chunks)

        h: int
        w: int
        h, w = dvs128gesture.get_H_W()

        data_list: list[npt.NDArray[np.float32]] = []
        labels_list: list[npt.NDArray[np.int32]] = []

        for j in chunks:
            data64: npt.NDArray[np.float64] = integrate_events_by_fixed_duration(events=dvs128gesture[j][0],
                                                                                         duration=self.__duration,
                                                                                         H=h,
                                                                                         W=w)
            data = data64.astype(np.float32)
            data = data.transpose((0, 2, 3, 1)) # N, C, H, W → N, H, W, C
            frame_chunks: int = data.shape[0] // self.__timesteps
            data = data[:frame_chunks * self.__timesteps] # Truncate excessive frames
            data = data.reshape((frame_chunks, self.__timesteps, *data.shape[1:])) # N, T, H, W, C
            label: int = dvs128gesture[j][1]
            data_list.append(data)
            labels_list.append(np.full(data.shape[0], label, dtype=np.int32))

        data_array = np.concatenate(data_list)
        labels_array = np.concatenate(labels_list)
        del data_list
        del labels_list

        data_ret = self._shared_memory_array(smm, data_array)
        labels_ret = self._shared_memory_array(smm, labels_array)

        logger.info('Process %s finished in %s s.', i, time.time() - start)
        return data_ret, labels_ret

    def __dvs128gesture_to_data(self, dvs128gesture: DVS128Gesture) -> RawData:
        """Parallel loading and processing of event data to construct frames and timesteps.

        :param dvs128gesture: SpikingJelly DVS128Gesture loader
        :return: Frame and timesteps data and labels
        """
        samples = len(dvs128gesture)
        cpus: int | None = os.cpu_count()
        total_chunks: int = cpus // 2 if cpus is not None else 2
        chunks_list = np.array_split(np.arange(samples, dtype=np.int32), total_chunks)

        with SharedMemoryManager() as smm, ProcessPoolExecutor(initializer=init_process) as executor:
            if smm.address is None: # After smm is started in context, address is necessary non-None
                raise RuntimeError

            train_futures = [executor.submit(self._load_frames, smm.address, i, dvs128gesture, chunks)
                       for i, chunks in enumerate(chunks_list)]

            def load_results(futures: list[Future[LoadFramesReturnT]],
                             resloader: Callable[[LoadFramesReturnT],
                                                 SharedMemoryArrayReturnT]) -> npt.NDArray[np.float32]:

                names = [resloader(f.result())[0] for f in futures]
                shapes = [resloader(f.result())[1] for f in futures]
                dtypes = [resloader(f.result())[2] for f in futures]
                bufs = [SharedMemory(n) for n in names]

                data_list = [np.frombuffer(buf.buf, count=math.prod(shape), dtype=dtype).reshape(shape)
                          for shape, dtype, buf in zip(shapes, dtypes, bufs)]

                data_array = np.concatenate(data_list)
                del data_list

                for buf in bufs:
                    buf.unlink()

                return data_array

            data = load_results(train_futures, lambda r: r[0])
            labels = load_results(train_futures, lambda r: r[1])

        return RawData(data, labels)

    @override
    def __call__(self) -> RawDataModel:
        """Load DVS128 Gesture data as frames over timesteps.

        :return: Data model structure with train and test sets containing frames with timesteps and labels
        """
        if self.__data_type != 'frame':
            logger.error('Unsupported data_type %s', self.__data_type)
            raise ValueError

        train_dvs128gesture = self.__load_dvs128gesture(train=True)
        test_dvs128gesture = self.__load_dvs128gesture(train=False)

        trainset = self.__dvs128gesture_to_data(train_dvs128gesture)
        testset = self.__dvs128gesture_to_data(test_dvs128gesture)


        logger.info('Shapes: train_x=%s, train_y=%s, test_x=%s, test_y=%s',
                    trainset.x.shape if trainset.x is not None else None,
                    trainset.y.shape if trainset.y is not None else None,
                    testset.x.shape if testset.x is not None else None,
                    testset.y.shape if testset.y is not None else None)

        return RawDataModel(sets=RawDataModel.Sets(train=trainset, test=testset), name=self.name)

    @property
    @override
    def name(self) -> str:
        return f'{self.__class__.__name__}_{self.__data_type}_d{self.__duration}_t{self.__timesteps}'
