import logging
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing
from qualia_core.datamodel import RawDataModel
from spikingjelly.datasets import integrate_events_by_fixed_duration
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

logger = logging.getLogger(__name__)

LoadFramesReturnT = tuple[tuple[str, tuple[int, ...], numpy.typing.DTypeLike], tuple[str, tuple[int, ...], numpy.typing.DTypeLike]]
SharedMemoryArrayReturnT = tuple[str, tuple[int, ...], numpy.typing.DTypeLike]

class DVSGesture:
    def __init__(self,
                 path: str='',
                 data_type: str = 'frame',
                 duration: int = 0) -> None:
        super().__init__()
        self.__path = Path(path)
        self.__data_type = data_type
        self.__duration = duration

    def __load_dvs128gesture(self, *, train: bool) -> DVS128Gesture:
        return DVS128Gesture(str(self.__path),
                             train=train,
                             data_type='event')

    def _shared_memory_array(self,
                             data_array: Union[numpy.typing.NDArray[np.float32],
                                               numpy.typing.NDArray[np.int32]]) -> SharedMemoryArrayReturnT:
        data_buffer = SharedMemory(size=data_array.nbytes, create=True)
        data_shared = np.frombuffer(data_buffer.buf, dtype=data_array.dtype).reshape(data_array.shape)

        np.copyto(data_shared, data_array)

        del data_shared

        ret = (data_buffer.name, data_array.shape, data_array.dtype)

        data_buffer.close()

        return ret

    def _load_frames(self,
                     i: int,
                     dvs128gesture: DVS128Gesture,
                     chunks: numpy.typing.NDArray[np.int32]) -> LoadFramesReturnT:
        start = time.time()

        logger.info('Process %s loading frames for chunks %s...', i, chunks)

        h: int
        w: int
        h, w = dvs128gesture.get_H_W()

        data_list: list[numpy.typing.NDArray[np.float32]] = []
        labels_list: list[numpy.typing.NDArray[np.int32]] = []

        for j in chunks:
            data64: numpy.typing.NDArray[np.float64] = integrate_events_by_fixed_duration(events=dvs128gesture[j][0],
                                                                                         duration=self.__duration,
                                                                                         H=h,
                                                                                         W=w)
            data = data64.astype(np.float32)
            data = data.transpose((0, 2, 3, 1)) # N, C, H, W → N, H, W, C
            label: int = dvs128gesture[j][1]
            data_list.append(data)
            labels_list.append(np.full(data.shape[0], label, dtype=np.int32))

        data_array = np.concatenate(data_list)
        labels_array = np.concatenate(labels_list)
        del data_list
        del labels_list

        data_ret = self._shared_memory_array(data_array)
        labels_ret = self._shared_memory_array(labels_array)

        logger.info('Process %s finished in %s s.', i, time.time() - start)
        return data_ret, labels_ret

    def __dvs128gesture_to_data(self, dvs128gesture: DVS128Gesture) -> RawDataModel.Data:
        samples = len(dvs128gesture)
        cpus: Optional[int] = os.cpu_count()
        total_chunks: int = cpus // 2 if cpus is not None else 2
        chunks_list = np.array_split(np.arange(samples, dtype=np.int32), total_chunks)

        with SharedMemoryManager() as smm, ProcessPoolExecutor() as executor:
            train_futures = [executor.submit(self._load_frames, i, dvs128gesture, chunks)
                       for i, chunks in enumerate(chunks_list)]

            def load_results(futures: list[Future[LoadFramesReturnT]],
                             resloader: Callable[[LoadFramesReturnT],
                                                 SharedMemoryArrayReturnT]) -> numpy.typing.NDArray[np.float32]:

                names = [resloader(f.result())[0] for f in futures]
                shapes = [resloader(f.result())[1] for f in futures]
                dtypes = [resloader(f.result())[2] for f in futures]
                bufs = [SharedMemory(n) for n in names]

                data_list = [np.frombuffer(buf.buf, dtype=dtype).reshape(shape)
                          for shape, dtype, buf in zip(shapes, dtypes, bufs)]

                data_array = np.concatenate(data_list)
                del data_list

                for buf in bufs:
                    buf.unlink()

                return data_array

            data = load_results(train_futures, lambda r: r[0])
            labels = load_results(train_futures, lambda r: r[1])

        return RawDataModel.Data(data, labels)

    def __call__(self) -> Optional[RawDataModel]:
        if self.__data_type != 'frame':
            logger.error('Unsupported data_type %s', self.__data_type)
            return None

        train_dvs128gesture = self.__load_dvs128gesture(train=True)
        test_dvs128gesture = self.__load_dvs128gesture(train=False)

        return RawDataModel(RawDataModel.Sets(train=self.__dvs128gesture_to_data(train_dvs128gesture),
                                              test=self.__dvs128gesture_to_data(test_dvs128gesture)),
                            name=self.name)

    def import_data(self) -> RawDataModel:
        return RawDataModel.import_data(name=self.name)

    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}_{self.__data_type}_{self.__duration}'
