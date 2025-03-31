"""Provide data structures for event-based data processing."""

from __future__ import annotations

import sys
from typing import Any, Callable

from qualia_core.datamodel.DataModel import DataModel
from qualia_core.datamodel.RawDataModel import RawData
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class EventData(RawData):
    """Dataset partition with events as x data."""

    x: np.recarray[Any, Any]

class EventDataSets(DataModel.Sets[EventData]):
    """Container for event-based dataset partitions."""

class EventDataModel(DataModel[EventData]):
    """Container for event-based data model."""

    sets: DataModel.Sets[EventData]
    #: Maximum y coordinate in data
    h: int
    #: Maximum x coordinate in data
    w: int

    def __init__(self, name: str, h: int, w: int, sets: DataModel.Sets[EventData] | None = None) -> None:
        """Instantiate the event-based dataset partitions container.

        :param sets: Collection of dataset partitions
        :param name: Dataset name
        :param h: Maximum y coordinate in data
        :param w: Maximum x coordinate in data
        """
        super().__init__(sets=sets, name=name)
        self.h = h
        self.w = w

    @override
    def import_sets(self,
                    set_names: list[str] | None = None,
                    sets_cls: type[DataModel.Sets[EventData]] = EventDataSets,
                    importer: Callable[[Path], EventData | None] = EventData.import_data) -> None:
        set_names = set_names if set_names is not None else list(EventDataSets.fieldnames())

        sets_dict = self._import_data_sets(name=self.name, set_names=set_names, importer=importer)

        if sets_dict is not None:
            self.sets = sets_cls(**sets_dict)
