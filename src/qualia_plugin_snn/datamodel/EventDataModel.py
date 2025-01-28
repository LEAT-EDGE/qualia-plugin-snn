"""Provide data structures for event-based data processing."""

from __future__ import annotations

import sys
from typing import Any, Callable

from qualia_core.datamodel.DataModel import DataModel
from qualia_core.datamodel.RawDataModel import RawData
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path  # noqa: TC003

    import numpy as np  # noqa: TC002

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

    def __init__(self, sets: EventDataSets, name: str, h: int, w: int) -> None:
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
    @classmethod
    def import_data(cls,
                    name: str,
                    set_names: list[str] | None = None,
                    sets_cls: type[DataModel.Sets[EventData]] = EventDataSets,
                    importer: Callable[[Path], EventData | None] = EventData.import_data) -> EventDataModel | None:
        set_names = set_names if set_names is not None else list(EventDataSets.fieldnames())

        return super().import_data(name=name, set_names=set_names, sets_cls=sets_cls, importer=importer)
