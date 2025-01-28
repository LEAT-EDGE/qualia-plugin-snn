"""Provide data structures for event-based data processing."""

from __future__ import annotations

from typing import Any

from qualia_core.datamodel.RawDataModel import RawData, RawDataModel
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # noqa: TC002
    from qualia_core.datamodel.DataModel import DataModel  # noqa: TC002


class EventData(RawData):
    """Dataset partition with events as x data."""

    x: np.recarray[Any, Any]

class EventDataModel(RawDataModel):
    """Container for event-based dataset partitions."""

    sets: DataModel.Sets[EventData]
    #: Maximum y coordinate in data
    h: int
    #: Maximum x coordinate in data
    w: int

    def __init__(self, sets: RawDataModel.Sets[EventData], name: str, h: int, w: int) -> None:
        """Instantiate the event-based dataset partitions container.

        :param sets: Collection of dataset partitions
        :param name: Dataset name
        :param h: Maximum y coordinate in data
        :param w: Maximum x coordinate in data
        """
        super().__init__(sets=sets, name=name)
        self.h = h
        self.w = w
