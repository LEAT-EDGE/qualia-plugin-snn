"""Generic event-based dataset structure."""

from __future__ import annotations

import sys

from qualia_core.dataset.Dataset import Dataset

from qualia_plugin_snn.datamodel.EventDataModel import EventData, EventDataModel

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class EventDataset(Dataset[EventData]):
    """Generic superclass for event-based datasets."""

    def __init__(self) -> None:
        """Construct :class:`qualia_plugin_snn.dataset.EventDataset.EventDataset`."""
        super().__init__(sets=list(EventDataModel.Sets.fieldnames()))

    @override
    def import_data(self) -> EventDataModel | None:
        """Import data for event-based dataset.

        Relies on :meth:`qualia_plugin_snn.datamodel.EventDataModel.import_data`
        :return: Data model with imported data
        """
        return EventDataModel.import_data(name=self.name, set_names=self.sets)
