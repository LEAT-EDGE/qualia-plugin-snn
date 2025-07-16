"""Contain implementation of GlobalSumPool1d layer with support for SpikingJelly ``step_mode``."""
from __future__ import annotations

import logging
import sys

from qualia_core.learningmodel.pytorch.layers.GlobalSumPool1d import GlobalSumPool1d as QualiaCoreGlobalSumPool1d
from spikingjelly.activation_based.base import StepModule  # type: ignore[import-untyped]

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class GlobalSumPool1d(QualiaCoreGlobalSumPool1d,
          StepModule):  # type: ignore[misc]
    """GlobalSumPool1d SpikingJelly's ``step_mode`` support to Qualia's GlobalSumPool1d layer.

    There is no need to override `:meth:foward` since it works the same for single-step or multi-step mode.
    """

    def __init__(self,
                 step_mode: str = 's') -> None:
        """Construct :class:`GlobalSumPool1d`.

        :param step_mode: SpikingJelly's ``step_mode``, either ``'s'`` or ``'m'``, see
                          :class:`spikingjelly.activation_based.layer.Linear`
        """
        super().__init__()
        self.step_mode = step_mode

    @override
    def extra_repr(self) -> str:
        """GlobalSumPool1d ``step_mode`` to the ``__repr__`` method.

        :return: String representation of :class:`qualia_core.learningmodel.pytorch.layers.GlobalSumPool1d` with ``step_mode``.
        """
        return super().extra_repr() + f', step_mode={self.step_mode}'
