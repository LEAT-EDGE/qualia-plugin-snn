"""Contain implementation of QuantizedGlobalSumPool2d layer with support for SpikingJelly ``step_mode``."""
from __future__ import annotations

import logging
import sys

from qualia_core.learningmodel.pytorch.layers.QuantizedGlobalSumPool2d import (
    QuantizedGlobalSumPool2d as QualiaCoreQuantizedGlobalSumPool2d,
)
from qualia_core.typing import TYPE_CHECKING
from spikingjelly.activation_based.base import StepModule  # type: ignore[import-untyped]

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from qualia_core.learningmodel.pytorch.layers.Quantizer import QuantizationConfig

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class QuantizedGlobalSumPool2d(QualiaCoreQuantizedGlobalSumPool2d,
          StepModule):  # type: ignore[misc]
    """GlobalSumPool2d SpikingJelly's ``step_mode`` support to Qualia's QuantizedGlobalSumPool2d layer.

    There is no need to override `:meth:foward` since it works the same for single-step or multi-step mode.
    """

    def __init__(self,
                 quant_params: QuantizationConfig,
                 step_mode: str = 's') -> None:
        """Construct :class:`QuantizedGlobalSumPool2d`.

        :param step_mode: SpikingJelly's ``step_mode``, either ``'s'`` or ``'m'``, see
                          :class:`spikingjelly.activation_based.layer.Linear`
        """
        super().__init__(quant_params=quant_params)
        self.step_mode = step_mode

    @override
    def extra_repr(self) -> str:
        """GlobalSumPool2d ``step_mode`` to the ``__repr__`` method.

        :return: String representation of :class:`qualia_core.learningmodel.pytorch.layers.QuantizedGlobalSumPool2d`
                 with ``step_mode``.
        """
        return super().extra_repr() + f', step_mode={self.step_mode}'
