"""Contain implementation of QuantizedAdd layer with support for SpikingJelly ``step_mode``."""
from __future__ import annotations

import logging
import sys

from qualia_core.learningmodel.pytorch.layers.QuantizedAdd import QuantizedAdd as QualiaCoreQuantizedAdd
from qualia_core.typing import TYPE_CHECKING
from spikingjelly.activation_based.base import StepModule  # type: ignore[import-untyped]

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    import torch  # noqa: TC002
    from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig  # noqa: TC002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class QuantizedAdd(QualiaCoreQuantizedAdd,
          StepModule):  # type: ignore[misc]
    """Add SpikingJelly's ``step_mode`` support to Qualia's QuantizedAdd layer."""

    def __init__(self,
                 quant_params: QuantizationConfig,
                 step_mode: str = 's') -> None:
        """Construct :class:`QuantizedAdd`.

        :param step_mode: SpikingJelly's ``step_mode``, either ``'s'`` or ``'m'``, see
                          :class:`spikingjelly.activation_based.layer.Linear`
        """
        super().__init__(quant_params=quant_params)
        self.step_mode = step_mode

    @override
    def extra_repr(self) -> str:
        """Add ``step_mode`` to the ``__repr__`` method.

        :return: String representation of :class:`qualia_core.learningmodel.pytorch.layers.QuantizedAdd` with ``step_mode``.
        """
        return super().extra_repr() + f', step_mode={self.step_mode}'

    @override
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward :class:`qualia_core.learningmodel.pytorch.layers.QuantizedAdd` with ``step_mode`` support.

        :param input: Input tensor
        :return: Output tensor
        """
        if self.step_mode == 's':
            return super().forward(a, b)

        if self.step_mode == 'm':
            if a.dim() < 3 or b.dim() < 3:  # noqa: PLR2004
                logger.error('Expected inputs with shape [T, N, …], but got inputs with shape (%s, %s)', a.shape, b.shape)
                raise ValueError

            y_shape = [a.shape[0], a.shape[1]]
            y = super().forward(a.flatten(0, 1), b.flatten(0, 1))
            y_shape.extend(y.shape[1:])
            return y.view(y_shape)

        logger.error("step_mode must be either 's' or 'm', got: '%s'", self.step_mode)
        raise ValueError
