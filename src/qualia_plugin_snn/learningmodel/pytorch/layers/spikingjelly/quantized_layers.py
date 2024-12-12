"""Contain implementation of a QuantizedLinear layer with support for SpikingJelly ``step_mode``."""

from __future__ import annotations

import sys

from qualia_core.learningmodel.pytorch.layers import quantized_layers
from qualia_core.typing import TYPE_CHECKING
from spikingjelly.activation_based import functional  # type: ignore[import-untyped]
from spikingjelly.activation_based.base import StepModule  # type: ignore[import-untyped]

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    import torch  # noqa: TC002
    from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig  # noqa: TC002
    from torch import nn  # noqa: TC002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class QuantizedLinear(quantized_layers.QuantizedLinear,
                      StepModule):  # type: ignore[misc]
    """Add SpikingJelly's ``step_mode`` support to Qualia's quantized Linear layer."""

    def __init__(self,  # noqa: PLR0913
                 in_features: int,
                 out_features: int,
                 quant_params: QuantizationConfig,
                 bias: bool = True,  # noqa: FBT001, FBT002
                 activation: nn.Module | None = None,
                 step_mode: str = 's') -> None:
        """Construct :class:`QuantizedLinear`.

        :param in_features: Dimension of input vector
        :param out_features: Dimension of output vector and number of neurons
        :param bias: If ``True``, adds a learnable bias to the output.
        :param quant_params: Dict containing quantization parameters, see
                             :class:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer`
        :param activation: Activation layer to fuse for quantization purposes, ``None`` if unfused or no activation
        :param step_mode: SpikingJelly's ``step_mode``, either ``'s'`` or ``'m'``, see
                          :class:`spikingjelly.activation_based.layer.Linear`
        """
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         quant_params=quant_params,
                         activation=activation)
        self.step_mode = step_mode

    @override
    def extra_repr(self) -> str:
        """Add ``step_mode`` to the ``__repr__`` method.

        :return: String representation of :class:`torch.nn.Linear` with ``step_mode``.
        """
        return super().extra_repr() + f', step_mode={self.step_mode}'

    # Copied to call super().forward(x) of quantized_layers1d.Conv1d instead of torch.nn.Conv1d
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward :class:`qualia_core.learningmodel.pytorch.layers.quantized_layers.QuantizedLinear` with ``step_mode`` support.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            x = functional.seq_to_ann_forward(x, super().forward)

        return x
