"""Contain implementation of quantized 1D layers with support for SpikingJelly ``step_mode``."""
from __future__ import annotations

import logging
import sys
from typing import cast

from qualia_core.learningmodel.pytorch.layers import quantized_layers1d
from qualia_core.typing import TYPE_CHECKING
from spikingjelly.activation_based import functional  # type: ignore[import-untyped]
from spikingjelly.activation_based.base import StepModule  # type: ignore[import-untyped]

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    import torch
    from qualia_core.learningmodel.pytorch.layers.Quantizer import QuantizationConfig
    from torch import nn

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class QuantizedConv1d(quantized_layers1d.QuantizedConv1d,
                      StepModule):  # type: ignore[misc]
    """Add SpikingJelly's ``step_mode`` support to Qualia's quantized Conv1d layer."""

    def __init__(self,  # noqa: PLR0913
                 in_channels: int,
                 out_channels: int,
                 quant_params: QuantizationConfig,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,  # noqa: FBT001, FBT002
                 activation: nn.Module | None = None,
                 step_mode: str = 's') -> None:
        """Construct :class:`QuantizedConv1d`.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels, i.e., number of filters
        :param kernel_size: Size of the kernel
        :param stride: Stride of the convolution
        :param padding: Padding added to both sides of the input
        :param dilation: Spacing between kernel elements
        :param groups: Number of blocked connections from input channels to output channels
        :param bias: If ``True``, adds a learnable bias to the output
        :param quant_params: Dict containing quantization parameters, see
                             :class:`qualia_core.learningmodel.pytorch.layers.Quantizer.Quantizer`
        :param activation: Activation layer to fuse for quantization purposes, ``None`` if unfused or no activation
        :param step_mode: SpikingJelly's ``step_mode``, either ``'s'`` or ``'m'``, see
                          :class:`spikingjelly.activation_based.layer.Linear`
        """
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         quant_params=quant_params,
                         activation=activation)
        self.step_mode = step_mode

    @override
    def extra_repr(self) -> str:
        """Add ``step_mode`` to the ``__repr__`` method.

        :return: String representation of :class:`torch.nn.Conv1d` with ``step_mode``.
        """
        return cast('str', super().extra_repr()) + f', step_mode={self.step_mode}'  # type: ignore[no-untyped-call]

    # From spikingjelly.activation_based.layer.Conv1d
    # Copied to call super().forward(x) of quantized_layers1d.Conv1d instead of torch.nn.Conv1d
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward :class:`qualia_core.learningmodel.pytorch.quantized_layers1d.QuantizedConv1d` with ``step_mode`` support.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4:  # noqa: PLR2004
                logger.error('Expected input with shape [T, N, C, L], but got input with shape %s', x.shape)
                raise ValueError
            x = functional.seq_to_ann_forward(x, super().forward)

        return x

class QuantizedMaxPool1d(quantized_layers1d.QuantizedMaxPool1d,
                         StepModule):  # type: ignore[misc]
    """Add SpikingJelly's ``step_mode`` support to Qualia's quantized MaxPool1d layer."""

    def __init__(self,  # noqa: PLR0913
                 kernel_size: int,
                 quant_params: QuantizationConfig,
                 stride: int | None = None,
                 padding: int = 0,
                 activation: nn.Module | None = None,
                 step_mode: str = 's') -> None:
        """Construct :class:`QuantizedMaxPool1d`.

        :param kernel_size: Size of the sliding window
        :param stride: Stride of the sliding window
        :param padding:  Implicit negative infinity padding to be added on both sides
        :param quant_params: Dict containing quantization parameters, see
                             :class:`qualia_core.learningmodel.pytorch.layers.Quantizer.Quantizer`
        :param activation: Activation layer to fuse for quantization purposes, ``None`` if unfused or no activation
        :param step_mode: SpikingJelly's ``step_mode``, either ``'s'`` or ``'m'``, see
                          :class:`spikingjelly.activation_based.layer.Linear`
        """
        super().__init__(kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         quant_params=quant_params,
                         activation=activation)
        self.step_mode = step_mode

    @override
    def extra_repr(self) -> str:
        """Add ``step_mode`` to the ``__repr__`` method.

        :return: String representation of :class:`torch.nn.MaxPool1d` with ``step_mode``.
        """
        return super().extra_repr() + f', step_mode={self.step_mode}'

    # From spikingjelly.activation_based.layer.MaxPool1d
    # Copied to call super().forward(x) of quantized_layers1d.MaxPool1d instead of torch.nn.MaxPool1d
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward :class:`qualia_core.learningmodel.pytorch.quantized_layers1d.QuantizedMaxPool1d` with ``step_mode`` support.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4:  # noqa: PLR2004
                logger.error('Expected input with shape [T, N, C, L], but got input with shape %s', x.shape)
                raise ValueError
            x = functional.seq_to_ann_forward(x, super().forward)

        return x

class QuantizedBatchNorm1d(quantized_layers1d.QuantizedBatchNorm1d,
                           StepModule):  # type: ignore[misc]
    """Add SpikingJelly's ``step_mode`` support to Qualia's quantized BatchNorm1d layer."""

    def __init__(self,  # noqa: PLR0913
                 num_features: int,
                 quant_params: QuantizationConfig,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,  # noqa: FBT001, FBT002
                 track_running_stats: bool = True,  # noqa: FBT001, FBT002
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 activation: torch.nn.Module | None = None,
                 step_mode: str = 's') -> None:
        """Construct :class:`QuantizedBatchNorm1d`.

        :param num_features: Number of features or channels of the input
        :param eps: Value added to the denominator for numerical stability
        :param momentum: value used for the running_mean and running_var computation
        :param affine: If ``True``, add learnable affine parameters
        :param track_running_stats: If ``True``, track the running mean and variance, otherwise always uses batch statistics
        :param device: Optional device to perform the computation on
        :param dtype: Optional data type
        :param quant_params: Dict containing quantization parameters, see
                             :class:`qualia_core.learningmodel.pytorch.layers.Quantizer.Quantizer`
        :param activation: Activation layer to fuse for quantization purposes, ``None`` if unfused or no activation
        :param step_mode: SpikingJelly's ``step_mode``, either ``'s'`` or ``'m'``, see
                          :class:`spikingjelly.activation_based.layer.Linear`
        """
        super().__init__(num_features=num_features,
                         eps=eps,
                         momentum=momentum,
                         affine=affine,
                         track_running_stats=track_running_stats,
                         device=device,
                         dtype=dtype,
                         quant_params=quant_params,
                         activation=activation)
        self.step_mode = step_mode

    @override
    def extra_repr(self) -> str:
        """Add ``step_mode`` to the ``__repr__`` method.

        :return: String representation of :class:`torch.nn.BatchNorm1d` with ``step_mode``.
        """
        return super().extra_repr() + f', step_mode={self.step_mode}'

    # From spikingjelly.activation_based.layer.MaxPool1d
    # Copied to call super().forward(x) of quantized_layers1d.MaxPool1d instead of torch.nn.MaxPool1d
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward :class:`qualia_core.learningmodel.pytorch.quantized_layers1d.QuantizedBatchNorm1d` with ``step_mode`` support.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4:  # noqa: PLR2004
                logger.error('Expected input with shape [T, N, C, L], but got input with shape %s', x.shape)
                raise ValueError
            x = functional.seq_to_ann_forward(x, super().forward)

        return x
