"""Aliases for 1D SpikingJelly and quantized SpikingJelly layers."""

from spikingjelly.activation_based.layer import AvgPool1d as AvgPool  # type: ignore[import-untyped]
from spikingjelly.activation_based.layer import BatchNorm1d as BatchNorm
from spikingjelly.activation_based.layer import Conv1d as Conv
from spikingjelly.activation_based.layer import MaxPool1d as MaxPool

from .quantized_layers1d import QuantizedBatchNorm1d as QuantizedBatchNorm
from .quantized_layers1d import QuantizedConv1d as QuantizedConv
from .quantized_layers1d import QuantizedMaxPool1d as QuantizedMaxPool

__all__ = [
           'AvgPool',
           'BatchNorm',
           'Conv',
           'MaxPool',
           'QuantizedBatchNorm',
           'QuantizedConv',
           'QuantizedMaxPool',
]
