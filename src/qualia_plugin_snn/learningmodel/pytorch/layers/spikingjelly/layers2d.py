"""Aliases for 2D SpikingJelly and quantized SpikingJelly layers."""

from spikingjelly.activation_based.layer import AvgPool2d as AvgPool  # type: ignore[import-untyped]
from spikingjelly.activation_based.layer import BatchNorm2d as BatchNorm
from spikingjelly.activation_based.layer import Conv2d as Conv
from spikingjelly.activation_based.layer import MaxPool2d as MaxPool

from .quantized_layers2d import QuantizedBatchNorm2d as QuantizedBatchNorm
from .quantized_layers2d import QuantizedConv2d as QuantizedConv
from .quantized_layers2d import QuantizedMaxPool2d as QuantizedMaxPool

__all__ = ['Conv',
           'BatchNorm',
           'AvgPool',
           'MaxPool',
           'QuantizedBatchNorm',
           'QuantizedConv',
           'QuantizedMaxPool']
