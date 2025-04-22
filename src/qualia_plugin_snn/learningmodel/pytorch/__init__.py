"""Pytorch learningmodel templates for Spiking Neural Networks based on SpikingJelly."""

from .QuantizedSCNN import QuantizedSCNN
from .QuantizedSMLP import QuantizedSMLP
from .QuantizedSResNet import QuantizedSResNet
from .SCNN import SCNN
from .SMLP import SMLP
from .SResNet import SResNet
from .SResNetStride import SResNetStride

__all__ = ['SCNN', 'SMLP', 'QuantizedSCNN', 'QuantizedSMLP', 'QuantizedSResNet', 'SResNet', 'SResNetStride']
