"""Pytorch learningmodel templates for Spiking Neural Networks based on SpikingJelly."""

from .QuantizedSCNN import QuantizedSCNN
from .QuantizedSResNet import QuantizedSResNet
from .SCNN import SCNN
from .SResNet import SResNet

__all__ = ['SCNN', 'QuantizedSCNN', 'QuantizedSResNet', 'SResNet']
