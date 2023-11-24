"""Contains implementation for custom or quantized layers for spiking neural networks."""

from .CustomNode import ATIF, IFSRL

custom_nodes = (ATIF, IFSRL)
__all__ = ['ATIF', 'IFSRL']

layers = custom_nodes
