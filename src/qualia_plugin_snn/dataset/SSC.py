"""Spiking Speech Commands dataset import module."""

from __future__ import annotations

from .SHD import SHD


class SSC(SHD):
    """Spiking Speech Commands dataset data loading."""

    def __init__(self, path: str = '', prefix: str = 'ssc') -> None:
        """Instantiate the Spiking Speech Commands dataset loader.

        :param path: Dataset source path
        :param prefix: source file name prefix, default to ``ssc``
        """
        super().__init__(path=path, prefix=prefix)
