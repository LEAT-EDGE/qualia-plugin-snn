"""Provide the Qualia-CodeGen Windows deployer class with support for Spiking Neural Networks."""

from __future__ import annotations

from importlib.resources import files

from qualia_core.deployment.qualia_codegen.Windows import Windows as WindowsQualiaCore
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.path import resources_to_path

if TYPE_CHECKING:
    from pathlib import Path  # noqa: TC003


class Windows(WindowsQualiaCore):
    """Qualia-CodeGen Windows deployer using example from qualia_codegen-plugin-snn for SNN support."""

    def __init__(self,
                 projectdir: str | Path | None = None,
                 outdir: str | Path | None = None) -> None:
        """Construct :class:`qualia_plugin_snn.deployment.qualia_codegen.Windows.Windows`.

        :param cxxflags: Override default compiler flags,
            see :meth:`qualia_core.deployment.qualia_codegen.Windows.Windows.__init__`
        :param modeldir: Path to model C code directory, default: ``out/qualia_codegen``
        :param projectdir: Path to Qualia-CodeGen-Plugin-SNN Linux project dir, default:
            ``<qualia_codegen_plugin_snn.examples>/Linux``
        :param outdir: Path to build products directory, default: ``out/deploy/Windows``
        """
        super().__init__(projectdir=projectdir if projectdir is not None else
                            resources_to_path(files('qualia_codegen_plugin_snn.examples')/'Linux'),
                         outdir=outdir)
