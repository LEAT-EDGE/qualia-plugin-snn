"""Provide the Qualia-CodeGen Linux deployer class with support for Spiking Neural Networks."""

from __future__ import annotations

from importlib.resources import files

from qualia_core.deployment.qualia_codegen.Linux import Linux as LinuxQualiaCore
from qualia_core.utils.path import resources_to_path


class Linux(LinuxQualiaCore):
    """Qualia-CodeGen Linux deployer using example from qualia_codegen-plugin-snn for SNN support."""

    def __init__(self,
                 projectdir: str | None = None,
                 outdir: str | None = None) -> None:
        """Construct :class:`qualia_plugin_snn.deployment.qualia_codegen.Linux.Linux`.

        :param cxxflags: Override default compiler flags, see :meth:`qualia_core.deployment.qualia_codegen.Linux.Linux.__init__`
        :param modeldir: Path to model C code directory, default: ``out/qualia_codegen``
        :param projectdir: Path to Qualia-CodeGen-Plugin-SNN Linux project dir, default:
            ``<qualia_codegen_plugin_snn.examples>/Linux``
        :param outdir: Path to build products directory, default: ``out/deploy/Linux``
        """
        super().__init__(projectdir=projectdir if projectdir is not None else
                            resources_to_path(files('qualia_codegen_plugin_snn.examples')/'Linux'),
                         outdir=outdir)
