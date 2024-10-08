"""Provide the Qualia-CodeGen NucleoL452REP deployer class with support for Spiking Neural Networks."""

from __future__ import annotations

from importlib.resources import files

from qualia_core.deployment.qualia_codegen.NucleoL452REP import NucleoL452REP as NucleoL452REPQualiaCore
from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.path import resources_to_path

if TYPE_CHECKING:
    from pathlib import Path  # noqa: TCH003


class NucleoL452REP(NucleoL452REPQualiaCore):
    """Qualia-CodeGen NucleoL452REP deployer using example from qualia_codegen-plugin-snn for SNN support."""

    def __init__(self,
                 projectdir: str | Path | None = None,
                 outdir: str | Path | None = None) -> None:
        """Construct :class:`qualia_plugin_snn.deployment.qualia_codegen.NucleoL452REP.NucleoL452REP`.

        :param cxxflags: Override default compiler flags,
            see :meth:`qualia_core.deployment.qualia_codegen.NucleoL452REP.NucleoL452REP.__init__`
        :param modeldir: Path to model C code directory, default: ``out/qualia_codegen``
        :param projectdir: Path to Qualia-CodeGen-Plugin-SNN NucleoL452REP project dir, default:
            ``<qualia_codegen_plugin_snn.examples>/NucleoL452REP``
        :param outdir: Path to build products directory, default: ``out/deploy/NucleoL452REP``
        """
        super().__init__(projectdir=projectdir if projectdir is not None else
                            resources_to_path(files('qualia_codegen_plugin_snn.examples')/'NucleoL452REP'),
                         outdir=outdir)
