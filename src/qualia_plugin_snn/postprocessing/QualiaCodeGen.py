"""Provide the model postprocessing class to generate C code for Spiking Neural Networks using Qualia-CodeGen-Plugin-SNN."""

from __future__ import annotations

import sys
from typing import Any, Callable, Literal

import qualia_core.postprocessing
from qualia_core.typing import TYPE_CHECKING

import qualia_plugin_snn.deployment.qualia_codegen

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from pathlib import Path  # noqa: TC003
    from types import ModuleType  # noqa: TC003

    from qualia_codegen_core.graph import ModelGraph  # noqa: TC002
    from torch import nn  # noqa: TC002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class QualiaCodeGen(qualia_core.postprocessing.QualiaCodeGen):
    """Qualia-CodeGen converter calling Qualia-CodeGen-Plugin-SNN to handle Spiking Neural Network layers."""

    deployers: ModuleType = qualia_plugin_snn.deployment.qualia_codegen
    """:mod:`qualia_plugin_snn.deployment.qualia_codegen` default deployers.

    Includes :class:`qualia_plugin_spleat.deployment.qualia_codegen.Linux.Linux`.

    :meta hide-value:
    """

    def __init__(self,
                 quantize: str,
                 long_width: int | None = None,
                 outdir: str | None = None,
                 metrics: list[str] | None = None,
                 timestep_mode: Literal['duplicate', 'iterate'] = 'duplicate') -> None:
        """Construct :class:`qualia_plugin_snn.postprocessing.QualiaCodeGen.QualiaCodeGen`.

        See :class:`qualia_core.postprocessing.QualiaCodeGen.QualiaCodeGen` for more information.

        :param quantize: Quantization data type
        :param long_width: Long number bit width
        :param outdir: Output directory
        :param metrics: List of metrics to implement
        :param timestep_mode: Input timestep handling mode, either ``'duplicate'`` to duplicate static input data over timesteps,
            or ``'iterate'`` to iterate over existing input data timestep dimension
        """
        super().__init__(quantize=quantize, long_width=long_width, outdir=outdir, metrics=metrics)
        self._timestep_mode = timestep_mode

    @override
    def convert_model_to_modelgraph(self, model: nn.Module) -> ModelGraph | None:
        """Convert PyTorch model to a Qualia-CodeGen ModelGraph graph representation.

        Uses the :class:`qualia_codegen_plugin_snn.graph.TorchModelGraph.TorchModelGraph` from Qualia-CodeGen-Plugin-SNN in order
        to support Spiking Neural Networks.
        The following layers are passed as ``custom_layers``:

        * :class:`spikingjelly.activation_based.layer.Conv1d`, handled like a :class:`torch.nn.Conv1d`
        * :class:`spikingjelly.activation_based.layer.Conv2d`, handled like a :class:`torch.nn.Conv2d`
        * :class:`spikingjelly.activation_based.layer.Flatten`, handled like a :class:`torch.nn.Flatten`
        * :class:`spikingjelly.activation_based.layer.Linear`, handled like a :class:`torch.nn.Linear`
        * :class:`qualia_core.learningmodel.pytorch.layers.Add.Add`, mapped to a
          :class:`qualia_codegen_core.graph.layers.TAddLayer.TAddLayer`
        * :class:`qualia_core.learningmodel.pytorch.layers.GlobalSumPool1d.GlobalSumPool1d`, mapped to a
          :class:`qualia_codegen_core.graph.layers.TSumLayer.TSumLayer`
        * :class:`qualia_core.learningmodel.pytorch.layers.GlobalSumPool2d.GlobalSumPool2d`, mapped to a
          :class:`qualia_codegen_core.graph.layers.TSumLayer.TSumLayer`

        SpikingJelly ``step_mode`` is forced to ``'s'`` for single-step operation to simplify visit of the graph.

        :param model: PyTorch model
        :return: Qualia-CodeGen ModelGraph or None in case of error
        """
        import spikingjelly.activation_based.functional as sjf  # type: ignore[import-untyped]
        import spikingjelly.activation_based.layer as sjl  # type: ignore[import-untyped]
        from qualia_codegen_core.graph.layers import TAddLayer, TBaseLayer, TSumLayer
        from qualia_codegen_plugin_snn.graph import TorchModelGraph
        from qualia_core.learningmodel.pytorch.layers import Add, GlobalSumPool1d, GlobalSumPool2d
        from torch import nn

        custom_layers: dict[type[nn.Module], Callable[[nn.Module, TBaseLayer], tuple[type[TBaseLayer], list[Any]]]]= {
                sjl.BatchNorm1d: TorchModelGraph.MODULE_MAPPING[nn.BatchNorm1d],
                sjl.BatchNorm2d: TorchModelGraph.MODULE_MAPPING[nn.BatchNorm2d],
                sjl.Conv2d: TorchModelGraph.MODULE_MAPPING[nn.Conv2d],
                sjl.Conv1d: TorchModelGraph.MODULE_MAPPING[nn.Conv1d],
                sjl.Conv2d: TorchModelGraph.MODULE_MAPPING[nn.Conv2d],
                sjl.Flatten: TorchModelGraph.MODULE_MAPPING[nn.Flatten],
                sjl.Linear: TorchModelGraph.MODULE_MAPPING[nn.Linear],
                Add: lambda *_: (TAddLayer, []),
                GlobalSumPool1d: lambda *_: (TSumLayer, [(-1,)]),
                GlobalSumPool2d: lambda *_: (TSumLayer, [(-2, -1)]),
                }
        sjf.set_step_mode(model, step_mode='s')

        return TorchModelGraph(model).convert(custom_layers=custom_layers)

    @override
    def convert_modelgraph_to_c(self, modelgraph: ModelGraph, output_path: Path) -> str | bool:
        """Generate C code for the given ModelGraph using Qualia-CodeGen.

        Uses the :class:`qualia_codegen_plugin_snn.Converter.Converter` from Qualia-CodeGen-Plugin-SNN in order to support Spiking
        Neural Networks.

        :param modelgraph: The ModelGraph object from :class:`qualia_codegen_plugin_snn.graph.TorchModelGraph.TorchModelGraph`
            after conversion with :meth:`convert_model_to_modelgraph`
        :param output_path: Generated C code output path
        :return: String containing the single-file C code
        """
        from qualia_codegen_plugin_snn import Converter
        converter = Converter(output_path=output_path, timestep_mode=self._timestep_mode)
        return converter.convert_model(modelgraph)
