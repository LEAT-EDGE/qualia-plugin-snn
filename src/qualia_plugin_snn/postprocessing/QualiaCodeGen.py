"""Provide the model postprocessing class to generate C code for Spiking Neural Networks using Qualia-CodeGen-Plugin-SNN."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import qualia_core.postprocessing
from qualia_core.typing import TYPE_CHECKING

import qualia_plugin_snn.deployment.qualia_codegen

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from types import ModuleType  # noqa: TCH003

    from qualia_codegen_core.graph import ModelGraph  # noqa: TCH002
    from torch import nn  # noqa: TCH002

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
    def convert_modelgraph_to_c(self, modelgraph: ModelGraph, output_path: Path) -> str | None:
        """Generate C code for the given ModelGraph using Qualia-CodeGen.

        Uses the :class:`qualia_codegen_plugin_snn.Converter.Converter` from Qualia-CodeGen-Plugin-SNN in order to support Spiking
        Neural Networks.

        :param modelgraph: The ModelGraph object from :class:`qualia_codegen_plugin_snn.graph.TorchModelGraph.TorchModelGraph`
            after conversion with :meth:`convert_model_to_modelgraph`
        :param output_path: Generated C code output path
        :return: String containing the single-file C code
        """
        from qualia_codegen_plugin_snn import Converter
        converter = Converter(output_path=output_path)
        return converter.convert_model(modelgraph)
