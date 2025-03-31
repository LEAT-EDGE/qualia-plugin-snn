"""Provide the OperationCounter postprocessing module based on Lemaire et al., 2022."""

from __future__ import annotations

import dataclasses
import logging
import math
import sys
from dataclasses import dataclass
from typing import Literal, NamedTuple

from qualia_core.typing import TYPE_CHECKING, ModelConfigDict
from qualia_core.utils.logger import CSVLogger

from qualia_plugin_snn.learningmodel.pytorch.SNN import SNN

from .EnergyEstimationMetric import EnergyEstimationMetric

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from qualia_codegen_core.graph import ModelGraph
    from qualia_core.qualia import TrainResult
    from torch.types import Number

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class OperationCounterLoggerFields(NamedTuple):
    """Interface object for CSV logging.

    Should contain the same fields as :class:`OperationMetrics` and returned by :meth:`OperationMetrics.asnamedtuple`.

    :param name: Layer name
    :param syn_acc: Number of accumulate operations for synaptic computation
    :param syn_mac: Number of multiply-accumulate operations for synaptic computation
    :param addr_acc: Number of accumulate operations for addressing
    :param addr_mac: Number of multiply-accumulate operations for addressing
    :param total_acc: Total number of accumulate operations
    :param total_mac: Total number of multiply-accumulate operations
    :param mem_read: Number of memory write operations
    :param mem_write: Number of memory read operations
    :param input_spikerate: Average input spike rate per timestep
    :param output_spikerate: Average output spike rate per timestep
    :param input_count: Input count per timestep
    :param output_count: Output count per timestep
    :param input_is_binary: If input tensor only contains binary values, i.e., spikes
    :param output_is_binary: If output tensor only contains binary values, i.e., spikes
    :param is_sj: If the layer is a SpikingJelly layer and has been processed as part of a Spiking Neural Network
    """

    #: Layer name
    name: str
    #: Number of accumulate operations for synaptic computation
    syn_acc: float
    #: Number of multiply-accumulate operations for synaptic computation
    syn_mac: float
    #: Number of accumulate operations for addressing
    addr_acc: float
    #: Number of multiply-accumulate operations for addressing
    addr_mac: float
    #: Total number of accumulate operations
    total_acc: float
    #: Total number of multiply-accumulate operations
    total_mac: float
    #: Number of memory write operations
    mem_write: float
    #: Number of memory read operations
    mem_read: float
    #: Average input spike rate per timestep
    input_spikerate: float | None
    #: Average output spike rate per timestep
    output_spikerate: float | None
    #: Input count per timestep
    input_count: Number | None
    #: Output count per timestep
    output_count: Number | None
    #: If input tensor only contains binary values, i.e., spikes
    input_is_binary: bool
    #: If output tensor only contains binary values, i.e., spikes
    output_is_binary: bool
    #: If the layer is a SpikingJelly layer and has been processed as part of a Spiking Neural Network
    is_sj: bool | Literal['Hybrid']


@dataclass
class OperationMetrics:
    """Holds the computed average operations per inference for each layer.

    :param name: Layer name
    :param syn_acc: Number of accumulate operations for synaptic computation
    :param syn_mac: Number of multiply-accumulate operations for synaptic computation
    :param addr_acc: Number of accumulate operations for addressing
    :param addr_mac: Number of multiply-accumulate operations for addressing
    :param mem_read: Number of memory write operations
    :param mem_write: Number of memory read operations
    :param input_spikerate: Average input spike rate per timestep
    :param output_spikerate: Average output spike rate per timestep
    :param input_count: Input count per timestep
    :param output_count: Output count per timestep
    :param input_is_binary: If input tensor only contains binary values, i.e., spikes
    :param output_is_binary: If output tensor only contains binary values, i.e., spikes
    :param is_sj: If the layer is a SpikingJelly layer and has been processed as part of a Spiking Neural Network
    """

    #: Layer name
    name: str
    #: Number of accumulate operations for synaptic computation
    syn_acc: float
    #: Number of multiply-accumulate operations for synaptic computation
    syn_mac: float
    #: Number of accumulate operations for addressing
    addr_acc: float
    #: Number of multiply-accumulate operations for addressing
    addr_mac: float
    #: Number of memory write operations
    mem_write: float
    #: Number of memory read operations
    mem_read: float
    #: Average input spike rate per timestep
    input_spikerate: float | None
    #: Average output spike rate per timestep
    output_spikerate: float | None
    #: Input count per timestep
    input_count: Number | None
    #: Output count per timestep
    output_count: Number | None
    #: If input tensor only contains binary values, i.e., spikes
    input_is_binary: bool
    #: If output tensor only contains binary values, i.e., spikes
    output_is_binary: bool
    #: If the layer is a SpikingJelly layer and has been processed as part of a Spiking Neural Network
    is_sj: bool | Literal['Hybrid']

    @property
    def total_acc(self) -> float:
        """Total count of accumulate operations."""
        return self.syn_acc + self.addr_acc

    @property
    def total_mac(self) -> float:
        """Total count of accumulate operations."""
        return self.syn_mac + self.addr_mac

    def asnamedtuple(self) -> OperationCounterLoggerFields:
        """Return the data from this class as a NamedTuple for use with the CSV logger.

        Instanciate a :class:`OperationCounterLoggerFields` object with
        all of this class fields and properties and return it.

        :return: the :class:`OperationCounterLoggerFields` with all data from this object copied into it
        """
        return OperationCounterLoggerFields(**dataclasses.asdict(self),
                                            total_acc=self.total_acc,
                                            total_mac=self.total_mac)

class OperationCounter(EnergyEstimationMetric):
    r"""Operation counter metric.

    From `An Analytical Estimation of Spiking Neural Networks Energy Efficiency <https://arxiv.org/abs/2210.13107>`_,
    Lemaire et al. ICONIP2022.

    .. code-block:: bibtex

        @inproceedings{EnergyEstimationMetricICONIP2022,
            title = {An Analytical Estimation of Spiking Neural Networks Energy Efficiency},
            author = {Lemaire, Edgar and Cordone, Loïc and Castagnetti, Andrea
                      and Novac, Pierre-Emmanuel and Courtois, Jonathan and Miramond, Benoît},
            booktitle = {Proceedings of the 29th International Conference on Neural Information Processing},
            pages = {574--587},
            year = {2023},
            doi = {10.1007/978-3-031-30105-6_48},
            series = {ICONIP},
        }

    Supports sequential (non-residual) formal and spiking convolutional neural networks with the following layers:

    * :class:`torch.nn.Conv1d`
    * :class:`torch.nn.Conv2d`
    * :class:`torch.nn.Linear`
    * :class:`torch.nn.ReLU` for formal neural networks
    * :class:`spikingjelly.activation_based.neuron.IFNode` for spiking neural networks
    * :class:`spikingjelly.activation_based.neuron.LIFNode` for spiking neural networks
    """

    #: CSV logger to record metrics for each layer in a file inside the `logs/<bench.name>/OperationCounter` directory
    operationcsvlogger: CSVLogger[OperationCounterLoggerFields]

    def __init__(self, total_spikerate_exclude_nonbinary: bool = True) -> None:  # noqa: FBT001, FBT002
        """Construct :class:`qualia_plugin_snn.postprocessing.OperationCounter.OperationCounter`.

        :param total_spikerate_exclude_nonbinary: If True, exclude non-binary inputs/outputs from total spikerate computation
        """
        super().__init__(mem_width=0,
                         fifo_size=1,
                         total_spikerate_exclude_nonbinary=total_spikerate_exclude_nonbinary)

        self.operationcsvlogger = CSVLogger(name='OperationCounter')
        self.operationcsvlogger.fields = OperationCounterLoggerFields

    def _compute_model_operations_fnn(self,
                                  modelgraph: ModelGraph) -> list[OperationMetrics]:
        """Compute the operations per inference for each layer of a formal neural network.

        Supports the following layers:

        * :class:`qualia_codegen_core.graph.layers.TConvLayer.TConvLayer`
        * :class:`qualia_codegen_core.graph.layers.TDenseLayer.TDenseLayer`
        * :class:`qualia_codegen_core.graph.layers.TAddLayer.TAddLayer`

        :meta public:
        :param modelgraph: Model to compute energy on
        :return: A list of OperationMetrics for each layer and a total with fields populated with operation estimation
        """
        from qualia_codegen_core.graph.layers import TAddLayer, TConvLayer, TDenseLayer, TFlattenLayer

        oms: list[OperationMetrics] = []
        for node in modelgraph.nodes:
            if isinstance(node.layer, TFlattenLayer):
                # Flatten is assumed to not do anything for on-target inference
                om = OperationMetrics(name=node.layer.name,
                                      syn_acc=0,
                                      syn_mac=0,
                                      addr_acc=0,
                                      addr_mac=0,
                                      mem_read=0,
                                      mem_write=0,
                                      input_spikerate=None,
                                      output_spikerate=None,
                                      input_count=None,
                                      output_count=None,
                                      input_is_binary=False,
                                      output_is_binary=False,
                                      is_sj=False)
            elif isinstance(node.layer, TConvLayer):
                om = OperationMetrics(name=node.layer.name,
                                      syn_mac=self._mac_ops_conv_fnn(node.layer),
                                      syn_acc=self._acc_ops_conv_fnn(node.layer),
                                      addr_mac=self._mac_addr_conv_fnn(node.layer),
                                      addr_acc=self._acc_addr_conv_fnn(node.layer),
                                      mem_read=(self._rdin_conv_fnn(node.layer)
                                                + self._rdweights_conv_fnn(node.layer)
                                                + self._rdbias_conv_fnn(node.layer)),
                                      mem_write=self._wrout_conv_fnn(node.layer),
                                      input_spikerate=None,
                                      output_spikerate=None,
                                      input_count=None,
                                      output_count=None,
                                      input_is_binary=False,
                                      output_is_binary=False,
                                      is_sj=False)
            elif isinstance(node.layer, TDenseLayer):
                om = OperationMetrics(name=node.layer.name,
                                      syn_mac=self._mac_ops_fc_fnn(node.layer),
                                      syn_acc=self._acc_ops_fc_fnn(node.layer),
                                      addr_mac=self._mac_addr_fc_fnn(node.layer),
                                      addr_acc=self._acc_addr_fc_fnn(node.layer),
                                      mem_read=(self._rdin_fc_fnn(node.layer)
                                                + self._rdweights_fc_fnn(node.layer)
                                                + self._rdbias_fc_fnn(node.layer)),
                                      mem_write=self._wrout_fc_fnn(node.layer),
                                      input_spikerate=None,
                                      output_spikerate=None,
                                      input_count=None,
                                      output_count=None,
                                      input_is_binary=False,
                                      output_is_binary=False,
                                      is_sj=False)
            elif isinstance(node.layer, TAddLayer):
                # Assume element-wise addition of inputs, meaning we need to read inputs and write outputs

                om = OperationMetrics(name=node.layer.name,
                                      syn_mac=0,
                                      syn_acc=self._acc_ops_add_fnn(node.layer),
                                      addr_mac=self._mac_addr_add_fnn(node.layer),
                                      addr_acc=self._acc_addr_add_fnn(node.layer),
                                      mem_read=self._rdin_add_fnn(node.layer),
                                      mem_write=self._wrout_add_fnn(node.layer),
                                      input_spikerate=None,
                                      output_spikerate=None,
                                      input_count=None,
                                      output_count=None,
                                      input_is_binary=False,
                                      output_is_binary=False,
                                      is_sj=False)

            else:
                logger.warning('%s skipped, result may be inaccurate', node.layer.name)
                continue

            oms.append(om)

        om_total = OperationMetrics(name='Total',
                                 syn_acc=sum(om.syn_acc for om in oms),
                                 syn_mac=sum(om.syn_mac for om in oms),
                                 addr_acc=sum(om.addr_acc for om in oms),
                                 addr_mac=sum(om.addr_mac for om in oms),
                                 mem_read=sum(om.mem_read for om in oms),
                                 mem_write=sum(om.mem_write for om in oms),
                                 input_spikerate=None,
                                 output_spikerate=None,
                                 input_count=None,
                                 output_count=None,
                                 input_is_binary=False,
                                 output_is_binary=False,
                                 is_sj=False)
        oms.append(om_total)

        return oms

    def _compute_model_operations_snn(self,  # noqa: PLR0913, C901, PLR0912, PLR0915
                                  modelgraph: ModelGraph,
                                  input_spikerates: dict[str, float],
                                  output_spikerates: dict[str, float],
                                  input_is_binary: dict[str, bool],
                                  output_is_binary: dict[str, bool],
                                  input_counts: dict[str, Number],
                                  output_counts: dict[str, Number],
                                  is_module_sj: dict[str, bool],
                                  timesteps: int) -> list[OperationMetrics]:
        """Compute the operations per inference for each layer of a spiking neural network.

        Supports the following layers:

        * :class:`qualia_codegen_core.graph.layers.TConvLayer.TConvLayer`
        * :class:`qualia_codegen_core.graph.layers.TDenseLayer.TDenseLayer`

        Input spike rates are per-timestep, this function multiplies by the number of timesteps to get the spike rates per infernce
        which are used by the operation count functions.

        :meta public:
        :param modelgraph: Model to computer operations on
        :param input_spikerate: Dict of layer names and average spike per input per timestep for the layer
        :param output_spikerate: Dict of layer names and average spike per output per timestep for the layer
        :param input_is_binary: Dict of layer names and whether its input is binary (spike) or not
        :param output_is_binary: Dict of layer names and whether its output is binary (spike) or not
        :param input_counts: Dict of layer names and number of inputs for the layer
        :param output_counts: Dict of layer names and number of outputs for the layer
        :param is_module_sj: Whether the layer is a spiking layer (a SpikingJelly module)
        :param timesteps: Number of timesteps
        :return: A list of OperationMetrics for each layer and a total with fields populated with operation count
        """
        from qualia_codegen_core.graph.layers import TAddLayer, TConvLayer, TDenseLayer, TFlattenLayer, TInputLayer
        from qualia_codegen_plugin_snn.graph.layers import TIfLayer, TLifLayer

        oms: list[OperationMetrics] = []
        for node in modelgraph.nodes:
            # Skip dummy input layer
            if isinstance(node.layer, TInputLayer):
                continue

            # TIfLayer is completely hidden in case the previous layer is Conv/Dense since it already contains the required info
            if isinstance(node.layer, TIfLayer) and len(node.innodes) > 0 and isinstance(node.innodes[0].layer, (TConvLayer,
                                                                                                                 TDenseLayer)):
                continue


            # Account for timesteps here since the spikerate has been averaged over timesteps
            input_spikerate = input_spikerates[node.layer.name] * timesteps
            # If no If activation, no output spikes are generated so no reset operation or writing to the output queue
            output_spikerate = (output_spikerates[node.layer.name] * timesteps
                                if len(node.outnodes) > 0 and isinstance(node.outnodes[0].layer, TIfLayer)
                                else 0)

            leak = len(node.outnodes) > 0 and isinstance(node.outnodes[0].layer, TLifLayer)

            om: OperationMetrics | None = None

            if isinstance(node.layer, TFlattenLayer):
                # Flatten is assumed to not do anything for on-target inference
                om = OperationMetrics(name=node.layer.name,
                                      syn_acc=0,
                                      syn_mac=0,
                                      addr_acc=0,
                                      addr_mac=0,
                                      mem_read=0,
                                      mem_write=0,
                                      input_spikerate=input_spikerates[node.layer.name],
                                      output_spikerate=output_spikerates[node.layer.name],
                                      input_count=input_counts[node.layer.name],
                                      output_count=output_counts[node.layer.name],
                                      input_is_binary=input_is_binary[node.layer.name],
                                      output_is_binary=output_is_binary[node.layer.name],
                                      is_sj=is_module_sj[node.layer.name])
            elif is_module_sj[node.layer.name]:
                is_sj: bool | Literal['Hybrid']

                if isinstance(node.layer, TConvLayer):
                    if not input_is_binary[node.layer.name]: # Non-binary dense input:
                        # Computed as sparse input over a single timestep but with MAC operations for membrane potentials increment
                        syn_acc = (self._acc_ops_conv_fnn(node.layer) # Bias
                               + output_spikerate * math.prod(node.layer.output_shape[0][1:])) # Reset
                        syn_mac = (self._mac_ops_conv_fnn(node.layer) * input_spikerates[node.layer.name] # Input * Weight MACs
                               + (math.prod(node.layer.output_shape[0][1:]) if leak else 0)) # Leak
                        addr_acc = self._acc_addr_conv_snn(node.layer, input_spikerates[node.layer.name])
                        addr_mac = self._mac_addr_conv_snn(node.layer, input_spikerates[node.layer.name])
                        mem_read = (self._rdin_snn(node.layer, input_spikerates[node.layer.name])
                                    + self._rdweights_conv_snn(node.layer, input_spikerate)
                                    + (self._rdbias_conv_snn(node.layer, timesteps) if node.layer.use_bias else 0)
                                    + self._rdpot_conv_snn(node.layer, input_spikerate, timesteps))
                        is_sj = 'Hybrid'
                    else:
                        syn_acc = (self._acc_ops_conv_snn(node.layer, input_spikerate, output_spikerate, timesteps))
                        syn_mac = (self._mac_ops_conv_snn(node.layer, timesteps, leak))
                        addr_acc = self._acc_addr_conv_snn(node.layer, input_spikerate)
                        addr_mac = self._mac_addr_conv_snn(node.layer, input_spikerate)
                        mem_read = (self._rdin_snn(node.layer, input_spikerate)
                                    + self._rdweights_conv_snn(node.layer, input_spikerate)
                                    + (self._rdbias_conv_snn(node.layer, timesteps) if node.layer.use_bias else 0)
                                    + self._rdpot_conv_snn(node.layer, input_spikerate, timesteps))
                        is_sj = True

                    om = OperationMetrics(name=node.layer.name,
                                          syn_acc=syn_acc,
                                          syn_mac=syn_mac,
                                          addr_acc=addr_acc,
                                          addr_mac=addr_mac,
                                          mem_read=mem_read,
                                          mem_write = (self._wrout_snn(node.layer, output_spikerate)
                                                       + self._wrpot_conv_snn(node.layer, input_spikerate, timesteps)),
                                          input_spikerate=input_spikerates[node.layer.name],
                                          output_spikerate=output_spikerates[node.layer.name],
                                          input_count=input_counts[node.layer.name],
                                          output_count=output_counts[node.layer.name],
                                          input_is_binary=input_is_binary[node.layer.name],
                                          output_is_binary=output_is_binary[node.layer.name],
                                          is_sj=is_sj,
                                          )
                elif isinstance(node.layer, TDenseLayer):
                    if not input_is_binary[node.layer.name]: # Non-binary dense input:
                        # Computed as sparse input over a single timestep but with MAC operations for membrane potentials increment
                        syn_acc = (self._acc_ops_fc_fnn(node.layer) # Bias
                               + output_spikerate * math.prod(node.layer.output_shape[0][1:])) # Reset
                        syn_mac = (self._mac_ops_fc_fnn(node.layer) * input_spikerates[node.layer.name] # Input * Weight MACs
                               + (math.prod(node.layer.output_shape[0][1:]) if leak else 0)) # Leak
                        addr_acc = self._acc_addr_fc_snn(node.layer, input_spikerates[node.layer.name])
                        addr_mac = self._mac_addr_fc_snn(node.layer)
                        mem_read = (self._rdin_snn(node.layer, input_spikerates[node.layer.name])
                                    + self._rdweights_fc_snn(node.layer, input_spikerate)
                                    + (self._rdbias_fc_snn(node.layer, timesteps) if node.layer.use_bias else 0)
                                    + self._rdpot_fc_snn(node.layer, input_spikerate, timesteps))
                        is_sj = 'Hybrid'
                    else:
                        syn_acc = (self._acc_ops_fc_snn(node.layer, input_spikerate, output_spikerate, timesteps))
                        syn_mac = (self._mac_ops_fc_snn(node.layer, timesteps, leak))
                        addr_acc = self._acc_addr_fc_snn(node.layer, input_spikerate)
                        addr_mac = self._mac_addr_fc_snn(node.layer)
                        mem_read = (self._rdin_snn(node.layer, input_spikerate)
                                    + self._rdweights_fc_snn(node.layer, input_spikerate)
                                    + (self._rdbias_fc_snn(node.layer, timesteps) if node.layer.use_bias else 0)
                                    + self._rdpot_fc_snn(node.layer, input_spikerate, timesteps))

                        is_sj = True

                    om = OperationMetrics(name=node.layer.name,
                                          syn_acc=syn_acc,
                                          syn_mac=syn_mac,
                                          addr_acc=addr_acc,
                                          addr_mac=addr_mac,
                                          mem_read=mem_read,
                                          mem_write = (self._wrout_snn(node.layer, output_spikerate)
                                                       + self._wrpot_fc_snn(node.layer, input_spikerate, timesteps)),
                                          input_spikerate=input_spikerates[node.layer.name],
                                          output_spikerate=output_spikerates[node.layer.name],
                                          input_count=input_counts[node.layer.name],
                                          output_count=output_counts[node.layer.name],
                                          input_is_binary=input_is_binary[node.layer.name],
                                          output_is_binary=output_is_binary[node.layer.name],
                                          is_sj=is_sj,
                                          )
                elif isinstance(node.layer, TAddLayer):
                    om = OperationMetrics(name=node.layer.name,
                                          syn_acc=0,
                                          syn_mac=0,
                                          addr_acc=0,
                                          addr_mac=0,
                                          mem_read=0,
                                          mem_write=0,
                                          input_spikerate=input_spikerates[node.layer.name],
                                          output_spikerate=output_spikerates[node.layer.name],
                                          input_count=input_counts[node.layer.name],
                                          output_count=output_counts[node.layer.name],
                                          input_is_binary=input_is_binary[node.layer.name],
                                          output_is_binary=output_is_binary[node.layer.name],
                                          is_sj=is_module_sj[node.layer.name],
                                          )

            else:  # noqa: PLR5501 keep separate if for clarity and consistency
                if isinstance(node.layer, TConvLayer):
                    om = OperationMetrics(name=node.layer.name,
                                          syn_acc=self._acc_ops_conv_fnn(node.layer),
                                          syn_mac=self._mac_ops_conv_fnn(node.layer),
                                          addr_acc=self._acc_addr_conv_fnn(node.layer),
                                          addr_mac=self._mac_addr_conv_fnn(node.layer),
                                          mem_read=(self._rdin_conv_fnn(node.layer)
                                                    + self._rdweights_conv_fnn(node.layer)
                                                    + (self._rdbias_conv_fnn(node.layer) if node.layer.use_bias else 0)),
                                          mem_write=self._wrout_conv_fnn(node.layer),
                                          input_spikerate=input_spikerates[node.layer.name],
                                          output_spikerate=output_spikerates[node.layer.name],
                                          input_count=input_counts[node.layer.name],
                                          output_count=output_counts[node.layer.name],
                                          input_is_binary=input_is_binary[node.layer.name],
                                          output_is_binary=output_is_binary[node.layer.name],
                                          is_sj=is_module_sj[node.layer.name],
                                          )
                elif isinstance(node.layer, TDenseLayer):
                    om = OperationMetrics(name=node.layer.name,
                                          syn_acc=self._acc_ops_fc_fnn(node.layer),
                                          syn_mac=self._mac_ops_fc_fnn(node.layer),
                                          addr_acc=self._acc_addr_fc_fnn(node.layer),
                                          addr_mac=self._mac_addr_fc_fnn(node.layer),
                                          mem_read=(self._rdin_fc_fnn(node.layer)
                                                    + self._rdweights_fc_fnn(node.layer)
                                                    + (self._rdbias_fc_fnn(node.layer) if node.layer.use_bias else 0)),
                                          mem_write=self._wrout_fc_fnn(node.layer),
                                          input_spikerate=input_spikerates[node.layer.name],
                                          output_spikerate=output_spikerates[node.layer.name],
                                          input_count=input_counts[node.layer.name],
                                          output_count=output_counts[node.layer.name],
                                          input_is_binary=input_is_binary[node.layer.name],
                                          output_is_binary=output_is_binary[node.layer.name],
                                          is_sj=is_module_sj[node.layer.name],
                                          )
                elif isinstance(node.layer, TAddLayer):
                    om = OperationMetrics(name=node.layer.name,
                                          syn_acc=self._acc_ops_add_fnn(node.layer),
                                          syn_mac=self._mac_ops_add_fnn(node.layer),
                                          addr_acc=self._acc_addr_add_fnn(node.layer),
                                          addr_mac=self._mac_addr_add_fnn(node.layer),
                                          mem_read=self._rdin_add_fnn(node.layer),
                                          mem_write=self._wrout_add_fnn(node.layer),
                                          input_spikerate=input_spikerates[node.layer.name],
                                          output_spikerate=output_spikerates[node.layer.name],
                                          input_count=input_counts[node.layer.name],
                                          output_count=output_counts[node.layer.name],
                                          input_is_binary=input_is_binary[node.layer.name],
                                          output_is_binary=output_is_binary[node.layer.name],
                                          is_sj=is_module_sj[node.layer.name],
                                          )

            if om is None: # We do not know how to handle this layer, set energy values to 0
                logger.warning('%s not handled, result may be inaccurate', node.layer.name)
                om = OperationMetrics(name=node.layer.name,
                                      syn_acc=0,
                                      syn_mac=0,
                                      addr_acc=0,
                                      addr_mac=0,
                                      mem_read=0,
                                      mem_write=0,
                                      input_spikerate=input_spikerates[node.layer.name],
                                      output_spikerate=output_spikerates[node.layer.name],
                                      input_count=input_counts[node.layer.name],
                                      output_count=output_counts[node.layer.name],
                                      input_is_binary=input_is_binary[node.layer.name],
                                      output_is_binary=output_is_binary[node.layer.name],
                                      is_sj=is_module_sj[node.layer.name])
            oms.append(om)

        total_is_sj: bool | Literal['Hybrid'] = (True if all(is_sj is True for is_sj in is_module_sj.values()) else
                                                 False if all(not is_sj for is_sj in is_module_sj.values()) else
                                                 'Hybrid')

        om_total = OperationMetrics(name='Total',
                                 syn_acc=sum(om.syn_acc for om in oms),
                                 syn_mac=sum(om.syn_mac for om in oms),
                                 addr_acc=sum(om.addr_acc for om in oms),
                                 addr_mac=sum(om.addr_mac for om in oms),
                                 mem_read=sum(om.mem_read for om in oms),
                                 mem_write=sum(om.mem_write for om in oms),
                                 input_spikerate=input_spikerates['__TOTAL__'],
                                 output_spikerate=output_spikerates['__TOTAL__'],
                                 input_count=input_counts['__TOTAL__'],
                                 output_count=output_counts['__TOTAL__'],
                                 input_is_binary=input_is_binary[modelgraph.nodes[1].layer.name], # First layer after 'input'
                                 output_is_binary=output_is_binary[modelgraph.nodes[-1].layer.name],
                                 is_sj=total_is_sj,
                                 )
        oms.append(om_total)

        return oms

    def _operations_summary(self, oms: list[OperationMetrics]) -> str:
        """Generate a human-friendly text summary of the operations per layer.

        :meta public:
        :param oms: List of OperationMetrics per layer and the total
        :return: The text summary
        """
        pad = 11
        pad_name = max(len(om.name) for om in oms)

        header = f'{"Layer": <{pad_name}} |'
        header += f' {"Syn. Acc": <{pad}} | {"Syn. MAc": <{pad}} |'
        header += f' {"Addr. Acc": <{pad}} | {"Addr. MAc": <{pad}} |'
        header += f' {"Tot. Acc": <{pad}} | {"Tot. MAc": <{pad}} |'
        header += f' {"Mem. read": <{pad}} | {"Mem. write": <{pad}} |'
        header += f' {"SNN": <{pad}} | {"Input Spike Rate": <{pad}} | {"Output Spike Rate": <{pad}}\n'
        s = '—' * len(header) + '\n'
        s += header
        s += '—' * len(header) + '\n'
        for i, om in enumerate(oms):
            # Print in nJ, original values are in pJ
            s += f'{om.name: <{pad_name}.{pad_name}} |'
            s += f' {float(om.syn_acc): <{pad}.4} | {float(om.syn_mac): <{pad}.4} |'
            s += f' {float(om.addr_acc): <{pad}.4} | {float(om.addr_mac): <{pad}.4} |'
            s += f' {float(om.total_acc): <{pad}.4} | {float(om.total_mac): <{pad}.4} |'
            s += f' {float(om.mem_read): <{pad}.4} | {float(om.mem_write): <{pad}.4} |'
            s += f' {om.is_sj!s: <{pad}} |'

            input_spikerate_pad = 16 if om.input_is_binary else 11
            if om.input_spikerate is not None:
                s += f' {om.input_spikerate: <{input_spikerate_pad}.4}'
            else:
                s += f' {"N/A": <{input_spikerate_pad}.16}'
            if not om.input_is_binary:
                s += ' (NB)'
            s += ' |'

            output_spikerate_pad = 16 if om.output_is_binary else 11
            if om.output_spikerate is not None:
                s += f' {om.output_spikerate: <{output_spikerate_pad}.4}'
            else:
                s += f' {"N/A": <{output_spikerate_pad}.16}'
            if not om.output_is_binary:
                s += ' (NB)'
            s += '\n'

            s += ('-' if i < len(oms) - 2 else '—') * len(header) + '\n'

        s += ' NB = Non binary data'

        return s

    @override
    def __call__(self,
                 trainresult: TrainResult,
                 model_conf: ModelConfigDict) -> tuple[TrainResult, ModelConfigDict]:
        """Compute operation count metric from Lemaire et al, 2022.

        First process the model to extract the graph and activity in case of SNN using :meth:`_process_model`.
        Then call either :meth:`_compute_model_operations_snn` or :meth:`_compute_model_operations_fnn` depending on whether
        the model is an SNN or an FNN.
        Print the resulting metrics and log them to a CSV file inside the `logs/<bench.name>/OperationCounter` directory.

        :meta public:
        :param trainresult: TrainResult containing the SNN or FNN model, the dataset and the training configuration
        :param model_conf: Unused
        :return: The unmodified trainresult
        """
        (modelgraph,
         input_spikerates,
         output_spikerates,
         input_is_binary,
         output_is_binary,
         input_counts,
         output_counts,
         is_module_sj) = self._process_model(trainresult=trainresult)

        if modelgraph is None:
            return trainresult, model_conf

        if getattr(trainresult.model, 'is_snn', False) or isinstance(trainresult.model, SNN):
            if (input_spikerates is None
                or output_spikerates is None
                or input_is_binary is None
                or output_is_binary is None
                or input_counts is None
                or output_counts is None
                or is_module_sj is None):
                logger.error('SNN model detected but one of the SNN metric could not be computed')
                raise RuntimeError

            oms = self._compute_model_operations_snn(modelgraph,
                                                 input_spikerates,
                                                 output_spikerates,
                                                 input_is_binary,
                                                 output_is_binary,
                                                 input_counts,
                                                 output_counts,
                                                 is_module_sj,
                                                 trainresult.model.timesteps)
        else:
            oms = self._compute_model_operations_fnn(modelgraph)

        for om in oms:
            self.operationcsvlogger(om.asnamedtuple())

        logger.info(('Estimated operation count for one inference and spike rate per neuron per timestep:\n%s'),
                    self._operations_summary(oms))

        return trainresult, model_conf
