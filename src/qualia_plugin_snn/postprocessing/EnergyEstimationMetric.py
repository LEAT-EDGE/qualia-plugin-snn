"""Provide the EnergyEstimationMetric postprocessing module based on Lemaire et al., 2022."""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import sys
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, cast

import torch
from qualia_core.learningmodel.pytorch.layers import Add
from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedIdentity
from qualia_core.learningmodel.pytorch.layers.QuantizedAdd import QuantizedAdd
from qualia_core.learningmodel.pytorch.Quantizer import Quantizer
from qualia_core.postprocessing.PostProcessing import PostProcessing
from qualia_core.typing import TYPE_CHECKING, ModelConfigDict
from qualia_core.utils.logger import CSVLogger
from torch import nn

from qualia_plugin_snn.learningframework.SpikingJelly import SpikingJelly
from qualia_plugin_snn.learningmodel.pytorch.SNN import SNN

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from qualia_codegen_core.graph import ModelGraph  # noqa: TCH002
    from qualia_codegen_core.graph.layers import TBaseLayer, TConvLayer, TDenseLayer  # noqa: TCH002
    from qualia_core.datamodel.RawDataModel import RawData  # noqa: TCH002
    from qualia_core.qualia import TrainResult  # noqa: TCH002
    from torch.types import Number  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class EnergyEstimationMetricLoggerFields(NamedTuple):
    """Interface object for CSV logging.

    Should contain the same fields as :class:`EnergyMetrics` and returned by :meth:`EnergyMetrics.asnamedtuple`.

    :param name: Layer name
    :param mem_pot: Energy for potential memory read/write
    :param mem_weights: Energy for weights memory read
    :param mem_bias: Energy for bias memory read
    :param mem_io: Energy for input/output memory read/write
    :param ops: Energy for synaptic operations
    :param addr: Energy for event address computation
    :param input_spikerate: Average input spike rate per timestep
    :param output_spikerate: Average output spike rate per timestep
    :param input_is_binary: If input tensor only contains binary values, i.e., spikes
    :param output_is_binary: If output tensor only contains binary values, i.e., spikes
    """

    #: Layer name
    name: str
    #: Energy for potential memory read/write
    mem_pot: float
    #: Energy for weights memory read
    mem_weights:float
    #: Energy for bias memory read
    mem_bias: float
    #: Energy for input/output memory read/write
    mem_io: float
    #: Energy for potential, weights, bias and input/output read/write, i.e., Erdram + Ewrram
    mem_total: float
    #: Energy for synaptic operations
    ops: float
    #: Energy for event address computation
    addr: float
    #: Energy for synaptic operations and event address computation, i.e., Eops+addr
    opsaddr: float
    #: Total energy, i.e., energy for memory read/write, synaptic operations and event adress computation
    total: float
    #: Average input spike rate per timestep
    input_spikerate: float
    #: Average output spike rate per timestep
    output_spikerate: float
    #: If input tensor only contains binary values, i.e., spikes
    input_is_binary: bool
    #: If output tensor only contains binary values, i.e., spikes
    output_is_binary: bool

@dataclass
class EnergyMetrics:
    """Holds the computed average energy per inference for each layer.

    :param name: Layer name
    :param mem_pot: Energy for potential memory read/write
    :param mem_weights: Energy for weights memory read
    :param mem_bias: Energy for bias memory read
    :param mem_io: Energy for input/output memory read/write
    :param ops: Energy for synaptic operations
    :param addr: Energy for event address computation
    :param input_spikerate: Average input spike rate per timestep
    :param output_spikerate: Average output spike rate per timestep
    :param input_is_binary: If input tensor only contains binary values, i.e., spikes
    :param output_is_binary: If output tensor only contains binary values, i.e., spikes
    """

    #: Layer name
    name: str
    #: Energy for potential memory read/write
    mem_pot: float
    #: Energy for weights memory read
    mem_weights: float
    #: Energy for bias memory read
    mem_bias: float
    #: Energy for input/output memory read/write
    mem_io: float
    #: Energy for synaptic operations
    ops: float
    #: Energy for event address computation
    addr: float
    #: Average input spike rate per timestep
    input_spikerate: float | None
    #: Average output spike rate per timestep
    output_spikerate: float | None
    #: If input tensor only contains binary values, i.e., spikes
    input_is_binary: bool
    #: If output tensor only contains binary values, i.e., spikes
    output_is_binary: bool

    @property
    def mem_total(self) -> float:
        """Energy for potential, weights, bias and input/output read/write. Erdram + Ewrram."""
        return self.mem_pot + self.mem_weights + self.mem_bias + self.mem_io

    @property
    def opsaddr(self) -> float:
        """Energy for synaptic operations and event address computation. Eops+addr."""
        return self.ops + self.addr

    @property
    def total(self) -> float:
        """Total energy. Energy for memory read/write, synaptic operations and event adress computation."""
        return self.mem_total + self.opsaddr

    def asnamedtuple(self) -> EnergyEstimationMetricLoggerFields:
        """Return the data from this class as a NamedTuple for use with the CSV logger.

        Instanciate a :class:`EnergyEstimationMetricLoggerFields` object with
        all of this class fields and properties and return it.

        :return: the :class:`EnergyEstimationMetricLoggerFields` with all data from this object copied into it
        """
        return EnergyEstimationMetricLoggerFields(**dataclasses.asdict(self),
                                                  mem_total=self.mem_total,
                                                  opsaddr=self.opsaddr,
                                                  total=self.total)

class EnergyEstimationMetric(PostProcessing[nn.Module]):
    r"""Analytical energy estimation metric.

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

    _m_e_add: float = 0.1
    """Energy for add operation in pJ, 45nm int32.

    Default from `Computing’s Energy Problem (and what we can do about it) <https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf>`_,
    Mark Horowitz, ISSCC 2014.

    :meta public:
    """  # noqa: RUF001

    _m_e_mul: float = 3.1
    """Energy for mul operation in pJ, 45nm int32.

    Default from `Computing’s Energy Problem (and what we can do about it) <https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf>`_,
    Mark Horowitz, ISSCC 2014.

    :meta public:
    """  # noqa: RUF001

    #: CSV logger to record metrics for each layer in a file inside the `logs/<bench.name>/EnergyEstimationMetric` directory
    csvlogger: CSVLogger[EnergyEstimationMetricLoggerFields]

    def __init__(self,
                 mem_width: int,
                 fifo_size: int = 0,
                 total_spikerate_exclude_nonbinary: bool = True) -> None:  # noqa: FBT001, FBT002
        """Construct :class:`qualia_plugin_snn.postprocessing.EnergyEstimationMetric.EnergyEstimationMetric`.

        :param mem_width: Memory access size in bits, e.g. 16 for 16-bit quantization
        :param fifo_size: Size of the input/output FIFOs for each layer in SPLEAT
        :param total_spikerate_exclude_nonbinary: If True, exclude non-binary inputs/outputs from total spikerate computation
        """
        super().__init__()
        self._mem_width = mem_width
        self._fifo_size = fifo_size
        self._total_spikerate_exclude_nonbinary = total_spikerate_exclude_nonbinary

        self.csvlogger = CSVLogger(name='EnergyEstimationMetric')
        self.csvlogger.fields = EnergyEstimationMetricLoggerFields

    #######
    # FNN #
    #######

    def _rdin_conv_fnn(self, layer: TConvLayer) -> int:
        """Count number of read operations for the input of a convolutional layer in a formal neural network.

        Eq. 3: Cin × Cout × Hout × Wout × Wkernel × Hkernel.

        :meta public:
        :param layer: A convolutional layer
        :return: Number of read operations for the input of a convolutional layer in a formal neural network
        """  # noqa: RUF002
        return layer.input_shape[0][-1] * math.prod(layer.output_shape[0][1:]) * math.prod(layer.kernel_size)

    def _e_rdin_conv_fnn(self, layer: TConvLayer, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for the input of a convolution layer in a formal neural network.

        :meta public:
        :param layer: A convolutional layer
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the input of a convolution layer in a formal neural network
        """
        return self._rdin_conv_fnn(layer) * e_rdram(math.prod(layer.input_shape[0][1:]))


    def _rdin_fc_fnn(self, layer: TDenseLayer) -> int:
        """Count number of read operations for the input of a fully-connected layer in a formal neural network.

        Eq. 4: Nin.

        :meta public:
        :param layer: A fully-connected layer
        :return: Number of read operations for the input of a fully-connected layer in a formal neural network
        """
        return layer.input_shape[0][-1]

    def _e_rdin_fc_fnn(self, layer: TDenseLayer, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for the input of a fully-connected layer in a formal neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the input of a fully-connected layer in a formal neural network
        """
        return self._rdin_fc_fnn(layer) * e_rdram(math.prod(layer.input_shape[0][1:]))


    def _rdweights_conv_fnn(self, layer: TConvLayer) -> int:
        """Count number of read operations for the weights of a convolutional layer (excluding bias) in a formal neural network.

        Eq. 6.1f: Cin × Wkernel × Hkernel × Cout × Wout × Hout.

        :meta public:
        :param layer: A convolutional layer
        :return: Number of read operations for the weights of a convolutional layer (excluding bias) in a formal neural network
        """  # noqa: RUF002
        return layer.input_shape[0][-1] * math.prod(layer.kernel_size) * math.prod(layer.output_shape[0][1:])

    def _e_rdweights_conv_fnn(self, layer: TConvLayer, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for the weights of a convolutional layer (excluding bias) in a formal neural network.

        :meta public:
        :param layer: A convolutional layer
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the weights of a convolutional layer (excluding bias) in a formal neural network
        """
        return self._rdweights_conv_fnn(layer) * e_rdram(layer.kernel.size)

    def _rdbias_conv_fnn(self, layer: TConvLayer) -> int:
        """Count number of read operations for the biases of a convolutional layer in a formal neural network.

        Eq. 6.1s: Cout × Wout × Hout.

        :meta public:
        :param layer: A convolutional layer
        :return: Number of read operations for the biases of a convolutional layer in a formal neural network
        """  # noqa: RUF002
        return math.prod(layer.output_shape[0][1:])

    def _e_rdbias_conv_fnn(self, layer: TConvLayer, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for the biases of a convolutional layer in a formal neural network.

        :meta public:
        :param layer: A convolutional layer
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the biases of a convolutional layer in a formal neural network or 0 if the layer
                 does not use biases
        """
        if layer.use_bias:
            return self._rdbias_conv_fnn(layer) * e_rdram(layer.bias.size)
        return 0


    def _rdweights_fc_fnn(self, layer: TDenseLayer) -> int:
        """Count number of read operations for the weights of a fully-connected layer (excluding bias) in a formal neural network.

        Eq. 6.2f: Nin × Nout.

        :meta public:
        :param layer: A fully-connected layer
        :return: Number of read operations for the weights of a fully-connected layer (excluding bias) in a formal neural network
        """  # noqa: RUF002
        return layer.input_shape[0][-1] * layer.output_shape[0][-1]

    def _e_rdweights_fc_fnn(self, layer: TDenseLayer, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for weights of a fully-connected layer (excluding bias) in a formal neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the weights of a fully-connected layer (excluding bias) in a formal neural network
        """
        return self._rdweights_fc_fnn(layer) * e_rdram(layer.kernel.size)

    def _rdbias_fc_fnn(self, layer: TDenseLayer) -> int:
        """Count number of read operations for the biases of a fully-connected layer in a formal neural network.

        Eq. 6.2s: Nout.

        :meta public:
        :param layer: A fully-connected layer
        :return: Number of read operations for the biases of a fully-connected layer in a formal neural network
        """
        return layer.output_shape[0][-1]

    def _e_rdbias_fc_fnn(self, layer: TDenseLayer, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for the biases of a fully-connected layer in a formal neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the biases of a fully-connected layer in a formal neural network or 0 if the layer
                 does not use biases
        """
        if layer.use_bias:
            return self._rdbias_fc_fnn(layer) * e_rdram(layer.bias.size)
        return 0


    def _wrout_conv_fnn(self, layer: TConvLayer) -> int:
        """Count number of write operations for the output of a convolutional layer in a formal neural network.

        Eq. 11: Cout × Hout × Wout.

        :meta public:
        :param layer: A convolutional layer
        :return: Number of write operations for the output of a convolutional layer in a formal neural network
        """  # noqa: RUF002
        return math.prod(layer.output_shape[0][1:])

    def _e_wrout_conv_fnn(self, layer: TConvLayer, e_wrram: Callable[[int], float]) -> float:
        """Compute energy for write operations for the output of a convolutional layer in a formal neural network.

        :meta public:
        :param layer: A convolutional layer
        :param e_rdram: Function to compute memory write access energy for a given memory size
        :return: Energy for write operations for the output of a convolutional layer in a formal neural network
        """
        return self._wrout_conv_fnn(layer) * e_wrram(math.prod(layer.output_shape[0][1:]))


    def _wrout_fc_fnn(self, layer: TDenseLayer) -> int:
        """Count number of write operations for the output of a fully-connected layer in a formal neural network.

        Eq. 12: Nout.

        :meta public:
        :param layer: A fully-connected layer
        :return: Number of write operations for the output of a fully-connected layer in a formal neural network
        """
        return layer.output_shape[0][-1]

    def _e_wrout_fc_fnn(self, layer: TDenseLayer, e_wrram: Callable[[int], float]) -> float:
        """Compute energy for write operations for the output of a fully-connected layer in a formal neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param e_rdram: Function to compute memory write access energy for a given memory size
        :return: Energy for write operations for the output of a fully-connected layer in a formal neural network
        """
        return self._wrout_fc_fnn(layer) * e_wrram(math.prod(layer.output_shape[0][1:]))


    def _mac_ops_conv_fnn(self, layer: TConvLayer) -> int:
        """Count number of multiply-accumulate operations inside a convolutional layer in a formal neural network.

        Eq. 1.1: Cout × Hout × Wout × Cin × Hkernel × Wkernel.

        :meta public:
        :param layer: A convolutional layer
        :return: Number of multiply-accumulate operations inside a convolutional layer in a formal neural network.
        """  # noqa: RUF002
        return math.prod(layer.output_shape[0][1:]) * layer.input_shape[0][-1] * math.prod(layer.kernel_size)

    def _acc_ops_conv_fnn(self, layer: TConvLayer) -> int:
        """Count number of accumulate operations inside a convolutional layer in a formal neural network.

        Eq. 1.2: Cout × Hout × Wout.

        :meta public
        :param layer: A convolutional layer
        :return: Number of accumulate operations inside a convolutional layer in a formal neural network.
        """  # noqa: RUF002
        return math.prod(layer.output_shape[0][1:])

    def _e_ops_conv_fnn(self, layer: TConvLayer) -> float:
        """Compute energy for multiply-accumulate and accumulate operations inside convolutional layer in a formal neural network.

        :meta public:
        :param layer: A convolutional layer
        :return: Energy for multiply-accumulate and accumulate operations inside a convolutional layer in a formal neural network.
        """
        return self._mac_ops_conv_fnn(layer) * (self._e_mul + self._e_add) + self._acc_ops_conv_fnn(layer) * self._e_add


    def _mac_ops_fc_fnn(self, layer: TDenseLayer) -> int:
        """Count number of multiply-accumulate operations inside a fully-connected layer in a formal neural network.

        Eq. 2.1: Nin × Nout.

        :meta public:
        :param layer: A fully-connected layer
        :return: Number of multiply-accumulate operations inside a fully-connected layer in a formal neural network.
        """  # noqa: RUF002
        return layer.input_shape[0][-1] * layer.output_shape[0][-1]

    def _acc_ops_fc_fnn(self, layer: TDenseLayer) -> int:
        """Count number of accumulate operations inside a fully-connected layer in a formal neural network.

        Eq. 2.2: Nout.

        :meta public:
        :param layer: A fully-connected layer
        :return: Number of accumulate operations inside a fully-connected layer in a formal neural network.
        """
        return layer.output_shape[0][-1]

    def _e_ops_fc_fnn(self, layer: TDenseLayer) -> float:
        """Compute energy for multiply-accumulate and accumulate operations inside fully-connected layer in formal neural network.

        :meta public:
        :param layer: A fully-connected layer
        :return: Energy for multiply-accumulate and accumulate operations inside a fully-connected layer in formal neural network.
        """
        return self._mac_ops_fc_fnn(layer) * (self._e_mul + self._e_add) + self._acc_ops_fc_fnn(layer) * self._e_add


    def _mac_addr_conv_fnn(self, _: TConvLayer) -> int:
        """Count number of multiply-accumulate operations for addressing a convolutional layer in a formal neural network.

        0

        :meta public:
        :return: 0
        """
        return 0

    def _acc_addr_conv_fnn(self, layer: TConvLayer) -> int:
        """Count number of accumulate operations for addressing a convolutional layer in a formal neural network.

        Cin × Hin × Win + Cout × Hout × Wout + Cout × Hkernel × Wkernel.

        :meta public:
        :param layer: A convolutional layer
        :return: Number of accumulate operations for addressing a convolutional layer in a formal neural network
        """  # noqa: RUF002
        return (math.prod(layer.input_shape[0][1:])
                + math.prod(layer.output_shape[0][1:])
                + layer.output_shape[0][-1] * math.prod(layer.kernel_size))

    def _e_addr_conv_fnn(self, layer: TConvLayer) -> float:
        """Compute energy for addressing a convolutional layer in a formal neural network.

        :meta public:
        :param layer: A convolutional layer
        :return: Energy for addressing a convolutional layer in a formal neural network.
        """
        return self._mac_addr_conv_fnn(layer) * (self._e_mul + self._e_add) + self._acc_addr_conv_fnn(layer) * self._e_add


    def _mac_addr_fc_fnn(self, _: TDenseLayer) -> int:
        """Count number of multiply-accumulate operations for addressing a fully-connected layer in a formal neural network.

        0

        :meta public:
        :return: 0
        """
        return 0

    def _acc_addr_fc_fnn(self, layer: TDenseLayer) -> int:
        """Count number of accumulate operations for addressing a fully-connected layer in a formal neural network.

        Nin × Nout.

        :meta public:
        :param layer: A fully-connected layer
        :return: Number of accumulate operations for addressing a fully-connected layer in a formal neural network
        """  # noqa: RUF002
        return layer.input_shape[0][-1] * layer.output_shape[0][-1]

    def _e_addr_fc_fnn(self, layer: TDenseLayer) -> float:
        """Compute energy for addressing a fully-connected layer in a formal neural network.

        :meta public:
        :param layer: A fully-connected layer
        :return: Energy for addressing a fully-connected layer in a formal neural network.
        """
        return self._mac_addr_fc_fnn(layer) * (self._e_mul + self._e_add) + self._acc_addr_fc_fnn(layer) * self._e_add

    def _compute_model_energy_fnn(self,
                                  modelgraph: ModelGraph,
                                  e_rdram: Callable[[int], float],
                                  e_wrram: Callable[[int], float]) -> list[EnergyMetrics]:
        """Compute the energy per inference for each layer of a formal neural network.

        Supports the following layers:

        * :class:`qualia_codegen_core.graph.layers.TConvLayer.TConvLayer`
        * :class:`qualia_codegen_core.graph.layers.TDenseLayer.TDenseLayer`

        :meta public:
        :param modelgraph: Model to computer energy on
        :param e_rdram: Function to compute memory read energy for a given memory size
        :param e_wrram: Function to computer memory write energy for a given memory size
        :return: A list of EnergyMetrics for each layer and a total with fields populated with energy estimation
        """
        from qualia_codegen_core.graph.layers import TConvLayer, TDenseLayer

        ems: list[EnergyMetrics] = []
        for node in modelgraph.nodes:
            if isinstance(node.layer, TConvLayer):
                em = EnergyMetrics(name=node.layer.name,
                                   mem_pot=0,
                                   mem_weights=self._e_rdweights_conv_fnn(node.layer, e_rdram),
                                   mem_bias=self._e_rdbias_conv_fnn(node.layer, e_rdram),
                                   mem_io=self._e_rdin_conv_fnn(node.layer, e_rdram) + self._e_wrout_conv_fnn(node.layer, e_wrram),
                                   ops=self._e_ops_conv_fnn(node.layer),
                                   addr=self._e_addr_conv_fnn(node.layer),
                                   input_spikerate=None,
                                   output_spikerate=None,
                                   input_is_binary=False,
                                   output_is_binary=False)
            elif isinstance(node.layer, TDenseLayer):
                em = EnergyMetrics(name=node.layer.name,
                                   mem_pot=0,
                                   mem_weights=self._e_rdweights_fc_fnn(node.layer, e_rdram),
                                   mem_bias=self._e_rdbias_fc_fnn(node.layer, e_rdram),
                                   mem_io=self._e_rdin_fc_fnn(node.layer, e_rdram) + self._e_wrout_fc_fnn(node.layer, e_wrram),
                                   ops=self._e_ops_fc_fnn(node.layer),
                                   addr=self._e_addr_fc_fnn(node.layer),
                                   input_spikerate=None,
                                   output_spikerate=None,
                                   input_is_binary=False,
                                   output_is_binary=False)
            else:
                logger.warning('%s skipped, result may be inaccurate', node.layer.name)
                continue

            ems.append(em)

        em_total = EnergyMetrics(name='Total',
                                 mem_pot=sum(em.mem_pot for em in ems),
                                 mem_weights=sum(em.mem_weights for em in ems),
                                 mem_bias=sum(em.mem_bias for em in ems),
                                 mem_io=sum(em.mem_io for em in ems),
                                 ops=sum(em.ops for em in ems),
                                 addr=sum(em.addr for em in ems),
                                 input_spikerate=None,
                                 output_spikerate=None,
                                 input_is_binary=False,
                                 output_is_binary=False)
        ems.append(em_total)

        return ems


    #######
    # SNN #
    #######

    def _rdin_snn(self, layer: TBaseLayer, input_spikerate: float) -> float:
        """Count average number of read operations for the input of a layer in a spiking neural network.

        Eq. 5: θl-1.

        :meta public:
        :param layer: A convolutional or fully-connected layer
        :param input_spikerate: Average spike per input per inference
        :return: Average number of read operations for the input of a layer in a spiking neural network
        """
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return theta_in  # noqa: RET504

    def _e_rdin_snn(self, layer: TBaseLayer, input_spikerate: float, e_rdram: Callable[[int], float]) -> float:
        """Compute average energy for read operations for the input of a layer in a spiking neural network.

        :meta public:
        :param layer: A convolutional or fully-connected layer
        :param input_spikerate: Average spike per input per inference
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Average energy for read operations for the input of a layer in a spiking neural network
        """
        return self._rdin_snn(layer, input_spikerate) * e_rdram(self._fifo_size)


    def _rdweights_conv_snn(self, layer: TConvLayer, input_spikerate: float) -> float:
        """Count average number of read operations for weights of convolutional layer (excluding bias) in spiking neural network.

        Eq. 7f: θl-1 × Cout × Wkernel × Hkernel.

        :meta public:
        :param layer: A convolutional layer
        :param input_spikerate: Average spike per input per inference
        :return: Average number of read operations for weights of convolutional layer (excluding bias) in spiking neural network
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return  theta_in * layer.output_shape[0][-1] * math.prod(layer.kernel_size)

    def _e_rdweights_conv_snn(self, layer: TConvLayer, input_spikerate: float, e_rdram: Callable[[int], float]) -> float:
        """Compute average energy for read operations for convolutional layer weights (excluding bias) in spiking neural network.

        :meta public:
        :param layer: A convolutional layer
        :param input_spikerate: Average spike per input per inference
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Average energy for read operations for weights of convolutional layer (excluding bias) in spiking neural network
        """
        return self._rdweights_conv_snn(layer, input_spikerate) * e_rdram(layer.kernel.size)

    def _rdbias_conv_snn(self, layer: TConvLayer) -> int:
        """Count number of read operations for the biases of a convolutional layer in a spiking neural network.

        Eq. 7s: Cout × Wout × Hout.

        :meta public:
        :param layer: A convolutional layer
        :return: Number of read operations for the biases of a convolutional layer in a spiking neural network
        """  # noqa: RUF002
        return math.prod(layer.output_shape[0][1:])

    def _e_rdbias_conv_snn(self, layer: TConvLayer, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for the biases of a convolutional layer in a spiking neural network.

        :meta public:
        :param layer: A convolutional layer
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the biases of a convolutional layer in a spiking neural network or 0 if the layer
                 does not use biases
        """
        if layer.use_bias:
            return self._rdbias_conv_snn(layer) * e_rdram(layer.bias.size)
        return 0


    def _rdweights_fc_snn(self, layer: TDenseLayer, input_spikerate: float) -> float:
        """Count average number of read operations for weights of fully-connected layer (excluding bias) in spiking neural network.

        Eq. 8f: θl-1 × Nout.

        :meta public:
        :param layer: A fully-connected layer
        :param input_spikerate: Average spike per input per inference
        :return: Average number of read operations for weights of fully-connected layer (excluding bias) in spiking neural network
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return theta_in * layer.output_shape[0][-1]

    def _e_rdweights_fc_snn(self, layer: TDenseLayer, input_spikerate: float, e_rdram: Callable[[int], float]) -> float:
        """Compute average energy for weights read operations of fully-connected layer (excluding bias) in spiking neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param input_spikerate: Average spike per input per inference
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Average energy for read operations for weights of fully-connected layer (excluding bias) in spiking neural network
        """
        return self._rdweights_fc_snn(layer, input_spikerate) * e_rdram(layer.kernel.size)

    def _rdbias_fc_snn(self, layer: TDenseLayer) -> int:
        """Count number of read operations for the biases of a fully-connected layer in a spiking neural network.

        Eq. 6.2s: Nout.

        :meta public:
        :param layer: A fully-connected layer
        :return: Number of read operations for the biases of a fully-connected layer in a spiking neural network
        """
        return layer.output_shape[0][-1]

    def _e_rdbias_fc_snn(self, layer: TDenseLayer, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for the biases of a fully-connected layer in a spiking neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the biases of a fully-connected layer in a spiking neural network or 0 if the layer
            does not use biases
        """
        if layer.use_bias:
            return self._rdbias_fc_snn(layer) * e_rdram(layer.bias.size)
        return 0


    def _wrout_snn(self, layer: TBaseLayer, output_spikerate: float) -> float:
        """Count average number of write operations for the output of a layer in a spiking neural network.

        Eq. 13: Noutput.

        :meta public:
        :param layer: A convolutional or fully-connected layer
        :param output_spikerate: Average spike per output per inference
        :return: Average number of write operations for the output of a layer in a spiking neural network
        """
        theta_out = output_spikerate * math.prod(layer.output_shape[0][1:])
        return theta_out  # noqa: RET504

    def _e_wrout_snn(self, layer: TBaseLayer, output_spikerate: float, e_wrram: Callable[[int], float]) -> float:
        """Compute average energy for write operations for the output of a layer in a spiking neural network.

        :meta public:
        :param layer: A convolutional or fully-connected layer
        :param output_spikerate: Average spike per output per inference
        :param e_rdram: Function to compute memory write access energy for a given memory size
        :return: Average energy for write operations for the output of a layer in a spiking neural network
        """
        return self._wrout_snn(layer, output_spikerate) * e_wrram(self._fifo_size)


    def _wrpot_conv_snn(self, layer: TConvLayer, input_spikerate: float) -> float:
        """Count average number of write operations for the potentials of a convolutional layer in a spiking neural network.

        Eq 14: θl-1 × Cout × Wkernel × Hkernel + Cout × Hout × Wout.

        :meta public:
        :param layer: A convolutional layer
        :param input_spikerate: Average spike per input per inference
        :return: Average number of write operations for the potentials of a convolutional layer in a spiking neural network
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return theta_in * layer.output_shape[0][-1] * math.prod(layer.kernel_size) + math.prod(layer.output_shape[0][1:])

    def _e_wrpot_conv_snn(self, layer: TConvLayer, input_spikerate: float, e_wrram: Callable[[int], float]) -> float:
        """Compute average energy for write operations for the potentials of a convolutional layer in a spiking neural network.

        :meta public:
        :param layer: A convolutional layer
        :param input_spikerate: Average spike per input per inference
        :param e_rdram: Function to compute memory write access energy for a given memory size
        :return: Average energy for write operations for the potentials of a convolutional layer in a spiking neural network
        """
        return self._wrpot_conv_snn(layer, input_spikerate) * e_wrram(math.prod(layer.output_shape[0][1:]))

    def _wrpot_fc_snn(self, layer: TDenseLayer, input_spikerate: float) -> float:
        """Count average number of write operations for the potentials of a fully-connected layer in a spiking neural network.

        Eq 15: θl-1 × Nout + Nout.

        :meta public:
        :param layer: A fully-connected layer
        :param input_spikerate: Average spike per input per inference
        :return: Average number of write operations for the potentials of a fully-connected layer in a spiking neural network
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return theta_in * layer.output_shape[0][-1] + layer.output_shape[0][-1]

    def _e_wrpot_fc_snn(self, layer: TDenseLayer, input_spikerate: float, e_wrram: Callable[[int], float]) -> float:
        """Compute average energy for write operations for the potentials of a fully-connected layer in a spiking neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param input_spikerate: Average spike per input per inference
        :param e_rdram: Function to compute memory write access energy for a given memory size
        :return: Average energy for write operations for the potentials of a fully-connected layer in a spiking neural network
        """
        return self._wrpot_fc_snn(layer, input_spikerate) * e_wrram(math.prod(layer.output_shape[0][1:]))


    def _mac_ops_conv_snn(self, layer: TConvLayer, timesteps: int, leak: bool) -> int:  # noqa: FBT001
        """Count number of multiply-accumulate operations inside a convolutional layer in a spiking neural network.

        Eq. 1.3: T × Cout × Hout × Wout.

        :meta public:
        :param layer: A convolutional layer
        :param timesteps: Number of timesteps
        :param leak: Whether the neuron has leak (LIF) or not (IF)
        :return: Number of multiply-accumulate operations inside a convolutional layer in a spiking neural network, 0 if no leak
        """  # noqa: RUF002
        if not leak:
            return 0
        return timesteps * math.prod(layer.output_shape[0][1:])

    def _acc_ops_conv_snn(self, layer: TConvLayer, input_spikerate: float, output_spikerate: float, timesteps: int) -> float:
        """Count average number of accumulate operations inside a convolutional layer in a spiking neural network.

        Eq. 1.4: θl-1 × ceil(Hkernel / S) × ceil(Wkernel / S) × Cout + T × Cout × Hout × Wout + θl.

        :meta public:
        :param layer: A convolutional layer
        :param input_spikerate: Average spike per input per inference
        :param output_spikerate: Average spike per output per inference
        :param timesteps: Number of timesteps
        :return: Average number of accumulate operations inside a convolutional layer in a spiking neural network.
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        theta_out = output_spikerate * math.prod(layer.output_shape[0][1:])
        return (theta_in
                * math.prod(math.ceil(k / s) for k, s in zip(layer.kernel_size, layer.strides))
                * layer.output_shape[0][-1]
                + timesteps * math.prod(layer.output_shape[0][1:])
                + theta_out)

    def _e_ops_conv_snn(self,  # noqa: PLR0913
                      layer: TConvLayer,
                      input_spikerate: float,
                      output_spikerate: float,
                      timesteps: int,
                      leak: bool) -> float:  # noqa: FBT001
        """Compute average energy for multiply-accumulate and accumulate inside a convolutional layer in spiking neural network.

        :meta public:
        :param layer: A convolutional layer
        :param input_spikerate: Average spike per input per inference
        :param output_spikerate: Average spike per output per inference
        :param timesteps: Number of timesteps
        :param leak: Whether the neuron has leak (LIF) or not (IF)
        :return: Average energy for multiply-accumulate and accumulate inside a convolutional layer in spiking neural network.
        """
        return (self._mac_ops_conv_snn(layer, timesteps, leak) * (self._e_mul + self._e_add)
                + self._acc_ops_conv_snn(layer, input_spikerate, output_spikerate, timesteps) * self._e_add)


    def _mac_ops_fc_snn(self, layer: TDenseLayer, timesteps: int, leak: bool) -> int:  # noqa: FBT001
        """Count number of multiply-accumulate operations inside a fully-connected layer in a spiking neural network.

        Eq. 2.3: T × Nout.

        :meta public:
        :param layer: A fully-connected layer
        :param timesteps: Number of timesteps
        :param leak: Whether the neuron has leak (LIF) or not (IF)
        :return: Number of multiply-accumulate operations inside a fully-connected layer in a spiking neural network, 0 if no leak
        """  # noqa: RUF002
        if not leak:
            return 0
        return timesteps * layer.output_shape[0][-1]

    def _acc_ops_fc_snn(self, layer: TDenseLayer, input_spikerate: float, output_spikerate: float, timesteps: int) -> float:
        """Count average number of accumulate operations inside a fully-connected layer in a spiking neural network.

        Eq. 2.4: θl-1 × Nout + T × Nout + θl.

        **Warning 1**: the article counts Nin times too many as 1 input spike triggers only Nout MAC (each output neuron).

        **Warning 2**: the article is missing the third term which corresponds to the reset.

        Original equation (not used here):
        Eq. 2.4: θl-1 × Nin × Nout + T × Nout.

        :meta public:
        :param layer: A fully-connected layer
        :param input_spikerate: Average spike per input per inference
        :param output_spikerate: Average spike per output per inference
        :param timesteps: Number of timesteps
        :return: Average number of accumulate operations inside a fully-connected layer in a spiking neural network.
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return (theta_in * layer.output_shape[0][-1]
                + timesteps * layer.output_shape[0][-1]
                + output_spikerate * math.prod(layer.output_shape[0][1:]))

    def _e_ops_fc_snn(self,  # noqa: PLR0913
                      layer: TDenseLayer,
                      input_spikerate: float,
                      output_spikerate: float,
                      timesteps: int,
                      leak: bool) -> float:  # noqa: FBT001
        """Compute average energy for multiply-accumulate and accumulate inside fully-connected layer in spiking neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param input_spikerate: Average spike per input per inference
        :param output_spikerate: Average spike per output per inference
        :param timesteps: Number of timesteps
        :param leak: Whether the neuron has leak (LIF) or not (IF)
        :return: Average energy for multiply-accumulate and accumulate inside a fully-connected layer in spiking neural network.
        """
        return (self._mac_ops_fc_snn(layer, timesteps, leak) * (self._e_mul + self._e_add)
                + self._acc_ops_fc_snn(layer, input_spikerate, output_spikerate, timesteps) * self._e_add)


    def _mac_addr_conv_snn(self, layer: TConvLayer, input_spikerate: float) -> float:
        """Count average number of multiply-accumulate operations for addressing a convolutional layer in a spiking neural network.

        Eq. 16.2: θl-1 × 2.

        :meta public:
        :param layer: A convolutional layer
        :param input_spikerate: Average spike per input per inference
        :return: average number of multiply-accumulate operations for addressing a convolutional layer in a spiking neural network.
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return theta_in * 2

    def _acc_addr_conv_snn(self, layer: TConvLayer, input_spikerate: float) -> float:
        """Count average number of accumulate operations for addressing a convolutional layer in a spiking neural network.

        Eq 16.3: θl-1 × Cout × Hkernel × Wkernel.

        :meta public:
        :param layer: A convolutional layer
        :param input_spikerate: Average spike per input per inference
        :return: Average number of accumulate operations for addressing a convolutional layer in a spiking neural network
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return theta_in * math.prod(layer.kernel_size)

    def _e_addr_conv_snn(self, layer: TConvLayer, input_spikerate: float) -> float:
        """Compute average energy for addressing a convolutional layer in a spiking neural network.

        :meta public:
        :param layer: A convolutional layer
        :param input_spikerate: Average spike per input per inference
        :return: Average energy for addressing a convolutional layer in a spiking neural network.
        """
        return (self._mac_addr_conv_snn(layer, input_spikerate) * (self._e_mul + self._e_add)
                + self._acc_addr_conv_snn(layer, input_spikerate) * self._e_add)


    def _mac_addr_fc_snn(self, _: TDenseLayer) -> int:
        """Count number of multiply-accumulate operations for addressing a fully-connected layer in a spiking neural network.

        0

        :meta public:
        :return: 0
        """
        return 0

    def _acc_addr_fc_snn(self, layer: TDenseLayer, input_spikerate: float) -> float:
        """Count average number of accumulate operations for addressing a fully-connected layer in a spiking neural network.

        Eq. 17.2: θl-1 × Nout.

        :meta public:
        :param layer: A fully-connected layer
        :param input_spikerate: Average spike per input per inference
        :return: Average number of accumulate operations for addressing a fully-connected layer in a spiking neural network
        """  # noqa: RUF002
        return input_spikerate * layer.output_shape[0][-1]

    def _e_addr_fc_snn(self, layer: TDenseLayer, input_spikerate: float) -> float:
        """Compute average energy for addressing a fully-connected layer in a spiking neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param input_spikerate: Average spike per input per inference
        :return: Average energy for addressing a fully-connected layer in a spiking neural network.
        """
        return (self._mac_addr_fc_snn(layer) * (self._e_mul + self._e_add)
                + self._acc_addr_fc_snn(layer, input_spikerate) * self._e_add)



    def _compute_model_energy_snn(self,  # noqa: PLR0913
                                  modelgraph: ModelGraph,
                                  input_spikerates: dict[str, float],
                                  output_spikerates: dict[str, float],
                                  input_is_binary: dict[str, bool],
                                  output_is_binary: dict[str, bool],
                                  timesteps: int,
                                  e_rdram: Callable[[int], float],
                                  e_wrram: Callable[[int], float]) -> list[EnergyMetrics]:
        """Compute the energy per inference for each layer of a spiking neural network.

        Supports the following layers:

        * :class:`qualia_codegen_core.graph.layers.TConvLayer.TConvLayer`
        * :class:`qualia_codegen_core.graph.layers.TDenseLayer.TDenseLayer`

        Input spike rates are per-timestep, this function multiplies by the number of timesteps to get the spike rates per infernce
        which are used by the operation count and energy computation functions.

        :meta public:
        :param modelgraph: Model to computer energy on
        :param input_spikerate: Dict of layer names and average spike per input per timestep for the layer
        :param output_spikerate: Dict of layer names and average spike per output per timestep for the layer
        :param input_is_binary: Dict of layer names and whether its input is binary (spike) or not
        :param output_is_binary: Dict of layer names and whether its output is binary (spike) or not
        :param timesteps: Number of timesteps
        :param e_rdram: Function to compute memory read energy for a given memory size
        :param e_wrram: Function to computer memory write energy for a given memory size
        :return: A list of EnergyMetrics for each layer and a total with fields populated with energy estimation
        """
        from qualia_codegen_core.graph.layers import TConvLayer, TDenseLayer
        from qualia_codegen_plugin_snn.graph.layers import TLifLayer

        ems: list[EnergyMetrics] = []
        for node in modelgraph.nodes:
            if not isinstance(node.layer, (TConvLayer, TDenseLayer)):
                logger.warning('%s skipped, result may be inaccurate', node.layer.name)
                continue

            # Account for timesteps here since the spikerate has been averaged over timesteps
            input_spikerate = input_spikerates[node.layer.name] * timesteps
            output_spikerate = output_spikerates[node.layer.name] * timesteps

            leak = len(node.outnodes) > 0 and isinstance(node.outnodes[0].layer, TLifLayer)

            if isinstance(node.layer, TConvLayer):
                em = EnergyMetrics(name=node.layer.name,
                                   mem_pot=self._e_wrpot_conv_snn(node.layer, input_spikerate, e_wrram),
                                   mem_weights=self._e_rdweights_conv_snn(node.layer, input_spikerate, e_rdram),
                                   mem_bias=self._e_rdbias_conv_snn(node.layer, e_rdram),
                                   mem_io=self._e_rdin_snn(node.layer, input_spikerate, e_rdram)
                                   + self._e_wrout_snn(node.layer, output_spikerate, e_wrram),
                                   ops=self._e_ops_conv_snn(node.layer, input_spikerate, output_spikerate, timesteps, leak),
                                   addr=self._e_addr_conv_snn(node.layer, input_spikerate),
                                   input_spikerate=input_spikerates[node.layer.name],
                                   output_spikerate=output_spikerates[node.layer.name],
                                   input_is_binary=input_is_binary[node.layer.name],
                                   output_is_binary=output_is_binary[node.layer.name],
                                   )
                ems.append(em)
            elif isinstance(node.layer, TDenseLayer):
                em = EnergyMetrics(name=node.layer.name,
                                   mem_pot=self._e_wrpot_fc_snn(node.layer, input_spikerate, e_wrram),
                                   mem_weights=self._e_rdweights_fc_snn(node.layer, input_spikerate, e_rdram),
                                   mem_bias=self._e_rdbias_fc_snn(node.layer, e_rdram),
                                   mem_io=self._e_rdin_snn(node.layer, input_spikerate, e_rdram)
                                   + self._e_wrout_snn(node.layer, output_spikerate, e_wrram),
                                   ops=self._e_ops_fc_snn(node.layer, input_spikerate, output_spikerate, timesteps, leak),
                                   addr=self._e_addr_fc_snn(node.layer, input_spikerate),
                                   input_spikerate=input_spikerates[node.layer.name],
                                   output_spikerate=output_spikerates[node.layer.name],
                                   input_is_binary=input_is_binary[node.layer.name],
                                   output_is_binary=output_is_binary[node.layer.name],
                                   )
                ems.append(em)

        em_total = EnergyMetrics(name='Total',
                                 mem_pot=sum(em.mem_pot for em in ems),
                                 mem_weights=sum(em.mem_weights for em in ems),
                                 mem_bias=sum(em.mem_bias for em in ems),
                                 mem_io=sum(em.mem_io for em in ems),
                                 ops=sum(em.ops for em in ems),
                                 addr=sum(em.addr for em in ems),
                                 input_spikerate=input_spikerates['__TOTAL__'],
                                 output_spikerate=output_spikerates['__TOTAL__'],
                                 input_is_binary=input_is_binary[modelgraph.nodes[1].layer.name], # First layer after 'input'
                                 output_is_binary=output_is_binary[modelgraph.nodes[-1].layer.name],
                                 )
        ems.append(em_total)

        return ems


    def _energy_summary(self, ems: list[EnergyMetrics]) -> str:
        """Generate a human-friendly text summary of the energy metrics per layer.

        :meta public:
        :param ems: List of EnergyMetrics per layer and the total
        :return: The text summary
        """
        pad = 12

        header = f'{"Layer": <{pad}} | {"EMemPot": <{pad}} | {"EMemWeights": <{pad}} | {"EMemBias": <{pad}} | {"EMemIO": <{pad}} |'
        header += f' {"EMemTotal": <{pad}} | {"EOps": <{pad}} | {"EAddr": <{pad}} | {"EOpsAddr": <{pad}} | {"ETotal": <{pad}} |'
        header += f' {"Input Spike Rate": <{pad}} | {"Output Spike Rate": <{pad}}\n'
        s = '—' * len(header) + '\n'
        s += header
        s += '—' * len(header) + '\n'
        for i, em in enumerate(ems):
            # Print in nJ, original values are in pJ
            s += f'{em.name: <{pad}.{pad}} | {em.mem_pot/1000: <{pad}.4} | {em.mem_weights/1000: <{pad}.4} |'
            s += f' {em.mem_bias/1000: <{pad}.4} | {em.mem_io/1000: <{pad}.4} | {em.mem_total/1000: <{pad}.4} |'
            s += f' {em.ops/1000: <{pad}.4} | {em.addr/1000: <{pad}.4} |'
            s += f' {em.opsaddr/1000: <{pad}.4} | {em.total/1000: <{pad}.4} |'

            input_spikerate_pad = 16 if em.input_is_binary else 11
            if em.input_spikerate is not None:
                s += f' {em.input_spikerate: <{input_spikerate_pad}.4}'
            else:
                s += f' {"N/A": <{input_spikerate_pad}.16}'
            if not em.input_is_binary:
                s += ' (NB)'
            s += ' |'

            output_spikerate_pad = 16 if em.output_is_binary else 11
            if em.output_spikerate is not None:
                s += f' {em.output_spikerate: <{output_spikerate_pad}.4}'
            else:
                s += f' {"N/A": <{output_spikerate_pad}.16}'
            if not em.output_is_binary:
                s += ' (NB)'
            s += '\n'

            s += ('-' if i < len(ems) - 2 else '—') * len(header) + '\n'

        s += ' NB = Non binary data'

        return s

    def _compute_spikerate(self,  # noqa: PLR0913
                           trainresult: TrainResult,
                           framework: SpikingJelly,
                           model: SNN,
                           dataset: RawData,
                           total_exclude_nonbinary: bool = True) -> tuple[dict[str, float],  # noqa: FBT002, FBT001
                                                                          dict[str, float],
                                                                          dict[str, bool],
                                                                          dict[str, bool]]:
        """Compute spike rate averaged over each neuron and timestep.

        A forward hook is added to each layer in order to record its input and output during inference performed over the given
        dataset.

        The layer name is computed using the internal :mod:`torch.fx` functions :meth:`torch.fx.graph._Namespace.create_name`
        and :meth:`torch.fx.graph._snake_case` so that it is compatible with
        :class:`qualia_codegen_core.graph.ModelGraph.ModelGraph`.

        :meta public:
        :param trainresult: TrainResult containing configuration for :class:`pytorch_lightning.Trainer` such as
                            :attr:`microai_core.microai.TrainResult.batch_size`
                            and :attr:`microai_core.microai.TrainResult.dataaugmentations`
        :param framework: A :class:`microai_plugin_snn.learningframework.SpikingJelly.SpikingJelly` instance
        :param model: A model to evaluate spike rate on
        :param dataset: The data to use for inference
        :return: A tuple of 4 dicts, first 2 dicts map a layer name to its input and output spike rates per timestep respectively,
                 the other 2 dicts map a layer name to ``True`` if the input or outputs are binary respectively,
                 ``False`` otherwise.
        """
        import torch
        from torch.fx.graph import _Namespace, _snake_case
        from torch.nn import Module
        namespace = _Namespace()  # type: ignore[no-untyped-call]

        @dataclass
        class SpikeCounter:
            spike_count: Number
            size: int
            binary: bool

        if_inputs_spike_count_and_size: dict[str, SpikeCounter]  = {}
        if_outputs_spike_count_and_size: dict[str, SpikeCounter] = {}

        def is_binary(a: torch.Tensor) -> bool:
            return torch.equal(a, (a > 0).float())

        def hook(layername: str, _: nn.Module, x: torch.Tensor, output: torch.Tensor) -> None:
            # WARNING: single input tensor only
            if isinstance(x, tuple) and len(x) > 1:
                logger.error('%s: Multiple inputs not supported.', layername)

            input_0 = (x[0] if isinstance(x, tuple) else x)
            inputnp = input_0.count_nonzero().item()
            outputnp = output.count_nonzero().item()

            if layername not in if_inputs_spike_count_and_size:
                if_inputs_spike_count_and_size[layername] = SpikeCounter(spike_count=inputnp,
                                                                         size=input_0.numel(),
                                                                         binary=is_binary(input_0))
            else:
                if_inputs_spike_count_and_size[layername].spike_count += inputnp
                if_inputs_spike_count_and_size[layername].size += input_0.numel()
                if_inputs_spike_count_and_size[layername].binary &= is_binary(input_0)

            if layername not in if_outputs_spike_count_and_size:
                if_outputs_spike_count_and_size[layername] = SpikeCounter(spike_count=outputnp,
                                                                          size=output.numel(),
                                                                          binary=is_binary(output))
            else:
                if_outputs_spike_count_and_size[layername].spike_count += outputnp
                if_outputs_spike_count_and_size[layername].size += output.numel()
                if_outputs_spike_count_and_size[layername].binary &= is_binary(output)

        handles = [layer.register_forward_hook(functools.partial(hook,
                                                                 namespace.create_name(_snake_case(layername), layer)))
                   for layername, layer in cast(Generator[tuple[str, Module], None, None], model.named_modules())
                   if not isinstance(layer, Quantizer)] # No hook to register for Quantizer module

        # Implement framework.predict()
        _ = framework.predict(model=model,
                              dataset=dataset,
                              batch_size=trainresult.batch_size,
                              dataaugmentations=trainresult.dataaugmentations,
                              experimenttracking=None)

        for handle in handles:
            handle.remove()

        input_spikerates = {n: sc.spike_count / sc.size for n, sc in if_inputs_spike_count_and_size.items()}
        output_spikerates = {n: sc.spike_count / sc.size for n, sc in if_outputs_spike_count_and_size.items()}

        # Filter out non-binary inputs/outputs if total_exclude_binary is True
        if total_exclude_nonbinary:
            logger.warning('Non-binary inputs/outputs are excluded from the total spike rate computation.')
        else:
            logger.warning('Non-binary inputs/outputs are included in the total spike rate computation.')
        filetered_if_inputs_spike_count_and_size = {n: sc for n, sc in if_inputs_spike_count_and_size.items()
                                                        if not total_exclude_nonbinary or sc.binary}
        filetered_if_outputs_spike_count_and_size = {n: sc for n, sc in if_outputs_spike_count_and_size.items()
                                                        if not total_exclude_nonbinary or sc.binary}
        # Special value account for the total spike rate across the whole network
        input_spikerates['__TOTAL__'] = (sum(sc.spike_count for sc in filetered_if_inputs_spike_count_and_size.values())
                                         / sum(sc.size for sc in filetered_if_inputs_spike_count_and_size.values()))
        output_spikerates['__TOTAL__'] = (sum(sc.spike_count for sc in filetered_if_outputs_spike_count_and_size.values())
                                         / sum(sc.size for sc in filetered_if_outputs_spike_count_and_size.values()))

        input_is_binary = {n: sc.binary for n, sc in if_inputs_spike_count_and_size.items()}
        output_is_binary = {n: sc.binary for n, sc in if_outputs_spike_count_and_size.items()}

        return input_spikerates, output_spikerates, input_is_binary, output_is_binary

    @override
    def __call__(self,  # noqa: C901
                 trainresult: TrainResult,
                 model_conf: ModelConfigDict) -> tuple[TrainResult, ModelConfigDict]:
        """Compute energy estimation metric from Lemaire et al, 2022.

        Uses Qualia CodeGen in order to build a :class:`qualia_codegen_core.graph.ModelGraph.ModelGraph` that is easier to parse.
        Call either :meth:`_compute_model_energy_snn` or :meth:`_compute_model_energy_fnn` depending on whether the model is an
        SNN or an FNN.
        Print the resulting metrics and log them to a CSV file inside the `logs/<bench.name>/EnergyEstimationMetric` directory.

        :meta public:
        :param trainresult: TrainResult containing the SNN or FNN model, the dataset and the training configuration
        :param model_conf: Unused
        :return: The unmodified trainresult
        """
        import spikingjelly.activation_based.layer as sjl  # type: ignore[import-untyped]
        import spikingjelly.activation_based.neuron as sjn  # type: ignore[import-untyped]
        from qualia_codegen_core.graph.layers import TAddLayer
        from qualia_codegen_plugin_snn.graph import TorchModelGraph
        from qualia_codegen_plugin_snn.graph.layers import TIfLayer
        from torch import nn

        import qualia_plugin_snn.learningmodel.pytorch.layers.quantized_SNN_layers as qsjn
        import qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly.quantized_layers as qsjl
        import qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly.quantized_layers1d as qsjl1d
        import qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly.quantized_layers2d as qsjl2d

        model = trainresult.model

        # SpikingJelly wrapped layers
        custom_layers: dict[type[nn.Module],
                            Callable[[nn.Module, TBaseLayer], tuple[type[TBaseLayer], list[Any]]]] = {
                                    sjl.AvgPool1d: TorchModelGraph.MODULE_MAPPING[nn.AvgPool1d],
                                    sjl.AvgPool2d: TorchModelGraph.MODULE_MAPPING[nn.AvgPool2d],
                                    sjl.BatchNorm1d: TorchModelGraph.MODULE_MAPPING[nn.BatchNorm1d],
                                    sjl.BatchNorm2d: TorchModelGraph.MODULE_MAPPING[nn.BatchNorm2d],
                                    sjl.Conv1d: TorchModelGraph.MODULE_MAPPING[nn.Conv1d],
                                    sjl.Conv2d: TorchModelGraph.MODULE_MAPPING[nn.Conv2d],
                                    sjl.Dropout: TorchModelGraph.MODULE_MAPPING[nn.Dropout],
                                    sjl.Flatten: TorchModelGraph.MODULE_MAPPING[nn.Flatten],
                                    sjl.Linear: TorchModelGraph.MODULE_MAPPING[nn.Linear],
                                    sjl.MaxPool1d: TorchModelGraph.MODULE_MAPPING[nn.MaxPool1d],
                                    sjl.MaxPool2d: TorchModelGraph.MODULE_MAPPING[nn.MaxPool2d],
                                    Add:  lambda *_: (TAddLayer, []),

                                    QuantizedIdentity:TorchModelGraph.MODULE_MAPPING[nn.Identity],
                                    qsjl.QuantizedLinear: TorchModelGraph.MODULE_MAPPING[nn.Linear],
                                    qsjl1d.QuantizedBatchNorm1d: TorchModelGraph.MODULE_MAPPING[nn.BatchNorm1d],
                                    qsjl2d.QuantizedBatchNorm2d: TorchModelGraph.MODULE_MAPPING[nn.BatchNorm2d],
                                    qsjl1d.QuantizedConv1d: TorchModelGraph.MODULE_MAPPING[nn.Conv1d],
                                    qsjl2d.QuantizedConv2d: TorchModelGraph.MODULE_MAPPING[nn.Conv2d],
                                    qsjl1d.QuantizedMaxPool1d: TorchModelGraph.MODULE_MAPPING[nn.MaxPool1d],
                                    qsjl2d.QuantizedMaxPool2d: TorchModelGraph.MODULE_MAPPING[nn.MaxPool2d],
                                    qsjn.QuantizedIFNode: TorchModelGraph.MODULE_MAPPING[sjn.IFNode],
                                    qsjn.QuantizedLIFNode: TorchModelGraph.MODULE_MAPPING[sjn.LIFNode],
                                    QuantizedAdd:  lambda *_: (TAddLayer, []),
                                    }
        step_mode = getattr(model, 'step_mode', 's')
        if step_mode != 's':
            logger.error("Only single-step mode (step_mode='s') is supported, current step_mode='%s'", step_mode)
            return trainresult, model_conf

        modelgraph = TorchModelGraph(model).convert(custom_layers=custom_layers)
        if modelgraph is None:
            logger.error('Model graph conversion failed')
            return trainresult, model_conf

        def e_rdram(x: int) -> float:
            return self._e_ram(x, self._mem_width)
        def e_wrram(x: int) -> float:
            return self._e_ram(x, self._mem_width)
        logger.info('Memory width set to %s bits', self._mem_width)

        if getattr(model, 'is_snn', False) or isinstance(model, SNN):
            if not isinstance(trainresult.framework, SpikingJelly):
                logger.error('LearningFramework must be SpikingJelly or a derived class, got: %s', type(trainresult.framework))
                raise TypeError

            if not self._fifo_size:
                logger.error('fifo_size is required for SNN models')
                raise ValueError

            # Tuple unpacking
            (input_spikerates,
             output_spikerates,
             input_is_binary,
             output_is_binary) = self._compute_spikerate(trainresult,
                                                         trainresult.framework,
                                                         model,
                                                         trainresult.testset,
                                                         total_exclude_nonbinary=self._total_spikerate_exclude_nonbinary)

            # Copy spikerate from IF activation layer to previous layer since it is the one energy is computed on
            for node in modelgraph.nodes:
                if isinstance(node.layer, TIfLayer):
                    output_spikerates[node.innodes[0].layer.name] = output_spikerates[node.layer.name]
                    # Also copy binary output check
                    output_is_binary[node.innodes[0].layer.name] = output_is_binary[node.layer.name]

            ems = self._compute_model_energy_snn(modelgraph,
                                                 input_spikerates,
                                                 output_spikerates,
                                                 input_is_binary,
                                                 output_is_binary,
                                                 model.timesteps,
                                                 e_rdram,
                                                 e_wrram)
        else:
            ems = self._compute_model_energy_fnn(modelgraph,
                                                 e_rdram,
                                                 e_wrram)

        for em in ems:
            self.csvlogger(em.asnamedtuple())

        logger.info(('Estimated model energy consumption for one inference on 45nm ASIC (nJ)'
                    ' and spike rate per neuron per timestep:\n%s'),
                    self._energy_summary(ems))

        return trainresult, model_conf

    #def _e_ram(self, mem_params: int) -> float:
    #    """Linear interpolation with {(8192, 10), (32768, 20), (1048576, 100)} point set (B, pJ) using least-squares methods.
    #
    #    45nm SRAM"""
    #    0.000082816 * mem_params + 13.2563

    def _e_ram(self, mem_params: int, bits: int) -> float:
        """From Computation_cost_metric.ipynb 45nm SRAM.

        :meta public:
        :param mem_params: Number of element stored in this memory
        :param bits: Data width in bits
        :return: Energy for a single read or write access in this memory
        """
        return 1.09 * 10**(-5) * mem_params * bits + 13.2

    @property
    def _e_add(self) -> float:
        """Energy for a single add operation.

        :meta public:
        """
        return self._m_e_add

    @property
    def _e_mul(self) -> float:
        """Energy for a single mul operation.

        :meta public:
        """
        return self._m_e_mul
