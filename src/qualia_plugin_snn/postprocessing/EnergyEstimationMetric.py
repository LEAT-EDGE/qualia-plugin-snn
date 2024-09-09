"""Provide the EnergyEstimationMetric postprocessing module based on Lemaire et al., 2022."""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import sys
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, Literal, NamedTuple, cast

import torch
from qualia_core.learningmodel.pytorch.layers import Add
from qualia_core.learningmodel.pytorch.layers.GlobalSumPool1d import GlobalSumPool1d
from qualia_core.learningmodel.pytorch.layers.GlobalSumPool2d import GlobalSumPool2d
from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedIdentity
from qualia_core.learningmodel.pytorch.layers.QuantizedAdd import QuantizedAdd
from qualia_core.learningmodel.pytorch.layers.QuantizedGlobalSumPool1d import QuantizedGlobalSumPool1d
from qualia_core.learningmodel.pytorch.layers.QuantizedGlobalSumPool2d import QuantizedGlobalSumPool2d
from qualia_core.learningmodel.pytorch.quantized_layers1d import QuantizedConv1d
from qualia_core.learningmodel.pytorch.quantized_layers2d import QuantizedConv2d
from qualia_core.learningmodel.pytorch.quantized_layers1d import QuantizedBatchNorm1d
from qualia_core.learningmodel.pytorch.quantized_layers2d import QuantizedBatchNorm2d
from qualia_core.learningmodel.pytorch.layers.quantized_layers import QuantizedLinear
from qualia_core.learningmodel.pytorch.Quantizer import Quantizer
from qualia_core.postprocessing.PostProcessing import PostProcessing
from qualia_core.typing import TYPE_CHECKING, ModelConfigDict
from qualia_core.utils.logger import Logger
from qualia_core.utils.logger.CSVFormatter import CSVFormatter
from torch import nn

from qualia_plugin_snn.learningframework.SpikingJelly import SpikingJelly
from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly.Add import Add as SNNAdd
from qualia_plugin_snn.learningmodel.pytorch.layers.spikingjelly.QuantizedAdd import QuantizedAdd as SNNQuantizedAdd
from qualia_plugin_snn.learningmodel.pytorch.SNN import SNN

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from qualia_codegen_core.graph import ModelGraph  # noqa: TC002
    from qualia_codegen_core.graph.layers import TAddLayer, TBaseLayer, TConvLayer, TDenseLayer  # noqa: TC002
    from qualia_core.datamodel.RawDataModel import RawData  # noqa: TC002
    from qualia_core.qualia import TrainResult  # noqa: TC002
    from torch.types import Number  # noqa: TC002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

@dataclass
class SpikeCounter:
    """Holds the statistics and properties of an input or output tensor of a layer after inference.

    :param spike_count: Number of spikes recorded
    :param tensor_sum: Sum of elements of the tensor
    :param size: Number of elements in the tensor
    :param binary: True if all elements are binary (0 or 1), False otherwise
    :param sample_count: Number of times the tensor has been updated
    """

    spike_count: Number
    tensor_sum: Number
    size: int
    binary: bool
    sample_count: int

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
    :param is_sj: If the layer is a SpikingJelly layer and has been processed as part of a Spiking Neural Network
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
    #: Input count per timestep
    input_count: Number
    #: Output count per timestep
    output_count: Number
    #: If input tensor only contains binary values, i.e., spikes
    input_is_binary: bool
    #: If output tensor only contains binary values, i.e., spikes
    output_is_binary: bool
    #: If the layer is a SpikingJelly layer and has been processed as part of a Spiking Neural Network
    is_sj: bool | Literal['Hybrid']

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
    :param input_count: Input count per timestep
    :param output_count: Output count per timestep
    :param input_is_binary: If input tensor only contains binary values, i.e., spikes
    :param output_is_binary: If output tensor only contains binary values, i.e., spikes
    :param is_sj: If the layer is a SpikingJelly layer and has been processed as part of a Spiking Neural Network
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

    # Predefined energy values for different bit widths
    energy_values = {
        8: {'add': 0.03, 'mul': 0.2},   # values for 8-bit
        # from `Computing’s Energy Problem (and what we can do about it) <https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf>`_,
        # Mark Horowitz, ISSCC 2014.

        32: {'add': 0.1, 'mul': 3.1},   # values for 32-bit
        # from `Computing’s Energy Problem (and what we can do about it) <https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf>`_,
        # Mark Horowitz, ISSCC 2014.
    }

    #: CSV logger to record metrics for each layer in a file inside the `logs/<bench.name>/EnergyEstimationMetric` directory
    csvlogger: CSVLogger[EnergyEstimationMetricLoggerFields]

    def __init__(self,
                 mem_width: int,
                 fifo_size: int = 0,
                 total_spikerate_exclude_nonbinary: bool = True,
                 estimation_type: None = None) -> None:  # noqa: FBT001, FBT002
        """Construct :class:`qualia_plugin_snn.postprocessing.EnergyEstimationMetric.EnergyEstimationMetric`.

        :param mem_width: Memory access size in bits, e.g. 16 for 16-bit quantization
        :param fifo_size: Size of the input/output FIFOs for each layer in SPLEAT
        :param total_spikerate_exclude_nonbinary: If True, exclude non-binary inputs/outputs from total spikerate computation
        """
        super().__init__()
        self._mem_width = mem_width
        self._fifo_size = fifo_size
        self._total_spikerate_exclude_nonbinary = total_spikerate_exclude_nonbinary
        self._set_energy_values(mem_width, estimation_type)

        # Initialize CSV suffix with the memory width, self_m_e_add, self_m_e_mul and estimation type 
        # without "{", "}", ":", ",", "'" or " " characters
        suffix = f'mem{mem_width}bit_{self._m_e_add}_{self._m_e_mul}'
        if estimation_type is not None:
            suffix += f'_{estimation_type}'
        suffix = suffix.replace('{', '').replace('}', '').replace(':', '').replace(',', '').replace("'", '').replace(" ", '')
        self.csvlogger = Logger(name='EnergyEstimationMetric', suffix=suffix+".csv", formatter=CSVFormatter())
        self.csvlogger.fields = EnergyEstimationMetricLoggerFields

    def _set_estimation_type(self, bit_width: int, type: str, estimation_type: str | int) -> float:
        """
        Set the estimation type for the energy values.

        If estimation_type is not in ['ICONIP', 'saturation', 'linear', 'quadratic'], raise ValueError.

        If estimation_type is ''ICONIP', use the energy values from the ICONIP 2022 paper (i.e. 32-bit values).

        If estimation_type is 'saturation', use self.energy_values[8][type] for 8-bit and below, and 
        self.energy_values[32][type] for bit widths between 9 and 32.

        If estimation_type is 'linear', use self.energy_values[8][type] and self.energy_values[32][type] 
        to estimate the energy values by solving a linear equation: y = m*bit_width + c.

        If estimation_type is 'quadratic', use self.energy_values[8][type] and self.energy_values[32][type] 
        to estimate the energy values by solving a quadratic equation: y = a*bit_width^2 + b*bit_width + c.

        :param estimation_type: The estimation type for the energy values.
        """
        
        # Check for valid estimation_type
        if estimation_type not in ['ICONIP','saturation', 'linear', 'quadratic']:
            raise ValueError("Invalid estimation_type. Must be one of ['ICONIP','saturation', 'linear', 'quadratic']")

        # Get the 8-bit and 32-bit energy values for the given type
        energy_8bit = self.energy_values[8][type]
        energy_32bit = self.energy_values[32][type]

        if estimation_type == 'ICONIP':
            # For ICONIP, return the 32-bit value for bit_width <= 32
            energy = energy_32bit

        elif estimation_type == 'saturation':
            # For saturation, return the 8-bit value for bit_width <= 8, otherwise the 32-bit value
            if bit_width <= 8:
                energy = energy_8bit
            elif 9 <= bit_width <= 32:
                energy = energy_32bit
            else:
                raise ValueError("Bit width out of supported range (1-32) for saturation estimation.")

        elif estimation_type == 'linear':
            # For linear, solve the equation y = m*bit_width + c
            # Use (x1, y1) = (8, energy_8bit) and (x2, y2) = (32, energy_32bit) to find m and c
            x1, y1 = 8, energy_8bit
            x2, y2 = 32, energy_32bit

            # Solve for slope (m) and intercept (c)
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            # Estimate energy for the given bit_width
            energy = m * bit_width + c

        elif estimation_type == 'quadratic':
            # For quadratic, solve the equation y = a*bit_width^2 + b*bit_width + c
            # We assume two points (8, energy_8bit) and (32, energy_32bit), plus assume c = 0 (or another known value)
            x1, y1 = 8, energy_8bit
            x2, y2 = 32, energy_32bit

            # We need a third point to solve a quadratic equation. We assume (bit_width=0, energy=0) as a reference point.
            x0, y0 = 0, 0  # This assumes no energy consumption at 0 bits (can be modified if you have another reference)
            
            import numpy as np
            # Solve the system of equations to find a, b, and c
            A = np.array([
                [x0**2, x0, 1],
                [x1**2, x1, 1],
                [x2**2, x2, 1]
            ])
            B = np.array([y0, y1, y2])

            # Solve for [a, b, c]
            a, b, c = np.linalg.solve(A, B)

            # Estimate energy for the given bit_width using quadratic equation
            energy = a * bit_width**2 + b * bit_width + c
        else:
            raise ValueError("Invalid estimation_type. Must be one of ['ICONIP', 'saturation', 'linear', 'quadratic']")    
        return energy

    
    def _set_energy_values(self, bit_width: int, estimation_type : None) -> None:
        """
        Set the energy values for the given bit width.

        If bit_width is 8 or 32, set the energy values using the predefined energy values.

        Else if estimation_type is not None, check if estimation_type['add'] and estimation_type['mul'] are defined,
        and set the energy values using the estimation type.

        Else raise ValueError.

        :param bit_width: The bit width for the energy values.
        :param estimation_type: The estimation type for the energy values.
        """

        if bit_width in self.energy_values and estimation_type is None:
            self._m_e_add = self.energy_values[bit_width]['add']
            self._m_e_mul = self.energy_values[bit_width]['mul']
            print(f'Using {bit_width}-bit energy values from predefined values in Mark Horowitz, ISSCC 2014')
        elif estimation_type is not None:
            if 'add' in estimation_type and 'mul' in estimation_type:
                self._m_e_add = self._set_estimation_type(bit_width, 'add', estimation_type['add'])
                self._m_e_mul = self._set_estimation_type(bit_width, 'mul', estimation_type['mul'])
                print(f'Using {bit_width}-bit energy values estimated using {estimation_type}.')
            else:
                raise ValueError("estimation_type must contain 'add' and 'mul' keys.")
        
            
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

    def _rdin_add_fnn(self, layer: TAddLayer) -> int:
        """Count number of read operations for the input of an add layer in a formal neural network.

        #InLayers × Nin.

        :meta public:
        :param layer: An add layer
        :return: Number of read operations for the input of an add layer in a formal neural network
        """  # noqa: RUF002
        return sum(sum(input_shape[1:]) for input_shape in layer.input_shape)

    def _e_rdin_add_fnn(self, layer: TAddLayer, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for the input of an add layer in a formal neural network.

        :meta public:
        :param layer: An add layer
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the input of an add layer in a formal neural network
        """
        return self._rdin_add_fnn(layer) * e_rdram(math.prod(layer.input_shape[0][1:]))


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

    def _wrout_add_fnn(self, layer: TAddLayer) -> int:
        """Count number of write operations for the output of an add layer in a formal neural network.

        Nout.

        :meta public:
        :param layer: An add layer
        :return: Number of write operations for the output of an add layer in a formal neural network
        """
        return layer.output_shape[0][-1]

    def _e_wrout_add_fnn(self, layer: TAddLayer, e_wrram: Callable[[int], float]) -> float:
        """Compute energy for write operations for the output of an add layer in a formal neural network.

        :meta public:
        :param layer: An add layer
        :param e_rdram: Function to compute memory write access energy for a given memory size
        :return: Energy for write operations for the output of an add layer in a formal neural network
        """
        return self._wrout_add_fnn(layer) * e_wrram(math.prod(layer.output_shape[0][1:]))

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

    def _mac_ops_add_fnn(self, layer: TAddLayer) -> int:
        """Count number of multiply-accumulate operations inside an add layer in a formal neural network.

        0.

        :meta public:
        :param layer: An add layer
        :return: 0.
        """
        return layer.input_shape[0][-1] * layer.output_shape[0][-1]

    def _acc_ops_add_fnn(self, layer: TAddLayer) -> int:
        """Count number of accumulate operations inside an add layer in a formal neural network.

        (#InLayers -1 ) × Nin
        Nout.

        :meta public:
        :param layer: An add layer
        :return: Number of accumulate operations inside an add layer in a formal neural network.
        """  # noqa: RUF002
        return min(sum(input_shape[1:]) for input_shape in layer.input_shape) * (len(layer.input_shape) - 1)

    def _e_ops_add_fnn(self, layer: TAddLayer) -> float:
        """Compute energy for multiply-accumulate and accumulate operations inside an add layer in formal neural network.

        :meta public:
        :param layer: A fully-connected layer
        :return: Energy for multiply-accumulate and accumulate operations inside an add layer in formal neural network.
        """
        return self._mac_ops_add_fnn(layer) * (self._e_mul + self._e_add) + self._acc_ops_add_fnn(layer) * self._e_add


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

    def _mac_addr_add_fnn(self, _: TAddLayer) -> int:
        """Count number of multiply-accumulate operations for addressing an add layer in a formal neural network.

        0

        :meta public:
        :return: 0
        """
        return 0

    def _acc_addr_add_fnn(self, layer: TAddLayer) -> int:
        """Count number of accumulate operations for addressing an add layer in a formal neural network.

        Nin.

        :meta public:
        :param layer: An add layer
        :return: Number of accumulate operations for addressing an add layer in a formal neural network
        """
        return min(sum(input_shape[1:]) for input_shape in layer.input_shape)

    def _e_addr_add_fnn(self, layer: TAddLayer) -> float:
        """Compute energy for addressing an add layer in a formal neural network.

        :meta public:
        :param layer: An add layer
        :return: Energy for addressing an add layer in a formal neural network.
        """
        return self._mac_addr_add_fnn(layer) * (self._e_mul + self._e_add) + self._acc_addr_add_fnn(layer) * self._e_add


    def _compute_model_energy_fnn(self,
                                  modelgraph: ModelGraph,
                                  e_rdram: Callable[[int], float],
                                  e_wrram: Callable[[int], float]) -> list[EnergyMetrics]:
        """Compute the energy per inference for each layer of a formal neural network.

        Supports the following layers:

        * :class:`qualia_codegen_core.graph.layers.TConvLayer.TConvLayer`
        * :class:`qualia_codegen_core.graph.layers.TDenseLayer.TDenseLayer`
        * :class:`qualia_codegen_core.graph.layers.TAddLayer.TAddLayer`

        :meta public:
        :param modelgraph: Model to computer energy on
        :param e_rdram: Function to compute memory read energy for a given memory size
        :param e_wrram: Function to computer memory write energy for a given memory size
        :return: A list of EnergyMetrics for each layer and a total with fields populated with energy estimation
        """
        from qualia_codegen_core.graph.layers import TConvLayer, TDenseLayer, TAddLayer, TFlattenLayer

        ems: list[EnergyMetrics] = []
        for node in modelgraph.nodes:
            if isinstance(node.layer, TFlattenLayer):
                # Flatten is assumed to not do anything for on-target inference
                em = EnergyMetrics(name=node.layer.name,
                                   mem_pot=0,
                                   mem_weights=0,
                                   mem_bias=0,
                                   mem_io=0,
                                   ops=0,
                                   addr=0,
                                   input_spikerate=None,
                                   output_spikerate=None,
                                   input_count=None,
                                   output_count=None,
                                   input_is_binary=False,
                                   output_is_binary=False,
                                   is_sj=False)
            elif isinstance(node.layer, TConvLayer):
                em = EnergyMetrics(name=node.layer.name,
                                   mem_pot=0,
                                   mem_weights=self._e_rdweights_conv_fnn(node.layer, e_rdram),
                                   mem_bias=self._e_rdbias_conv_fnn(node.layer, e_rdram),
                                   mem_io=self._e_rdin_conv_fnn(node.layer, e_rdram) + self._e_wrout_conv_fnn(node.layer, e_wrram),
                                   ops=self._e_ops_conv_fnn(node.layer),
                                   addr=self._e_addr_conv_fnn(node.layer),
                                   input_spikerate=None,
                                   output_spikerate=None,
                                   input_count=None,
                                   output_count=None,
                                   input_is_binary=False,
                                   output_is_binary=False,
                                   is_sj=False)
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
                                   input_count=None,
                                   output_count=None,
                                   input_is_binary=False,
                                   output_is_binary=False,
                                   is_sj=False)
            elif isinstance(node.layer, TAddLayer):
                # Assume element-wise addition of inputs, meaning we need to read inputs and write outputs

                em = EnergyMetrics(name=node.layer.name,
                                    mem_pot=0,  # No potentials for add layer
                                    mem_weights=0,  # No weights for add layer
                                    mem_bias=0,  # No biases for add layer
                                    mem_io=self._e_rdin_add_fnn(node.layer, e_rdram) + self._e_wrout_add_fnn(node.layer, e_wrram),
                                    ops=self._e_ops_add_fnn(node.layer),
                                    addr=self._e_addr_add_fnn(node.layer),
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
                                 input_count=None,
                                 output_count=None,
                                 input_is_binary=False,
                                 output_is_binary=False,
                                 is_sj=False)
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

    def _rdbias_conv_snn(self, layer: TConvLayer, timesteps: int) -> int:
        """Count number of read operations for the biases of a convolutional layer in a spiking neural network.

        Eq. 7s: Cout × Wout × Hout.

        :meta public:
        :param layer: A convolutional layer
        :param timesteps: Number of timesteps
        :return: Number of read operations for the biases of a convolutional layer in a spiking neural network
        """  # noqa: RUF002
        return math.prod(layer.output_shape[0][1:]) * timesteps

    def _e_rdbias_conv_snn(self, layer: TConvLayer, timesteps: int, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for the biases of a convolutional layer in a spiking neural network.

        :meta public:
        :param layer: A convolutional layer
        :param timesteps: Number of timesteps
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the biases of a convolutional layer in a spiking neural network or 0 if the layer
                 does not use biases
        """
        if layer.use_bias:
            return self._rdbias_conv_snn(layer, timesteps) * e_rdram(layer.bias.size)
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

    def _rdbias_fc_snn(self, layer: TDenseLayer, timesteps: int) -> int:
        """Count number of read operations for the biases of a fully-connected layer in a spiking neural network.

        Eq. 6.2s: Nout.

        :meta public:
        :param layer: A fully-connected layer
        :param timesteps: Number of timesteps
        :return: Number of read operations for the biases of a fully-connected layer in a spiking neural network
        """
        return layer.output_shape[0][-1] * timesteps

    def _e_rdbias_fc_snn(self, layer: TDenseLayer, timesteps: int, e_rdram: Callable[[int], float]) -> float:
        """Compute energy for read operations for the biases of a fully-connected layer in a spiking neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param timesteps: Number of timesteps
        :param e_rdram: Function to compute memory read access energy for a given memory size
        :return: Energy for read operations for the biases of a fully-connected layer in a spiking neural network or 0 if the layer
            does not use biases
        """
        if layer.use_bias:
            return self._rdbias_fc_snn(layer, timesteps) * e_rdram(layer.bias.size)
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


    def _wrpot_conv_snn(self, layer: TConvLayer, input_spikerate: float, timesteps: int) -> float:
        """Count average number of write operations for the potentials of a convolutional layer in a spiking neural network.

        Eq 14: θl-1 × Cout × Wkernel × Hkernel + Cout × Hout × Wout.

        :meta public:
        :param layer: A convolutional layer
        :param timesteps: Number of timesteps
        :param input_spikerate: Average spike per input per inference
        :return: Average number of write operations for the potentials of a convolutional layer in a spiking neural network
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return (theta_in * layer.output_shape[0][-1] * math.prod(layer.kernel_size)
                + math.prod(layer.output_shape[0][1:]) * timesteps)

    def _e_wrpot_conv_snn(self,
                          layer: TConvLayer,
                          input_spikerate: float,
                          timesteps: int,
                          e_wrram: Callable[[int], float]) -> float:
        """Compute average energy for write operations for the potentials of a convolutional layer in a spiking neural network.

        :meta public:
        :param layer: A convolutional layer
        :param timesteps: Number of timesteps
        :param input_spikerate: Average spike per input per inference
        :param e_rdram: Function to compute memory write access energy for a given memory size
        :return: Average energy for write operations for the potentials of a convolutional layer in a spiking neural network
        """
        return self._wrpot_conv_snn(layer, input_spikerate, timesteps) * e_wrram(math.prod(layer.output_shape[0][1:]))

    def _wrpot_fc_snn(self, layer: TDenseLayer, input_spikerate: float, timesteps: int) -> float:
        """Count average number of write operations for the potentials of a fully-connected layer in a spiking neural network.

        Eq 15: θl-1 × Nout + Nout.

        :meta public:
        :param layer: A fully-connected layer
        :param timesteps: Number of timesteps
        :param input_spikerate: Average spike per input per inference
        :return: Average number of write operations for the potentials of a fully-connected layer in a spiking neural network
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return theta_in * layer.output_shape[0][-1] + layer.output_shape[0][-1] * timesteps

    def _e_wrpot_fc_snn(self,
                        layer: TDenseLayer,
                        input_spikerate: float,
                        timesteps: int,
                        e_wrram: Callable[[int], float]) -> float:
        """Compute average energy for write operations for the potentials of a fully-connected layer in a spiking neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param timesteps: Number of timesteps
        :param input_spikerate: Average spike per input per inference
        :param e_rdram: Function to compute memory write access energy for a given memory size
        :return: Average energy for write operations for the potentials of a fully-connected layer in a spiking neural network
        """
        return self._wrpot_fc_snn(layer, input_spikerate, timesteps) * e_wrram(math.prod(layer.output_shape[0][1:]))


    def _rdpot_conv_snn(self, layer: TConvLayer, input_spikerate: float, timesteps: int) -> float:
        """Count average number of read operations for the potentials of a convolutional layer in a spiking neural network.

        Eq 9: θl-1 × Cout × Wkernel × Hkernel + Cout × Hout × Wout.

        :meta public:
        :param layer: A convolutional layer
        :param timesteps: Number of timesteps
        :param input_spikerate: Average spike per input per inference
        :return: Average number of read operations for the potentials of a convolutional layer in a spiking neural network
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return (theta_in * layer.output_shape[0][-1] * math.prod(layer.kernel_size)
                + math.prod(layer.output_shape[0][1:]) * timesteps)

    def _e_rdpot_conv_snn(self,
                          layer: TConvLayer,
                          input_spikerate: float,
                          timesteps: int,
                          e_rdram: Callable[[int], float]) -> float:
        """Compute average energy for write operations for the potentials of a convolutional layer in a spiking neural network.

        :meta public:
        :param layer: A convolutional layer
        :param timesteps: Number of timesteps
        :param input_spikerate: Average spike per input per inference
        :param e_rdram: Function to compute memory write access energy for a given memory size
        :return: Average energy for read operations for the potentials of a convolutional layer in a spiking neural network
        """
        return self._rdpot_conv_snn(layer, input_spikerate, timesteps) * e_rdram(math.prod(layer.output_shape[0][1:]))

    def _rdpot_fc_snn(self, layer: TDenseLayer, input_spikerate: float, timesteps: int) -> float:
        """Count average number of read operations for the potentials of a fully-connected layer in a spiking neural network.

        Eq 10: θl-1 × Nout + Nout.

        :meta public:
        :param layer: A fully-connected layer
        :param timesteps: Number of timesteps
        :param input_spikerate: Average spike per input per inference
        :return: Average number of read operations for the potentials of a fully-connected layer in a spiking neural network
        """  # noqa: RUF002
        theta_in = input_spikerate * math.prod(layer.input_shape[0][1:])
        return theta_in * layer.output_shape[0][-1] + layer.output_shape[0][-1] * timesteps

    def _e_rdpot_fc_snn(self,
                        layer: TDenseLayer,
                        input_spikerate: float,
                        timesteps: int,
                        e_wrram: Callable[[int], float]) -> float:
        """Compute average energy for read operations for the potentials of a fully-connected layer in a spiking neural network.

        :meta public:
        :param layer: A fully-connected layer
        :param timesteps: Number of timesteps
        :param input_spikerate: Average spike per input per inference
        :param e_rdram: Function to compute memory write access energy for a given memory size
        :return: Average energy for read operations for the potentials of a fully-connected layer in a spiking neural network
        """
        return self._rdpot_fc_snn(layer, input_spikerate, timesteps) * e_wrram(math.prod(layer.output_shape[0][1:]))


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

    def _e_ops_conv_snn(self,
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

    def _e_ops_fc_snn(self,
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



    def _compute_model_energy_snn(self,  # noqa: PLR0913, C901, PLR0912, PLR0915
                                  modelgraph: ModelGraph,
                                  input_spikerates: dict[str, float],
                                  output_spikerates: dict[str, float],
                                  input_is_binary: dict[str, bool],
                                  output_is_binary: dict[str, bool],
                                  input_counts: dict[str, Number],
                                  output_counts: dict[str, Number],
                                  is_module_sj: dict[str, bool],
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
        :param input_counts: Dict of layer names and number of inputs for the layer
        :param output_counts: Dict of layer names and number of outputs for the layer
        :param timesteps: Number of timesteps
        :param e_rdram: Function to compute memory read energy for a given memory size
        :param e_wrram: Function to computer memory write energy for a given memory size
        :return: A list of EnergyMetrics for each layer and a total with fields populated with energy estimation
        """
        from qualia_codegen_core.graph.layers import TAddLayer, TConvLayer, TDenseLayer, TFlattenLayer, TInputLayer
        from qualia_codegen_plugin_snn.graph.layers import TIfLayer, TLifLayer

        ems: list[EnergyMetrics] = []
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

            em: EnergyMetrics | None = None

            if isinstance(node.layer, TFlattenLayer):
                # Flatten is assumed to not do anything for on-target inference
                em = EnergyMetrics(name=node.layer.name,
                                   mem_pot=0,
                                   mem_weights=0,
                                   mem_bias=0,
                                   mem_io=0,
                                   ops=0,
                                   addr=0,
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
                        e_mem_io = (self._e_rdin_snn(node.layer, input_spikerates[node.layer.name], e_rdram)
                                    + self._e_wrout_snn(node.layer, output_spikerate, e_wrram))
                        e_ops = (self._e_ops_conv_fnn(node.layer) * input_spikerates[node.layer.name] # MAC operations Bias is not considered here !!... 
                                 + (math.prod(node.layer.output_shape[0][1:]) * self._e_mul if leak else 0) # Leak
                                 + output_spikerate * math.prod(node.layer.output_shape[0][1:]) * self._e_add) # Reset
                        e_addr = self._e_addr_conv_snn(node.layer, input_spikerates[node.layer.name])
                        mem_weights = self._e_rdweights_conv_snn(node.layer, input_spikerates[node.layer.name], e_rdram)
                        is_sj = 'Hybrid'
                    else:
                        e_mem_io = (self._e_rdin_snn(node.layer, input_spikerate, e_rdram)
                                    + self._e_wrout_snn(node.layer, output_spikerate, e_wrram))
                        e_ops = self._e_ops_conv_snn(node.layer, input_spikerate, output_spikerate, timesteps, leak)
                        e_addr = self._e_addr_conv_snn(node.layer, input_spikerate)
                        mem_weights = self._e_rdweights_conv_snn(node.layer, input_spikerate, e_rdram)
                        is_sj = True

                    em = EnergyMetrics(name=node.layer.name,
                                       mem_pot=self._e_wrpot_conv_snn(node.layer, input_spikerate, timesteps, e_wrram)
                                       + self._e_rdpot_conv_snn(node.layer, input_spikerate, timesteps, e_rdram),
                                       mem_weights=mem_weights,
                                       mem_bias=self._e_rdbias_conv_snn(node.layer, timesteps, e_rdram),
                                       mem_io=e_mem_io,
                                       ops=e_ops,
                                       addr=e_addr,
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
                        e_mem_io = (self._e_rdin_fc_fnn(node.layer, e_rdram) # Dense reading of inputs
                                    + self._e_wrout_snn(node.layer, output_spikerate, e_wrram))
                        e_ops = (self._e_ops_fc_fnn(node.layer) # MAC operations
                                 + (math.prod(node.layer.output_shape[0][1:]) * self._e_mul if leak else 0) # Leak
                                 + output_spikerate * math.prod(node.layer.output_shape[0][1:]) * self._e_add) # Reset
                        e_addr = self._e_addr_fc_fnn(node.layer) # Dense addressing
                        is_sj = 'Hybrid'
                    else:
                        e_mem_io = (self._e_rdin_snn(node.layer, input_spikerate, e_rdram)
                                    + self._e_wrout_snn(node.layer, output_spikerate, e_wrram))
                        e_ops = self._e_ops_fc_snn(node.layer, input_spikerate, output_spikerate, timesteps, leak)
                        e_addr = self._e_addr_fc_snn(node.layer, input_spikerate)
                        is_sj = True

                    em = EnergyMetrics(name=node.layer.name,
                                       mem_pot=self._e_wrpot_fc_snn(node.layer, input_spikerate, timesteps, e_wrram)
                                       + self._e_rdpot_fc_snn(node.layer, input_spikerate, timesteps, e_wrram),
                                       mem_weights=self._e_rdweights_fc_snn(node.layer, input_spikerate, e_rdram),
                                       mem_bias=self._e_rdbias_fc_snn(node.layer, timesteps, e_rdram),
                                       mem_io=e_mem_io,
                                       ops=e_ops,
                                       addr=e_addr,
                                       input_spikerate=input_spikerates[node.layer.name],
                                       output_spikerate=output_spikerates[node.layer.name],
                                       input_count=input_counts[node.layer.name],
                                       output_count=output_counts[node.layer.name],
                                       input_is_binary=input_is_binary[node.layer.name],
                                       output_is_binary=output_is_binary[node.layer.name],
                                       is_sj=is_sj,
                                       )
                elif isinstance(node.layer, TAddLayer):
                    em = EnergyMetrics(name=node.layer.name,
                                       mem_pot=0,
                                       mem_weights=0,
                                       mem_bias=0,
                                       mem_io=0,
                                       ops=0,
                                       addr=0,
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
                    em = EnergyMetrics(name=node.layer.name,
                                       mem_pot=0,
                                       mem_weights=self._e_rdweights_conv_fnn(node.layer, e_rdram),
                                       mem_bias=self._e_rdbias_conv_fnn(node.layer, e_rdram),
                                       mem_io=self._e_rdin_conv_fnn(node.layer, e_rdram)
                                       + self._e_wrout_conv_fnn(node.layer, e_wrram),
                                       ops=self._e_ops_conv_fnn(node.layer),
                                       addr=self._e_addr_conv_fnn(node.layer),
                                       input_spikerate=input_spikerates[node.layer.name],
                                       output_spikerate=output_spikerates[node.layer.name],
                                       input_count=input_counts[node.layer.name],
                                       output_count=output_counts[node.layer.name],
                                       input_is_binary=input_is_binary[node.layer.name],
                                       output_is_binary=output_is_binary[node.layer.name],
                                       is_sj=is_module_sj[node.layer.name],
                                       )
                elif isinstance(node.layer, TDenseLayer):
                    em = EnergyMetrics(name=node.layer.name,
                                       mem_pot=0,
                                       mem_weights=self._e_rdweights_fc_fnn(node.layer, e_rdram),
                                       mem_bias=self._e_rdbias_fc_fnn(node.layer, e_rdram),
                                       mem_io=self._e_rdin_fc_fnn(node.layer, e_rdram) + self._e_wrout_fc_fnn(node.layer, e_wrram),
                                       ops=self._e_ops_fc_fnn(node.layer),
                                       addr=self._e_addr_fc_fnn(node.layer),
                                       input_spikerate=input_spikerates[node.layer.name],
                                       output_spikerate=output_spikerates[node.layer.name],
                                       input_count=input_counts[node.layer.name],
                                       output_count=output_counts[node.layer.name],
                                       input_is_binary=input_is_binary[node.layer.name],
                                       output_is_binary=output_is_binary[node.layer.name],
                                       is_sj=is_module_sj[node.layer.name],
                                       )
                elif isinstance(node.layer, TAddLayer):
                    em = EnergyMetrics(name=node.layer.name,
                                       mem_pot=0,
                                       mem_weights=0,
                                       mem_bias=0,
                                       mem_io=self._e_rdin_add_fnn(node.layer, e_rdram)
                                       + self._e_wrout_add_fnn(node.layer, e_wrram),
                                       ops=self._e_ops_add_fnn(node.layer),
                                       addr=self._e_addr_add_fnn(node.layer),
                                       input_spikerate=input_spikerates[node.layer.name],
                                       output_spikerate=output_spikerates[node.layer.name],
                                       input_count=input_counts[node.layer.name],
                                       output_count=output_counts[node.layer.name],
                                       input_is_binary=input_is_binary[node.layer.name],
                                       output_is_binary=output_is_binary[node.layer.name],
                                       is_sj=is_module_sj[node.layer.name],
                                       )

            if em is None: # We do not know how to handle this layer, set energy values to 0
                logger.warning('%s not handled, result may be inaccurate', node.layer.name)
                em = EnergyMetrics(name=node.layer.name,
                                   mem_pot=0,
                                   mem_weights=0,
                                   mem_bias=0,
                                   mem_io=0,
                                   ops=0,
                                   addr=0,
                                   input_spikerate=input_spikerates[node.layer.name],
                                   output_spikerate=output_spikerates[node.layer.name],
                                   input_count=input_counts[node.layer.name],
                                   output_count=output_counts[node.layer.name],
                                   input_is_binary=input_is_binary[node.layer.name],
                                   output_is_binary=output_is_binary[node.layer.name],
                                   is_sj=is_module_sj[node.layer.name])
            ems.append(em)

        total_is_sj: bool | Literal['Hybrid'] = (True if all(is_sj is True for is_sj in is_module_sj.values()) else
                                                 False if all(not is_sj for is_sj in is_module_sj.values()) else
                                                 'Hybrid')

        em_total = EnergyMetrics(name='Total',
                                 mem_pot=sum(em.mem_pot for em in ems),
                                 mem_weights=sum(em.mem_weights for em in ems),
                                 mem_bias=sum(em.mem_bias for em in ems),
                                 mem_io=sum(em.mem_io for em in ems),
                                 ops=sum(em.ops for em in ems),
                                 addr=sum(em.addr for em in ems),
                                 input_spikerate=input_spikerates['__TOTAL__'],
                                 output_spikerate=output_spikerates['__TOTAL__'],
                                 input_count=input_counts['__TOTAL__'],
                                 output_count=output_counts['__TOTAL__'],
                                 input_is_binary=input_is_binary[modelgraph.nodes[1].layer.name], # First layer after 'input'
                                 output_is_binary=output_is_binary[modelgraph.nodes[-1].layer.name],
                                 is_sj=total_is_sj,
                                 )
        ems.append(em_total)

        return ems


    def _energy_summary(self, ems: list[EnergyMetrics]) -> str:
        """Generate a human-friendly text summary of the energy metrics per layer.

        :meta public:
        :param ems: List of EnergyMetrics per layer and the total
        :return: The text summary
        """
        pad = 11
        pad_name = max(len(em.name) for em in ems)

        header = f'{"Layer": <{pad_name}} | {"EMemPot": <{pad}} | {"EMemWeights": <{pad}} | {"EMemBias": <{pad}} |'
        header += f' {"EMemIO": <{pad}} | {"EMemTotal": <{pad}} | {"EOps": <{pad}} | {"EAddr": <{pad}} | {"EOpsAddr": <{pad}} |'
        header += f' {"ETotal": <{pad}} | {"SNN": <{pad}} | {"Input Spike Rate": <{pad}} | {"Output Spike Rate": <{pad}}\n'
        s = '—' * len(header) + '\n'
        s += header
        s += '—' * len(header) + '\n'
        for i, em in enumerate(ems):
            # Print in nJ, original values are in pJ
            s += f'{em.name: <{pad_name}.{pad_name}} | {em.mem_pot/1000: <{pad}.4} | {em.mem_weights/1000: <{pad}.4} |'
            s += f' {em.mem_bias/1000: <{pad}.4} | {em.mem_io/1000: <{pad}.4} | {em.mem_total/1000: <{pad}.4} |'
            s += f' {em.ops/1000: <{pad}.4} | {em.addr/1000: <{pad}.4} |'
            s += f' {em.opsaddr/1000: <{pad}.4} | {em.total/1000: <{pad}.4} |'
            s += f' {em.is_sj!s: <{pad}} |'

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

    def _record_spike_count(self,
                            trainresult: TrainResult,
                            framework: SpikingJelly,
                            model: SNN,
                            dataset: RawData) -> tuple[dict[str, bool],
                                                       dict[str, SpikeCounter],
                                                       dict[str, SpikeCounter]]:
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
        import spikingjelly.activation_based.base as sjb  # type: ignore[import-untyped]
        import torch
        from torch.fx.graph import _Namespace, _snake_case
        from torch.nn import Module
        namespace = _Namespace()  # type: ignore[no-untyped-call]


        if_inputs_spike_count_and_size: dict[str, SpikeCounter]  = {}
        if_outputs_spike_count_and_size: dict[str, SpikeCounter] = {}
        is_module_sj: dict[str, bool] = {} # Is this module a SpikingJelly module and therefore we should compute as an SNN layer

        def is_binary(a: torch.Tensor) -> bool:
            return torch.equal(a, (a > 0).float())

        def is_sj(m: nn.Module) -> bool:
            # Module is assumed to be a SpikingJelly module if it inherits from StepModule
            return isinstance(m, sjb.StepModule)

        def hook(layername: str, module: nn.Module, x: torch.Tensor, output: torch.Tensor) -> None:
            # Concatenate in last dim to handle Add layer
            input_cat = (torch.cat(cast(tuple[torch.Tensor], x), dim=-1) if isinstance(x, tuple) else x)

            inputnp = input_cat.count_nonzero().item()
            outputnp = output.count_nonzero().item()

            # Used for special case of ResNet where spikes are not binary
            input_sum = input_cat.sum().item()
            output_sum = output.sum().item()

            # In case of multi-step mode, apply number of timesteps here to compute average over timestep
            # since we do not loop multiple times over the sample unlike single-step mode
            nb_sample = (math.prod(input_cat.shape[0:2])
                         if isinstance(module, sjb.StepModule) and module.step_mode == 'm'
                         else input_cat.shape[0])

            is_module_sj[layername] = is_sj(module)

            if layername not in if_inputs_spike_count_and_size:
                if_inputs_spike_count_and_size[layername] = SpikeCounter(spike_count=inputnp,
                                                                         tensor_sum=input_sum,
                                                                         size=input_cat.numel(),
                                                                         binary=is_binary(input_cat),
                                                                         sample_count=nb_sample)
            else:
                if_inputs_spike_count_and_size[layername].spike_count += inputnp
                if_inputs_spike_count_and_size[layername].tensor_sum += input_sum
                if_inputs_spike_count_and_size[layername].size += input_cat.numel()
                if_inputs_spike_count_and_size[layername].binary &= is_binary(input_cat)
                if_inputs_spike_count_and_size[layername].sample_count += nb_sample

            if layername not in if_outputs_spike_count_and_size:
                if_outputs_spike_count_and_size[layername] = SpikeCounter(spike_count=outputnp,
                                                                          tensor_sum=output_sum,
                                                                          size=output.numel(),
                                                                          binary=is_binary(output),
                                                                          sample_count=nb_sample)
            else:
                if_outputs_spike_count_and_size[layername].spike_count += outputnp
                if_outputs_spike_count_and_size[layername].tensor_sum += output_sum
                if_outputs_spike_count_and_size[layername].size += output.numel()
                if_outputs_spike_count_and_size[layername].binary &= is_binary(output)
                if_outputs_spike_count_and_size[layername].sample_count += nb_sample


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

        return is_module_sj, if_inputs_spike_count_and_size, if_outputs_spike_count_and_size

    def _compute_layer_spikerates(self,
                                  if_inputs_spike_count_and_size: dict[str, SpikeCounter],
                                  if_outputs_spike_count_and_size: dict[str, SpikeCounter],
                                  timesteps: int) -> tuple[dict[str, float],
                                                           dict[str, float],
                                                           dict[str, bool],
                                                           dict[str, bool],
                                                           dict[str, Number],
                                                           dict[str, Number]]:

        input_spikerates = {n: sc.spike_count / sc.size for n, sc in if_inputs_spike_count_and_size.items()}
        output_spikerates = {n: sc.spike_count / sc.size for n, sc in if_outputs_spike_count_and_size.items()}


        # Dict for the input counts of each layers divided  by the batch size
        input_counts = {n: timesteps * sc.spike_count / sc.sample_count
                        for n, sc in if_inputs_spike_count_and_size.items()}
        output_counts = {n: timesteps * sc.spike_count / sc.sample_count
                         for n, sc in if_outputs_spike_count_and_size.items()}

        input_is_binary = {n: sc.binary for n, sc in if_inputs_spike_count_and_size.items()}
        output_is_binary = {n: sc.binary for n, sc in if_outputs_spike_count_and_size.items()}

        return (input_spikerates,
                output_spikerates,
                input_is_binary,
                output_is_binary,
                input_counts,
                output_counts)

    def _compute_total_spikerate(self,
                                 if_inputs_spike_count_and_size: dict[str, SpikeCounter],
                                 if_outputs_spike_count_and_size: dict[str, SpikeCounter],
                                 timesteps: int,
                                 total_exclude_nonbinary: bool = True) -> tuple[float, float, float, float]:  # noqa: FBT001, FBT002
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
        input_total_spikerate = (sum(sc.spike_count for sc in filetered_if_inputs_spike_count_and_size.values())
                                         / sum(sc.size for sc in filetered_if_inputs_spike_count_and_size.values())
                                 if len(filetered_if_inputs_spike_count_and_size) > 0 else 0.0)
        output_total_spikerate = (sum(sc.spike_count for sc in filetered_if_outputs_spike_count_and_size.values())
                                         / sum(sc.size for sc in filetered_if_outputs_spike_count_and_size.values())
                                 if len(filetered_if_outputs_spike_count_and_size) > 0 else 0.0)

        # Special value account for the total count across the whole network
        input_total_count = sum(timesteps * sc.spike_count / sc.sample_count
                                        for sc in filetered_if_inputs_spike_count_and_size.values())
        output_total_count = sum(timesteps * sc.spike_count / sc.sample_count
                                         for sc in filetered_if_outputs_spike_count_and_size.values())
        return input_total_spikerate, output_total_spikerate, input_total_count, output_total_count

    @override
    def __call__(self,  # noqa: C901, PLR0912, PLR0915
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
        import spikingjelly.activation_based.functional as sjf  # type: ignore[import-untyped]
        import spikingjelly.activation_based.layer as sjl  # type: ignore[import-untyped]
        import spikingjelly.activation_based.neuron as sjn  # type: ignore[import-untyped]
        from qualia_codegen_core.graph.layers import TAddLayer, TSumLayer
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
                                    SNNAdd:  lambda *_: (TAddLayer, []),
                                    GlobalSumPool1d: lambda *_: (TSumLayer, [(-1,)]),
                                    GlobalSumPool2d: lambda *_: (TSumLayer, [(-2, -1)]),
                                    QuantizedConv1d: TorchModelGraph.MODULE_MAPPING[nn.Conv1d],
                                    QuantizedConv2d: TorchModelGraph.MODULE_MAPPING[nn.Conv2d],
                                    QuantizedBatchNorm1d: TorchModelGraph.MODULE_MAPPING[nn.BatchNorm1d],
                                    QuantizedBatchNorm2d: TorchModelGraph.MODULE_MAPPING[nn.BatchNorm2d],
                                    QuantizedLinear: TorchModelGraph.MODULE_MAPPING[nn.Linear],

                                    QuantizedIdentity: TorchModelGraph.MODULE_MAPPING[nn.Identity],
                                    qsjl.QuantizedLinear: TorchModelGraph.MODULE_MAPPING[nn.Linear],
                                    SNNQuantizedAdd:  lambda *_: (TAddLayer, []),
                                    qsjl1d.QuantizedBatchNorm1d: TorchModelGraph.MODULE_MAPPING[nn.BatchNorm1d],
                                    qsjl2d.QuantizedBatchNorm2d: TorchModelGraph.MODULE_MAPPING[nn.BatchNorm2d],
                                    qsjl1d.QuantizedConv1d: TorchModelGraph.MODULE_MAPPING[nn.Conv1d],
                                    qsjl2d.QuantizedConv2d: TorchModelGraph.MODULE_MAPPING[nn.Conv2d],
                                    qsjl1d.QuantizedMaxPool1d: TorchModelGraph.MODULE_MAPPING[nn.MaxPool1d],
                                    qsjl2d.QuantizedMaxPool2d: TorchModelGraph.MODULE_MAPPING[nn.MaxPool2d],
                                    qsjn.QuantizedIFNode: TorchModelGraph.MODULE_MAPPING[sjn.IFNode],
                                    qsjn.QuantizedLIFNode: TorchModelGraph.MODULE_MAPPING[sjn.LIFNode],
                                    QuantizedAdd:  lambda *_: (TAddLayer, []),
                                    QuantizedGlobalSumPool1d: lambda *_: (TSumLayer, [(-1,)]),
                                    QuantizedGlobalSumPool2d: lambda *_: (TSumLayer, [(-2, -1)]),
                                    }

        orig_step_mode: str = getattr(model, 'step_mode', '')
        if (getattr(model, 'is_snn', False) or isinstance(model, SNN)) and model.step_mode != 's':
            logger.warning("Setting model.step_mode to 's' for tracing by Qualia-CodeGen")
            sjf.set_step_mode(model, step_mode='s')

        modelgraph = TorchModelGraph(model).convert(custom_layers=custom_layers)
        if modelgraph is None:
            logger.error('Model graph conversion failed')
            return trainresult, model_conf

        logger.info('ModelGraph:\n%s', modelgraph)

        if getattr(model, 'step_mode', '') != orig_step_mode:
            logger.info("Reverting back to original step_mode='%s'", orig_step_mode)
            sjf.set_step_mode(model, step_mode=orig_step_mode)

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
            (is_module_sj,
             if_inputs_spike_count_and_size,
             if_outputs_spike_count_and_size) = self._record_spike_count(trainresult,
                                                                         framework=trainresult.framework,
                                                                         model=model,
                                                                         dataset=trainresult.testset)

            # SNN model post-processing
            for node in modelgraph.nodes:
                # Copy spikerate from IF activation layer to previous layer since it is the one energy is computed on
                # Also set previous layer to be computed as an SNN layer
                # And rename the previous layer to include the IF layer name
                if isinstance(node.layer, TIfLayer):
                    in_name = node.innodes[0].layer.name
                    new_in_name = in_name + node.layer.name

                    if_outputs_spike_count_and_size[new_in_name] = if_outputs_spike_count_and_size[node.layer.name]
                    # Rename previous layer
                    if_inputs_spike_count_and_size[new_in_name] = if_inputs_spike_count_and_size[in_name]
                    is_module_sj[new_in_name] = is_module_sj[in_name]
                    node.innodes[0].layer.name = new_in_name

                if isinstance(node.layer, TAddLayer):  # noqa: SIM102 if statements separated for clarity and documentation
                    # Special case for Add to handle SResNet, only for Add layer with full "binary" inputs
                    # Includes Add layers that may have had their input connected to a previous Add layer
                    # that switched to "binary" during this very same process.
                    if if_inputs_spike_count_and_size[node.layer.name].binary:
                        # If inputs are binary, output is defined as binary even with accumulated spikes
                        output_spikecounter = if_outputs_spike_count_and_size[node.layer.name]
                        output_spikecounter.spike_count = output_spikecounter.tensor_sum
                        output_spikecounter.binary = True
                        for outnode in node.outnodes:
                            # Check for special case of Add that can branch into another Add
                            # with multiple inputs that may or may not all be spikes
                            # If they are all spikes the "last" Add input would trigger the change to assuming binary input
                            if all(if_outputs_spike_count_and_size[outnodeinnode.layer.name].binary
                                   for outnodeinnode in outnode.innodes):
                                input_spikecounter = if_inputs_spike_count_and_size[outnode.layer.name]
                                # Non-binary integers translate to multiple spikes, so use sum of tensor
                                input_spikecounter.spike_count = input_spikecounter.tensor_sum
                                # But still assume input is spike
                                input_spikecounter.binary = True


            (input_spikerates,
             output_spikerates,
             input_is_binary,
             output_is_binary,
             input_counts,
             output_counts) = self._compute_layer_spikerates(if_inputs_spike_count_and_size,
                                                            if_outputs_spike_count_and_size,
                                                            timesteps=model.timesteps)

            # Compute total only after the post-processing steps as it changes the layer spikerates
            (input_spikerates['__TOTAL__'],
             output_spikerates['__TOTAL__'],
             input_counts['__TOTAL__'],
             output_counts['__TOTAL__']) = self._compute_total_spikerate(if_inputs_spike_count_and_size,
                                                                         if_outputs_spike_count_and_size,
                                                                         timesteps=model.timesteps,
                                                                         total_exclude_nonbinary=self._total_spikerate_exclude_nonbinary)

            ems = self._compute_model_energy_snn(modelgraph,
                                                 input_spikerates,
                                                 output_spikerates,
                                                 input_is_binary,
                                                 output_is_binary,
                                                 input_counts,
                                                 output_counts,
                                                 is_module_sj,
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
