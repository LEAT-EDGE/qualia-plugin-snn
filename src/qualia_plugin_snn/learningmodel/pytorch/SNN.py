"""Contains the base class for spiking neural network models."""

from __future__ import annotations

import logging

import spikingjelly.activation_based.neuron as sj  # type: ignore[import-untyped]
from qualia_core.learningmodel.pytorch.LearningModelPyTorch import LearningModelPyTorch
from qualia_core.typing import TYPE_CHECKING

import qualia_plugin_snn.learningmodel.pytorch.layers.quantized_SNN_layers as qsj
from qualia_plugin_snn.learningmodel.pytorch.layers import CustomNode

if TYPE_CHECKING:
    from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig  # noqa: TCH002
    from qualia_core.typing import RecursiveConfigDict, RecursiveConfigUnion
    from torch import nn  # noqa: TCH002

logger = logging.getLogger(__name__)

class SNN(LearningModelPyTorch):
    """Base class for spiking neural network models to inherit from."""

    timesteps: int #: Number of timesteps
    step_mode: str #: SpikingJelly ``step_mode``
    is_snn: bool = True #: Always ``True`` in case of spiking neural networks

    def __select_neuron(self, neuron: RecursiveConfigDict) -> type[nn.Module]:
        """Select a spiking neuron class from a kind specified in configuration file.

        The class will be looked up by its name in the following modules successively, first match is used:

            #. :mod:`spikingjelly.activation_based.neuron`
            #. :mod:`qualia_plugin_snn.learningmodel.pytorch.layers.CustomNode`
            #. :mod:`qualia_plugin_snn.learningmodel.pytorch.quantized_SNN_layers`

        :meta public:
        :param neuron: A spiking neuron configuration dict
        :return: The class corresponding to the ``kind`` specified in the neuron configuration dict
        :raise ValueError: When the ``'kind'`` key cannot be found in the ``neuron`` dict
        :raise TypeError: When the value associated to ``kind`` is not a string
        :raise AttributeError: When the class for ``kind`` cannot be looked up
        """
        # SpikeNeuron Selection
        if 'kind' not in neuron:
            logger.error('`params.neuron.kind` is required')
            raise ValueError
        if not isinstance(neuron['kind'], str):
            logger.error('`params.neuron.kind` must be a string, got: %s', type(neuron['kind']))
            raise TypeError
        neurons_kind = neuron['kind']
        neuron_type: type[nn.Module]

        try:
            neuron_type = getattr(sj, neurons_kind)
        except AttributeError:
            logger.info("Module 'spikingjelly.activation_based.neuron' has no attribute %s",  neurons_kind)
            logger.info('Checking learningmodel/pytorch/layers/CustomNode')
            try:
                neuron_type = getattr(CustomNode, neurons_kind)
            except AttributeError:
                logger.info("Module 'learningmodel.pytorch.layers.CustomNode' has no attribute %s", neurons_kind)
                logger.info('Checking learningmodel/pytorch/Quantized_SNN_layers')
                try:
                    neuron_type = getattr(qsj, neurons_kind)
                except AttributeError:
                    logger.exception("Module 'learningmodel.pytorch.Quantized_SNN_layers' has no attribute %s", neurons_kind)
                    raise

        return neuron_type

    def __extract_neuron_params(self, neuron: RecursiveConfigDict) -> dict[str, RecursiveConfigUnion | None]:
        """Extract params from the given ``neuron`` configuration dict, also convert ``v_reset``.

        ``v_reset`` is specified as either a float (hard-reset) or ``false`` (soft-reset) in the TOML configuration since there is
        no ``None`` equivalent in TOML. However SpikingJelly neurons use ``None`` to signify soft-reset, so convert
        ``v_reset=false`` to ``v_reset=None``.

        :meta public:
        :param neuron: The spiking neuron configuration dict
        :return: The ``params`` dict for the given neuron configuration dict with converted ``v_reset``
        :raise TypeError: When the value associated to ``params`` is not a dict
        """
        # Extract neuron params
        neuron_params = neuron.get('params', {})
        if not isinstance(neuron_params, dict):
            logger.error('`params.neuron.params` must be a dict, got: %s', type(neuron_params))
            raise TypeError

        # Replace v_reset = False with v_reset = None for soft reset of SpikingJelly
        filtered_neuron_params: dict[str, RecursiveConfigUnion | None] = {
                k: v for k, v in neuron_params.items()
                if k != 'v_reset'
                }

        if 'v_reset' in neuron_params:
            if neuron_params['v_reset'] is False:
                filtered_neuron_params['v_reset'] = None
            else:
                filtered_neuron_params['v_reset'] = neuron_params['v_reset']

        return filtered_neuron_params

    def create_neuron(self, quant_params: QuantizationConfig | None = None) -> nn.Module:
        """Instanciate a spiking neuron from the kind and params found in ``neuron`` of :meth:``__init__``.

        :param quant_params: Optional quantization configuration dict in case of quantized network, see
                             :class:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer`
        :return: A spiking neuron instance
        """
        if quant_params is not None:
            return self.__neuron_type(**self.__neuron_params, quant_params=quant_params)
        return self.__neuron_type(**self.__neuron_params)

    def __init__(self,
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 timesteps: int,
                 neuron: RecursiveConfigDict | None) -> None:
        """Construct :class:`SNN`.

        :attr:`step_mode` is extracted from the neuron configuration dict ``neuron``.
        :meth:`__select_neuron` and :meth:`__extract_neuron_params` are called to prepare the neuron for instanciation with
        :meth:`create_neuron` which should be used in derived classes to instanciate the spiking neurons.

        Below is an example of a neuron configuration dict for SpikingJelly's multi-step
        :class:`spikingjelly.activation_based.neuron.IFNode` soft-reset with threshold at 1.0.

        * Python

        .. code-block:: python

            neuron = {
                'kind': 'IFNode',
                'params': {
                    'v_threshold': 1.0,
                    'v_reset': False,
                    'step_mode': 'm',
                    }
            }

        * TOML

        .. code-block:: toml

            [[model]]
            params.neuron.kind = 'IFNode'
            params.neuron.params.v_threshold = 1.0
            params.neuron.params.v_reset = false
            params.neuron.params.step_mode = 'm'

        :param input_shape: Input shape passed to :class:`qualia_core.learningmodel.pytorch.LearningModel.LearningModel`
                            with ``timesteps`` prepended
        :param output_shape: Output shape passed to :class:`qualia_core.learningmodel.pytorch.LearningModel.LearningModel`
        :param timesteps: Number of timesteps
        :param neuron: Spiking neuron configuration dict
        :raise ValueError: When ``neuron`` is ``None`` or empty
        :raise TypeError: When the value associated to ``params.step_mode`` is not a string
        """
        super().__init__(input_shape=(timesteps, *tuple(input_shape)), output_shape=output_shape)
        self.timesteps = timesteps

        if not neuron:
            logger.error('`params.neuron` is required')
            raise ValueError

        # Find neuron class
        self.__neuron_type = self.__select_neuron(neuron)

        self.__neuron_params = self.__extract_neuron_params(neuron)

        # Extract step_mode
        step_mode = self.__neuron_params.get('step_mode', 's')
        if not isinstance(step_mode, str):
            logger.error('`params.neuron.params.step_mode` must be a string, got: %s', type(step_mode))
            raise TypeError
        self.step_mode = step_mode

