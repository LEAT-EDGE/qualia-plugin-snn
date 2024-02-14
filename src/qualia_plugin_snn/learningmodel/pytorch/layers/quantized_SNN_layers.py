"""Contains quantized spiking neuron implementations."""

from __future__ import annotations

import sys
from typing import Literal

import torch
from qualia_core.learningmodel.pytorch.layers.QuantizedLayer import QuantizedLayer, QuantizerActProtocol, QuantizerInputProtocol
from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig, Quantizer, update_params
from spikingjelly.activation_based.neuron import IFNode, LIFNode  # type: ignore[import-untyped]

from .CustomNode import ATIF, SpikeFunctionSigmoid

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class QuantizedLIFNode(LIFNode,  # type: ignore[misc]
                       QuantizerInputProtocol,
                       QuantizerActProtocol,
                       QuantizedLayer):
    """Quantized variant of SpikingJelly's :class:`spikingjelly.activation_based.neuron.LIFNode`.

    Hyperparameters ``v_threshold``, ``v_reset`` and ``tau`` are quantized as well as membrane potential ``v``.
    """

    v: torch.Tensor
    v_threshold: float
    v_reset: float | None
    tau: float

    def __init__(self,  # noqa: PLR0913
                 quant_params: QuantizationConfig,
                 tau: float = 2.,
                 decay_input: bool = True,  # noqa: FBT001, FBT002
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 detach_reset: bool = False,  # noqa: FBT002, FBT001
                 step_mode: str = 's',
                 backend: str = 'torch') -> None:
        """Construct :class:`QuantizedLIFNode`.

        For more information about spiking neuron parameters, see:
        :meth:`spikingjelly.activation_based.neuron.LIFNode.__init__`

        :param tau: Membrane time constant
        :param decay_input: Whether the input will decay
        :param v_threshold: Threshold of this neurons layer
        :param v_reset: Reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
                        after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :param detach_reset: Whether detach the computation graph of reset in backward
        :param step_mode: The step mode, which can be `s` (single-step) or `m` (multi-step)
        :param backend: backend fot this neurons layer, only 'torch' is supported
        :param quant_params: Quantization configuration dict, see :class:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer`
        """
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__(tau=tau,
                         decay_input=decay_input,
                         v_threshold=v_threshold,
                         v_reset=v_reset,
                         detach_reset=detach_reset,
                         step_mode=step_mode,
                         backend=backend)
        # Create the quantizer instance
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        # Does not work like weights since it's dynamic so need to keep track of global max, don't use 'input' since it can be
        # skipped so use 'act' type
        quant_params_v = update_params(tensor_type='act', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_v = Quantizer(**quant_params_v)
        self.quantizer_act = Quantizer(**quant_params_act)

    @property
    @override  # type: ignore[misc]
    def supported_backends(self) -> tuple[Literal['torch']]:
        """Supported step_mode and backend.

        Only torch backend is supported.

        :return: Tuple of ``'torch'``
        :raise ValueError: When :attr:`step_mode` is not ``'s'`` or ``'m'``
        """
        if self.step_mode in ['s', 'm']:
            return ('torch',)
        raise ValueError(self.step_mode)

    @override  # type: ignore[misc]
    def neuronal_charge(self, x: torch.Tensor) -> None:
        """Quantized :meth:`spikingjelly.activation_based.neuron.LIFNode.neuronal_charge`.

        Membrane potential and hyperparameters are quantized before and after computation using :meth:`quantize_v_and_hyperparams`.

        :param x: Input tensor
        """
        self.quantize_v_and_hyperparams()
        super().neuronal_charge(x)
        self.quantize_v_and_hyperparams()

    @override  # type: ignore[misc]
    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantized :meth:`spikingjelly.activation_based.neuron.LIFNode.single_step_forward`.

        Input is (optionally) quantized.
        Membrane potential and hyperparameters are quantized before and after computation using :meth:`quantize_v_and_hyperparams`.

        :param x: Input tensor
        """
        self.v_float_to_tensor(x)
        x = self.quantizer_input(x)

        spike: torch.Tensor

        if self.training:
            self.neuronal_charge(x)
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
        else:
            self.quantize_v_and_hyperparams()
            if self.v_reset is None:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x,
                                                                                             self.v,
                                                                                             self.v_threshold,
                                                                                             self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x,
                                                                                                self.v,
                                                                                                self.v_threshold,
                                                                                                self.tau)
            elif self.decay_input:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(x,
                                                                                         self.v,
                                                                                         self.v_threshold,
                                                                                         self.v_reset,
                                                                                         self.tau)
            else:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(x,
                                                                                            self.v,
                                                                                            self.v_threshold,
                                                                                            self.v_reset,
                                                                                            self.tau)
            self.quantize_v_and_hyperparams()

        return self.quantizer_act(spike)

    @override  # type: ignore[misc]
    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Implement multi-step as loop over single-step for quantized neurons, inefficient but at least it works.

        :param x_seq: Input tensor with timesteps
        :return: Output tensor with timesteps
        """
        out = torch.zeros_like(x_seq)
        for t, d in enumerate(x_seq):
            out[t] = self.single_step_forward(d)
        return out

    @property
    def reciprocal_tau(self) -> float:
        """Return 1 / tau.

        :return: 1 / tau
        """
        return 1 / self.tau

    def quantize_v_and_hyperparams(self) -> None:
        """Quantize potential and hyperparameters (v_threshold, tau, v_reset) in-place with the same quantizer at the same time.

        tau is not quantized directly, instead quantize reciprocal_tau (1 / tau) because this is what is used during inference to
        avoid division.
        """
        self.v, hyperparams = self.quantizer_v(self.v,
                                               bias_tensor=self.get_hyperparams_tensor(device=self.v.device, dtype=self.v.dtype))
        self.v_threshold = hyperparams[0].item()
        self.tau = 1 / hyperparams[1].item() # Turn back reciprocal_tau int tau
        if self.v_reset is not None:
            self.v_reset = hyperparams[2].item()

    def get_hyperparams_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Pack v_threshold, reciprocal_tau and optionally v_reset into the same Tensor.

        :param device: Device to create the tensor on
        :param dtype: Data type for the created tensor
        :return: New tensor with hyperparemeters concatenated
        """
        if self.v_reset is None:
            return torch.tensor([self.v_threshold, self.reciprocal_tau], device=device, dtype=dtype)
        return torch.tensor([self.v_threshold, self.reciprocal_tau, self.v_reset], device=device, dtype=dtype)

    @property
    @override
    def weights_q(self) -> int | None:
        """Number of fractional part bits for the membrane potential and hyperparameters in case of fixed-point quantization.

        See :meth:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer.fractional_bits`.

        :return: Fractional part bits for the membrane potential and hyperparameters or ``None`` if not applicable.
        """
        return self.quantizer_v.fractional_bits

    @property
    @override
    def weights_round_mode(self) -> str | None:
        return self.quantizer_v.roundtype

class QuantizedIFNode(IFNode,  # type: ignore[misc]
                      QuantizerInputProtocol,
                      QuantizerActProtocol,
                      QuantizedLayer):
    """Quantized variant of SpikingJelly's :class:`spikingjelly.activation_based.neuron.IFNode`.

    Hyperparameters ``v_threshold`` and ``v_reset`` are quantized as well as membrane potential ``v``.
    """

    v: torch.Tensor
    v_threshold: float
    v_reset: float | None

    def __init__(self,  # noqa: PLR0913
                 quant_params: QuantizationConfig,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 detach_reset: bool = False,  # noqa: FBT001, FBT002
                 step_mode: str = 's',
                 backend: str = 'torch') -> None:
        """Construct :class:`QuantizedIFNode`.

        For more information about spiking neuron parameters, see:
        :meth:`spikingjelly.activation_based.neuron.IFNode.__init__`

        :param v_threshold: Threshold of this neurons layer
        :param v_reset: Reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
                        after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :param detach_reset: Whether detach the computation graph of reset in backward
        :param step_mode: The step mode, which can be `s` (single-step) or `m` (multi-step)
        :param backend: backend fot this neurons layer, only 'torch' is supported
        :param quant_params: Quantization configuration dict, see :class:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer`
        """
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__(v_threshold=v_threshold,
                         v_reset=v_reset,
                         detach_reset=detach_reset,
                         step_mode=step_mode,
                         backend=backend)
        # Create the quantizer instance
        quant_params_input = update_params(tensor_type = 'input', quant_params = quant_params)
        # Does not work like weights since it's dynamic so need to keep track of global max, don't use 'input' since it can be
        # skipped so use 'act' type
        quant_params_v = update_params(tensor_type='act', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_v = Quantizer(**quant_params_v)
        self.quantizer_act = Quantizer(**quant_params_act)

    @property
    @override  # type: ignore[misc]
    def supported_backends(self) -> tuple[Literal['torch']]:
        """Supported step_mode and backend.

        Only torch backend is supported.

        :return: Tuple of ``'torch'``
        :raise ValueError: When :attr:`step_mode` is not ``'s'`` or ``'m'``
        """
        if self.step_mode in ['s', 'm']:
            return ('torch',)
        raise ValueError(self.step_mode)

    @override  # type: ignore[misc]
    def neuronal_charge(self, x: torch.Tensor) -> None:
        """Quantized :meth:`spikingjelly.activation_based.neuron.IFNode.neuronal_charge`.

        Membrane potential and hyperparameters are quantized before and after computation using :meth:`quantize_v_and_hyperparams`.

        :param x: Input tensor
        """
        self.quantize_v_and_hyperparams()
        super().neuronal_charge(x)
        self.quantize_v_and_hyperparams()

    @override  # type: ignore[misc]
    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantized :meth:`spikingjelly.activation_based.neuron.IFNode.single_step_forward`.

        Input is (optionally) quantized.
        Membrane potential and hyperparameters are quantized before and after computation using :meth:`quantize_v_and_hyperparams`.

        :param x: Input tensor
        """
        self.v_float_to_tensor(x)
        x = self.quantizer_input(x)

        spike: torch.Tensor

        if self.training:
            self.neuronal_charge(x)
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
        else:
            self.quantize_v_and_hyperparams()
            if self.v_reset is None:
                spike, self.v = self.jit_eval_single_step_forward_soft_reset(x, self.v, self.v_threshold)
            else:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset(x, self.v, self.v_threshold, self.v_reset)
            self.quantize_v_and_hyperparams()

        return self.quantizer_act(spike)

    @override  # type: ignore[misc]
    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Implement multi-step as loop over single-step for quantized neurons, inefficient but at least it works.

        :param x_seq: Input tensor with timesteps
        :return: Output tensor with timesteps
        """
        out = torch.zeros_like(x_seq)
        for t, d in enumerate(x_seq):
            out[t] = self.single_step_forward(d)
        return out

    def quantize_v_and_hyperparams(self) -> None:
        """Quantize potential and hyperparameters (v_threshold, v_reset) in-place with the same quantizer at the same time."""
        self.v, hyperparams = self.quantizer_v(self.v,
                                               bias_tensor=self.get_hyperparams_tensor(device=self.v.device, dtype=self.v.dtype))
        self.v_threshold = hyperparams[0].item()
        if self.v_reset is not None:
            self.v_reset = hyperparams[1].item()

    def get_hyperparams_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Pack v_threshold and optionally v_reset into the same Tensor.

        :param device: Device to create the tensor on
        :param dtype: Data type for the created tensor
        :return: New tensor with hyperparemeters concatenated
        """
        if self.v_reset is None:
            return torch.tensor([self.v_threshold], device=device, dtype=dtype)
        return torch.tensor([self.v_threshold, self.v_reset], device=device, dtype=dtype)

    @property
    @override
    def weights_q(self) -> int | None:
        """Number of fractional part bits for the membrane potential and hyperparameters in case of fixed-point quantization.

        See :meth:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer.fractional_bits`.

        :return: Fractional part bits for the membrane potential and hyperparameters or ``None`` if not applicable.
        """
        return self.quantizer_v.fractional_bits

    @property
    @override
    def weights_round_mode(self) -> str | None:
        return self.quantizer_v.roundtype

class QuantizedATIF(ATIF,
                    QuantizerInputProtocol,
                    QuantizerActProtocol,
                    QuantizedLayer):
    """Quantized Integrate and Fire soft-reset with learnable Vth and activation scaling, based on spikingjelly."""

    def __init__(self,  # noqa: PLR0913
                 quant_params: QuantizationConfig,
                 v_threshold: float = 1.0,
                 vth_init_l: float = 0.8,
                 vth_init_h: float = 1.,
                 alpha: float = 1.,
                 device: str = 'cpu') -> None:
        """Construct :class:`ATIF`.

        :param v_threshold: Factor to apply to the uniform initialization bounds
        :param vth_init_l: Lower bound for uniform initialization of threshold Tensor
        :param vth_init_h: Higher bound for uniform initialization of threshold Tensor
        :param alpha: Sigmoig surrogate scale factor
        :param device: Device to run the computation on
        :param quant_params: Quantization configuration dict, see :class:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer`
        """
        super().__init__(v_threshold=v_threshold,
                         vth_init_l=vth_init_l,
                         vth_init_h=vth_init_h,
                         alpha=alpha,
                         device=device)
        # Create the quantizer instance
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        # Does not work like weights since it's dynamic so need to keep track of global max, don't use 'input' since it can be
        # skipped so use 'act' type
        quant_params_v = update_params(tensor_type='act', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_v = Quantizer(**quant_params_v)
        self.quantizer_act = Quantizer(**quant_params_act)

    @override
    def ifsrl_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Quantized :meth:`qualia_plugin_snn.learningmodel.pytorch.layers.CustomNode.ATIF.ifsrl_fn`.

        Input is quantized.
        Membrane potential is quantized before and after computation.
        Threshold is quantized by :meth:`get_coeffs`.

        :param x: Input tensor
        :return: Output tensor
        """
        # Primary membrane charge
        self.v_float_to_tensor(x)
        x = self.quantizer_input(x)
        self.v = self.quantizer_v(self.v)
        self.v = self.v + x
        self.v = self.quantizer_v(self.v)

        # Fire
        q = self.quantizer_v(self.v - self.vp_th)

        z = SpikeFunctionSigmoid.apply(q, self.alpha * torch.ones(1).to(self.device)).float()

        # Soft-Reset
        self.v = (1. - z) * self.v + z * (self.v - self.vp_th)
        self.v = self.quantizer_v(self.v)
        return self.quantizer_act(z * self.get_coeffs())

    @override
    def get_coeffs(self) -> torch.Tensor:
        """Return the quantized Tensor of threshold :attr:`v_threshold`.

        :return: Quantized Tensor of threshold :attr:`v_threshold`
        """
        return self.quantizer_v(self.v_threshold)

    @override
    def set_coeffs(self, v_threshold: torch.Tensor) -> None:
        """Quantized and replace the Tensor of threshold :attr:`v_threshold`.

        :param v_threshold: New Tensor of quantized threshold to replace :attr:`v_threshold`
        """
        _ = self.v_threshold.copy_(self.quantizer_v(v_threshold))

    @property
    @override
    def weights_q(self) -> int | None:
        """Number of fractional part bits for the membrane potential and hyperparameters in case of fixed-point quantization.

        See :meth:`qualia_core.learningmodel.pytorch.Quantizer.Quantizer.fractional_bits`.

        :return: Fractional part bits for the membrane potential and hyperparameters or ``None`` if not applicable.
        """
        return self.quantizer_v.fractional_bits

    @property
    @override
    def weights_round_mode(self) -> str | None:
        return self.quantizer_v.roundtype
