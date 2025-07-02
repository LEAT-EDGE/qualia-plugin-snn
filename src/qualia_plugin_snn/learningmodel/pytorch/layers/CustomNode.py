"""ATIF-u: spiking neuron with learnable quantization steps.

Author: Andrea Castagetti <Andrea.CASTAGNETTI@univ-cotedazur.fr>
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal, cast

import torch
from qualia_core.typing import TYPE_CHECKING
from spikingjelly.activation_based.neuron import BaseNode  # type: ignore[import-untyped]
from torch import nn

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

def heaviside(x: torch.Tensor) -> torch.Tensor:
    """Heaviside function.

    :param x: Input tensor
    :return: Boolean tensor of the same dimension as `x` where each element is ``True`` if the element in `x` is greater than or
             equal to 0, ``False`` otherwise
    """
    return (x >= 0).to(x)

class SpikeFunctionSigmoid(torch.autograd.Function):
    """Spike functions with surrogate bkw gradient."""

    @staticmethod
    @override
    def forward(ctx: FunctionCtx, *args: torch.Tensor, **_: Any) -> torch.Tensor:
        """Forward of :func:`heaviside` function.

        :param ctx: A context object used to save tensors for :meth:`backward`
        :param args: Tuple of 2 tensors for ``x`` and ``alpha``, respectively, saved in ``ctx`` for backward pass
        :param _: Unused
        :return: Tensor of :func:`heaviside` applied over ``x``.
        """
        x, alpha = args
        if x.requires_grad:
            ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    @override
    def backward(ctx: torch.autograd.Function, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        """Backward pass of surrogate gradient using :meth:`torch.Tensor.sigmoid_` function.

        :param ctx: Context to restore the ``x`` and ``alpha`` tensors from
        :param grad_outputs: Output tensor from the computation of the :meth:`forward` pass
        :return: A tuple of Tensor and None with the first element being the computed gradient for ``x`` or None if there is no
                 gradient to compute and the second element a placeholder for the gradient of ``alpha``.
        """
        grad_x = None
        grad_output = grad_outputs[0]
        if ctx.needs_input_grad[0]:
            x, alpha = cast('tuple[torch.Tensor, torch.Tensor]', ctx.saved_tensors) # Couple of tensors saved in forward()
            device = x.device
            sgax = ((x * alpha.to(device=device)).sigmoid_()).to(device=device)
            grad_x = (grad_output.to(device=device)
                      * (1. - sgax.to(device=device))
                      * sgax.to(device=device)
                      * alpha.to(device=device))
        return grad_x, None

    @classmethod
    @override
    def apply(cls, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply heaviside activation with sigmoid surrogate gradient.

        :param args: Input tensor
        :param kwargs: Unused
        :return: Output tensor
        """
        return cast('torch.Tensor', super().apply(*args, **kwargs))  # type: ignore[no-untyped-call]

class IFSRL(nn.Module):
    """IFSRL: Integrate and Fire soft-reset with learnable Vth and activation scaling."""

    v: torch.Tensor

    def __init__(self,
                 v_threshold: float = 1.0,
                 vth_init_l: float = 0.8,
                 vth_init_h: float = 1.,
                 alpha: float = 1.,
                 device: str = 'cpu') -> None:
        """Construct :class:`IFSRL`.

        :param v_threshold: Factor to apply to the uniform initialization bounds
        :param vth_init_l: Lower bound for uniform initialization of threshold Tensor
        :param vth_init_h: Higher bound for uniform initialization of threshold Tensor
        :param alpha: Sigmoig surrogate scale factor
        :param device: Device to run the computation on
        """
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()
        self.v_threshold = v_threshold
        self.alpha = alpha
        self.device = device

        self.vp_th = torch.nn.Parameter(torch.ones(1).to(self.device), requires_grad=True)
        self.v = torch.zeros(1, device=device)

        with torch.no_grad():
            _ = nn.init.uniform_(self.vp_th,
                                 a=vth_init_l * self.v_threshold,
                                 b=vth_init_h * self.v_threshold)

    def get_coeffs(self) -> torch.Tensor:
        """Return the Tensor of threshold :attr:`vp_th`.

        :return: Tensor of threshold :attr:`vp_th`
        """
        return self.vp_th

    def set_coeffs(self, vp_th: torch.Tensor) -> None:
        """Replace the Tensor of threshold :attr:`vp_th`.

        :param vp_th: New Tensor of threshold to replace :attr:`vp_th`
        """
        _ = self.vp_th.copy_(vp_th)

    def reset(self) -> None:
        """Reset potential to 0."""
        _ = self.v.zero_() # midrise quantizer

    def ifsrl_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate-and-Fire soft-reset neuron with learnable threshold.

        :param x: Input tensor
        :return: Output tensor
        """
        # Primary membrane charge
        self.v = self.v + x

        # Fire
        q = (self.v - self.vp_th)
        z = SpikeFunctionSigmoid.apply(q, self.alpha * torch.ones(1).to(self.device)).float()

        # Soft-Reset
        self.v = (1. - z) * self.v + z * (self.v - self.vp_th)

        return z

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward of :meth:`ifsrl_fn`.

        :param input: Input tensor
        :return: Output tensor
        """
        x = input
        return self.ifsrl_fn(x)


class ATIF(BaseNode):  # type: ignore[misc]
    """IFSRLSJ: Integrate and Fire soft-reset with learnable Vth and activation scaling, based on spikingjelly."""

    v: torch.Tensor

    def __init__(self,
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
        """
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__(v_threshold=v_threshold,
                         v_reset=None,
                         surrogate_function=SpikeFunctionSigmoid.apply,
                         step_mode='s') # init the base class
        self.device = device
        self.alpha = alpha
        self.v_threshold = torch.nn.Parameter(torch.ones(1).to(self.device) * v_threshold, requires_grad=True)

        self.v = torch.zeros(1, device=device)

        with torch.no_grad():
            _= nn.init.uniform_(self.v_threshold,
                             a=vth_init_l * v_threshold,
                             b=vth_init_h * v_threshold)

    @property
    @override
    def supported_backends(self) -> tuple[Literal['torch']]:
        """Supported step_mode and backend.

        Only single-step mode with torch backend is supported.

        :return: Tuple of ``'torch'`` if :attr:`step_mode` is ``'s'``
        :raise ValueError: When :attr:`step_mode` is not ``'s'``
        """
        if self.step_mode == 's':
            return ('torch',)
        logger.error("Only step_mode='s' is supported, current step_mode='%s'", self.step_mode)
        raise ValueError

    def get_coeffs(self) -> torch.Tensor:
        """Return the Tensor of threshold :attr:`v_threshold`.

        :return: Tensor of threshold :attr:`v_threshold`
        """
        return self.v_threshold

    def set_coeffs(self, v_threshold: torch.Tensor) -> None:
        """Replace the Tensor of threshold :attr:`v_threshold`.

        :param v_threshold: New Tensor of threshold to replace :attr:`v_threshold`
        """
        _ = self.v_threshold.copy_(v_threshold)

    def ifsrl_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate-and-Fire soft-reset neuron with learnable threshold.

        :param x: Input tensor
        :return: Output tensor
        """
        # Primary membrane charge
        self.v_float_to_tensor(x)
        self.v = self.v + x

        # Fire
        q = (self.v - self.v_threshold)
        z = SpikeFunctionSigmoid.apply(q, self.alpha * torch.ones(1).to(self.device)).float()

        # Soft-Reset
        self.v = (1. - z) * self.v + z * (self.v - self.v_threshold)

        return z * self.get_coeffs()

    @override
    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step mode forward of ATIF.

        Calls :meth:`ifsrl_fn`.

        :param x: Input tensor
        :return: Output tensor
        """
        return self.ifsrl_fn(x)
