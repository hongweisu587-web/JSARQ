import torch
import torch.nn as nn
from .quant_config import Quantize_config, QuantizeModel_config
from typing import Union
import torch.nn.functional as F
from torch.autograd import Function

class Basic2DquantLayer(nn.Module):
    """
    A basic 2D quantization layer for PyTorch models.
    This layer supports both symmetric and asymmetric quantization, with configurable quantization bits,
    signed/unsigned quantization, and optional channel-wise quantization for weights. It maintains running
    statistics for activation quantization and supports initialization and momentum-based updates of quantization
    parameters.
    Args:
        quantize_config (Quantize_config): Configuration object specifying quantization parameters such as
            number of bits, signed/unsigned, symmetric/asymmetric, and user type ("weight" or "activation").
        delta (float, optional): Initial quantization scale (step size). Default is 1.
        zero_point (float, optional): Initial zero point for quantization. Default is 0.
    Attributes:
        quantize_config (Quantize_config): Stores the quantization configuration.
        quant_bits (int): Number of bits for quantization.
        sign (bool): Whether to use signed quantization.
        sym (bool): Whether to use symmetric quantization.
        delta (torch.Tensor): Quantization scale (step size).
        zero_point (torch.Tensor): Zero point for quantization.
        init_flag (bool): Indicates if the layer has been initialized.
        running_stat (bool): Indicates if running statistics are being used for activation quantization.
        x_min (nn.Parameter): Minimum value observed (per-channel or global).
        x_max (nn.Parameter): Maximum value observed (per-channel or global).
    Methods:
        set_init_state(flag=True): Set the initialization flag.
        set_running_stat(flag): Enable or disable running statistics for activation quantization.
        forward(x): Quantize and dequantize the input tensor `x`.
        Init_basicinfo(x): Initialize quantization parameters based on input tensor `x`.
        init_quantization_scale(x, channel_wise=False): Initialize quantization scale and zero point.
        get_tensor_maxmin(x, type="max"): Get per-channel or global max/min values from tensor `x`.
        channel_wise_reshape(value, shape): Reshape per-channel statistics to match input tensor shape.
        set_param(): Compute quantization scale (delta) and zero point based on current min/max.
        quantize(x, delta, zero_point): Quantize input tensor `x` using given scale and zero point.
        dequantize(x, delta, zero_point): Dequantize quantized tensor `x` using given scale and zero point.
        act_momentum_update(x, act_range_momentum=0.95): Update running min/max statistics for activations.
    Note:
        - This layer assumes the existence of a custom `Round` function for STE rounding.
        - The `Quantize_config` class must provide attributes: `quant_bits`, `sign`, `sym`, and `user`.
    """
    def __init__(self,quantize_config: Quantize_config, delta=1, zero_point=0,):
        super().__init__()
        self.quantize_config = quantize_config
        self.quant_bits = quantize_config.quant_bits
        self.sign = quantize_config.sign
        self.sym = quantize_config.sym
        self.delta, self.zero_point = torch.tensor(delta, dtype=float),torch.tensor(zero_point, dtype=float)
        self.init_flag = False
        self.running_stat = False
        self.x_min, self.x_max =  nn.Parameter(torch.tensor(-0.5, dtype=float)), nn.Parameter(torch.tensor(0.5, dtype=float))
        # Dormant by default so all existing checkpoints/configurations retain
        # the exact uniform-quantizer behaviour.  Selective PLQ uses one
        # learnable symmetric breakpoint and allocates the 16 A4 codes as
        # left/central/right = 4/8/4.
        self.raw_breakpoint = nn.Parameter(
            torch.tensor(0.0, dtype=float), requires_grad=False
        )
        self.register_buffer(
            "piecewise_enabled", torch.tensor(False, dtype=torch.bool)
        )
        self.register_buffer(
            "piecewise_code_allocation",
            torch.tensor((4, 8, 4), dtype=torch.int64),
        )
        self.register_buffer(
            "piecewise_version", torch.tensor(1, dtype=torch.int64)
        )
        self._piecewise_runtime_enabled = False
        self._piecewise_runtime_code_allocation = (4, 8, 4)
        # Native reference DOBI is opt-in runtime state.  It deliberately is
        # not checkpointed so legacy experiments keep their original affine
        # min/max initialization and loading behaviour.
        self.channelwise_axis = None
        self.channelwise_role = ""
        self.reference_dobi_enabled = False
        self.reference_dobi_update_mode = "disabled"
        self.reference_dobi_sample_elements = 262144
        self.reference_dobi_candidates = (
            (0.0, 1.0),
            (0.0001, 0.9999),
            (0.0005, 0.9995),
            (0.001, 0.999),
            (0.002, 0.998),
        )
        self.reference_dobi_momentum = 0.95
        self.reset_reference_dobi_statistics()

    def set_init_state(self, flag=True):
        self.init_flag = flag

    def set_running_stat(self, flag):
        self.running_stat = flag

    @torch.no_grad()
    def project_range_(self, minimum_gap: float = 1.0e-6) -> int:
        """Project learned bounds back onto the valid ordered-range set.

        ``x_min`` and ``x_max`` are optimized as independent parameters during
        reconstruction.  Adam can therefore move individual channels across
        one another.  Preserve the learned centre and span while restoring the
        invariant required by asymmetric quantization.
        """

        minimum_gap = float(minimum_gap)
        if minimum_gap <= 0.0:
            raise ValueError("minimum_gap must be positive")
        if not bool(torch.isfinite(self.x_min).all()):
            raise FloatingPointError("quantizer x_min contains NaN or Inf")
        if not bool(torch.isfinite(self.x_max).all()):
            raise FloatingPointError("quantizer x_max contains NaN or Inf")

        gap = self.x_max - self.x_min
        repaired = int((gap < minimum_gap).sum().item())

        lower = torch.minimum(self.x_min, self.x_max)
        upper = torch.maximum(self.x_min, self.x_max)
        center = (lower + upper) * 0.5
        half_span = (upper - lower) * 0.5
        minimum_half_span = torch.maximum(
            center.abs() * 1.0e-6,
            torch.full_like(center, minimum_gap * 0.5),
        )
        half_span = torch.maximum(half_span, minimum_half_span)

        self.x_min.copy_(center - half_span)
        self.x_max.copy_(center + half_span)
        if not bool((self.x_max > self.x_min).all()):
            raise RuntimeError("failed to project quantizer range")
        return repaired

    @property
    def piecewise_runtime_enabled(self):
        return bool(self._piecewise_runtime_enabled)

    def sync_piecewise_runtime_state(self):
        self._piecewise_runtime_enabled = bool(self.piecewise_enabled.item())
        self._piecewise_runtime_code_allocation = tuple(
            int(value)
            for value in self.piecewise_code_allocation.detach().cpu().tolist()
        )

    def _piecewise_radius(self):
        eps = torch.finfo(self.x_min.dtype).eps
        return torch.minimum((-self.x_min).clamp_min(eps), self.x_max.clamp_min(eps))

    def piecewise_breakpoint(self):
        radius = self._piecewise_radius()
        eps = torch.finfo(radius.dtype).eps
        usable = (radius - 2 * eps).clamp_min(eps)
        return eps + usable * torch.sigmoid(self.raw_breakpoint)

    def set_piecewise_breakpoint(self, breakpoint):
        """Set the constrained breakpoint through its unconstrained parameter."""

        with torch.no_grad():
            radius = self._piecewise_radius()
            eps = torch.finfo(radius.dtype).eps
            desired = torch.as_tensor(
                breakpoint, dtype=radius.dtype, device=radius.device
            )
            desired = desired.clamp(min=2 * eps, max=(radius - 2 * eps).clamp_min(2 * eps))
            ratio = ((desired - eps) / (radius - 2 * eps).clamp_min(eps))
            ratio = ratio.clamp(min=1e-6, max=1.0 - 1e-6)
            self.raw_breakpoint.copy_(torch.logit(ratio))

    def configure_piecewise(
        self, enabled=True, breakpoint=None, code_allocation=(4, 8, 4)
    ):
        """Enable/disable the selective asymmetric three-region A4 mapping."""

        allocation = tuple(int(value) for value in code_allocation)
        if enabled:
            if self.quantize_config.user != "act":
                raise ValueError("piecewise A4 is activation-only")
            if self.quant_bits != 4:
                raise ValueError(
                    f"piecewise A4 requires a 4-bit quantizer, got {self.quant_bits}"
                )
            if self.x_min.numel() != 1 or self.x_max.numel() != 1:
                raise ValueError("piecewise A4 currently requires per-tensor ranges")
            if len(allocation) != 3 or sum(allocation) != 2 ** self.quant_bits:
                raise ValueError(
                    "piecewise code allocation must have three regions and sum to 16"
                )
            if min(allocation) < 1:
                raise ValueError("each piecewise region must receive at least one code")
            if not bool((self.x_min.detach() < 0).all() and (self.x_max.detach() > 0).all()):
                raise ValueError("piecewise A4 requires a range crossing zero")
            self.piecewise_code_allocation.copy_(
                torch.tensor(
                    allocation,
                    dtype=self.piecewise_code_allocation.dtype,
                    device=self.piecewise_code_allocation.device,
                )
            )
            self._piecewise_runtime_code_allocation = allocation
            if breakpoint is not None:
                self.set_piecewise_breakpoint(breakpoint)
        self.piecewise_enabled.fill_(bool(enabled))
        self._piecewise_runtime_enabled = bool(enabled)
        self.raw_breakpoint.requires_grad_(bool(enabled))

    def _piecewise_qdq(self, x):
        lower = self.x_min.to(dtype=x.dtype)
        upper = self.x_max.to(dtype=x.dtype)
        breakpoint = self.piecewise_breakpoint().to(dtype=x.dtype)
        left_levels, central_levels, right_levels = (
            self._piecewise_runtime_code_allocation
        )
        clipped = x.clamp(min=lower, max=upper)
        left_mask = clipped < -breakpoint
        right_mask = clipped > breakpoint
        region_lower = torch.where(
            left_mask,
            lower,
            torch.where(right_mask, breakpoint, -breakpoint),
        )
        region_upper = torch.where(
            left_mask,
            -breakpoint,
            torch.where(right_mask, upper, breakpoint),
        )
        level_count = torch.where(
            left_mask,
            clipped.new_tensor(left_levels),
            torch.where(
                right_mask,
                clipped.new_tensor(right_levels),
                clipped.new_tensor(central_levels),
            ),
        )
        step = ((region_upper - region_lower) / level_count).clamp_min(
            torch.finfo(x.dtype).eps
        )
        code = Round.apply((clipped - region_lower) / step - 0.5)
        code = torch.maximum(
            code.new_zeros(()), torch.minimum(code, level_count - 1)
        )
        return region_lower + (code + 0.5) * step

    def forward(self, x: torch.Tensor):
        if not self.init_flag:
            self.Init_basicinfo(x)
        if self.running_stat:
            self.act_momentum_update(x)
        delta, zero_point = self.set_param()
        self.delta = delta.detach().clone()
        self.zero_point = zero_point.detach().clone()
        if self._piecewise_runtime_enabled:
            return self._piecewise_qdq(x)
        # Vector FP64 bounds would otherwise promote FP16/FP32 activations and
        # make the following Conv/Linear fail on a dtype mismatch.  The cast is
        # differentiable, so range reconstruction still updates x_min/x_max.
        forward_delta = delta.to(dtype=x.dtype)
        forward_zero_point = zero_point.to(dtype=x.dtype)
        x = self.quantize(x, forward_delta, forward_zero_point)
        x = self.dequantize(x, forward_delta, forward_zero_point)
        return x

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Old checkpoints predate selective PLQ.  Supplying the dormant defaults
        # here preserves strict loading without changing their inference path.
        defaults = {
            "raw_breakpoint": self.raw_breakpoint.detach(),
            "piecewise_enabled": self.piecewise_enabled.detach(),
            "piecewise_code_allocation": self.piecewise_code_allocation.detach(),
            "piecewise_version": self.piecewise_version.detach(),
        }
        for suffix, value in defaults.items():
            key = prefix + suffix
            if key not in state_dict:
                state_dict[key] = value
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.sync_piecewise_runtime_state()
        self.raw_breakpoint.requires_grad_(self._piecewise_runtime_enabled)

    def Init_basicinfo(self, x):
        # print("Init_basicinfo")
        x_clone = x.clone().detach()
        if self.quantize_config.user == "weight":
            channel_wise = True
        else:
            channel_wise = False
        self.init_quantization_scale(x_clone, channel_wise)
        self.set_init_state()

    def reset_reference_dobi_statistics(self):
        self.reference_dobi_search_count = 0
        self.reference_dobi_selection_counts = [0] * len(
            self.reference_dobi_candidates
        )
        self.reference_dobi_first_selection = None
        self.reference_dobi_last_selection = None
        self.reference_dobi_fallback_count = 0
        self.reference_dobi_last_percentile_sample_count = 0
        self.reference_dobi_last_full_scored_elements = 0

    def configure_reference_dobi(
        self,
        enabled=True,
        *,
        update_mode="ema",
        sample_elements=262144,
        tail_probabilities=(0.0, 0.0001, 0.0005, 0.001, 0.002),
        momentum=0.95,
        reset_statistics=True,
    ):
        tails = tuple(float(value) for value in tail_probabilities)
        if tails != (0.0, 0.0001, 0.0005, 0.001, 0.002):
            raise ValueError("native reference DOBI requires C0-C4 tails")
        if int(sample_elements) != 262144:
            raise ValueError(
                "native reference DOBI requires a 262144-element percentile cap"
            )
        if float(momentum) != 0.95:
            raise ValueError("native reference DOBI requires momentum=0.95")
        update_mode = str(update_mode)
        if update_mode not in {"init_once", "ema"}:
            raise ValueError("reference DOBI update_mode must be init_once or ema")
        if self.quantize_config.user != "act":
            raise ValueError("native reference DOBI is activation-only")
        if self.x_min.numel() != 1 or self.x_max.numel() != 1:
            raise RuntimeError(
                "native reference DOBI requires a fresh scalar-range checkpoint"
            )
        self.reference_dobi_enabled = bool(enabled)
        self.reference_dobi_update_mode = update_mode if enabled else "disabled"
        self.reference_dobi_sample_elements = int(sample_elements)
        self.reference_dobi_candidates = tuple((p, 1.0 - p) for p in tails)
        self.reference_dobi_momentum = float(momentum)
        if reset_statistics:
            self.reset_reference_dobi_statistics()

    def _reference_dobi_qdq(self, values, x_min, x_max):
        eps = torch.tensor(1.0e-8, device=values.device, dtype=values.dtype)
        if self.sym:
            delta = torch.maximum(x_max.abs(), x_min.abs()).clamp_min(eps)
            delta = delta / (2 ** (self.quant_bits - 1))
            zero_point = torch.zeros_like(delta)
        else:
            delta = (x_max - x_min).clamp_min(eps)
            delta = delta / (2 ** self.quant_bits - 1)
            if self.sign:
                zero_point = Round.apply((x_max + x_min) / (2 * delta))
            else:
                zero_point = Round.apply(x_min / delta)
        quantized = self.quantize(values, delta, zero_point)
        return self.dequantize(quantized, delta, zero_point)

    @torch.no_grad()
    def _reference_dobi_bounds(self, x):
        values = x.detach().float()
        flat = values.flatten()
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError("native reference DOBI input contains NaN or Inf")
        if flat.numel() > self.reference_dobi_sample_elements:
            # Match reference DOBI: a fresh random subset proposes percentile
            # bounds, while the complete current tensor judges every C0-C4.
            indices = torch.randperm(flat.numel(), device=flat.device)[
                : self.reference_dobi_sample_elements
            ]
            sample = flat[indices]
        else:
            sample = flat
        self.reference_dobi_last_percentile_sample_count = int(sample.numel())
        self.reference_dobi_last_full_scored_elements = int(values.numel())
        best_error = None
        best_min = None
        best_max = None
        best_index = None
        for candidate_index, (low_p, high_p) in enumerate(
            self.reference_dobi_candidates
        ):
            x_min = torch.quantile(sample, low_p)
            x_max = torch.quantile(sample, high_p)
            if not bool(torch.isfinite(x_min) and torch.isfinite(x_max)):
                continue
            if bool((x_max - x_min).abs() < 1.0e-8):
                continue
            restored = self._reference_dobi_qdq(values, x_min, x_max)
            error = torch.mean((restored - values).square())
            if best_error is None or bool(error < best_error):
                best_error = error
                best_min = x_min.detach()
                best_max = x_max.detach()
                best_index = candidate_index
        self.reference_dobi_search_count += 1
        if best_index is None:
            self.reference_dobi_fallback_count += 1
            best_min = values.min().detach()
            best_max = values.max().detach()
        else:
            self.reference_dobi_selection_counts[best_index] += 1
            self.reference_dobi_last_selection = int(best_index)
            if self.reference_dobi_first_selection is None:
                self.reference_dobi_first_selection = int(best_index)
        return (
            best_min.to(device=x.device, dtype=x.dtype),
            best_max.to(device=x.device, dtype=x.dtype),
        )

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        if self.reference_dobi_enabled:
            if channel_wise or self.quantize_config.user != "act":
                raise RuntimeError(
                    "native reference DOBI requires a tensorwise activation"
                )
            x_min, x_max = self._reference_dobi_bounds(x)
            self.x_min.data = x_min
            self.x_max.data = x_max
            self.delta, self.zero_point = self.set_param()
            return
        x_clone = x.clone().detach()
        if channel_wise:
            x_clone = x.clone().detach()
            x_max = self.get_tensor_maxmin(x_clone, type="max")
            x_max = self.channel_wise_reshape(x_max, x.shape)
            x_min = self.get_tensor_maxmin(x_clone, type="min")
            x_min = self.channel_wise_reshape(x_min, x.shape)
        else:
            x_max = x_clone.max().clone()
            x_min = x_clone.min().clone()
        center = (x_max + x_min) * 0.5
        half_span = (x_max - x_min).abs() * 0.5
        minimum_half_span = torch.maximum(
            center.abs() * 1.0e-6,
            torch.full_like(center, 0.5e-6),
        )
        half_span = torch.maximum(half_span, minimum_half_span)
        self.x_min.data = center - half_span
        self.x_max.data = center + half_span
        self.delta, self.zero_point = self.set_param()

    def get_tensor_maxmin(self, x, type="max"):
        x_clone = x.clone().detach()
        if type == 'max':
            if len(x.shape) == 4:
                x_max = x_clone.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.max(dim=-1)[0]
            else:
                x_max = x_clone.max()
            return x_max
        elif type == 'min':
            if len(x.shape) == 4:
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 3:
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 2:
                x_min = x_clone.min(dim=-1)[0]
            else:
                x_min = x_clone.min()
            return x_min
        else:
            return None

    def channel_wise_reshape(self, value, shape):
        if len(shape) == 4:
            value = value.view(-1, 1, 1, 1)
        elif len(shape) == 3:
            value = value.view(-1, 1, 1)
        elif len(shape) == 2:
            value = value.view(-1, 1)
        else:
            pass
        return value

    def set_param(self):
        delta, zero_point = None, None
        if self.sym:
            delta = torch.max(torch.abs(self.x_max), torch.abs(self.x_min)) / (2 ** (self.quant_bits - 1))
            delta = delta.clamp_min(torch.finfo(delta.dtype).eps)
            zero_point = torch.zeros_like(delta)
        else:
            # The training loop projects these parameters immediately after
            # Adam updates.  Ordering them here is a synchronization-free
            # defensive fallback for callers outside that loop; save/load
            # boundaries still validate the stored parameters strictly.
            lower = torch.minimum(self.x_min, self.x_max)
            upper = torch.maximum(self.x_min, self.x_max)
            delta = (upper - lower) / (2 ** (self.quant_bits) - 1)
            delta = delta.clamp_min(torch.finfo(delta.dtype).eps)
            if self.sign == True:
                zero_point = Round.apply((upper + lower) / (2 * delta))
            else:
                zero_point = Round.apply(lower / delta)
        return delta, zero_point

    def quantize(self, x: torch.Tensor, delta: torch.Tensor, zero_point: torch.Tensor):
        x = x / delta
        # x = torch.round(x)
        # x = round_ste(x)
        x = Round.apply(x)
        if not self.sym:
            x = x - zero_point
        if self.sign == True:
            x = torch.clamp(x, -2 ** (self.quant_bits - 1), 2 ** (self.quant_bits - 1) - 1)
        else:
            x = torch.clamp(x, 0, 2 ** self.quant_bits - 1)
        return x
    
    def dequantize(self, x: torch.Tensor, delta: torch.Tensor, zero_point: torch.Tensor):
        if not self.sym:
            x = x + zero_point
        x = x * delta
        return x

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert(self.init_flag)

        if self.reference_dobi_enabled:
            if self.reference_dobi_update_mode != "ema":
                raise RuntimeError(
                    "init-once reference DOBI must keep quantizer running_stat=False"
                )
            x_min, x_max = self._reference_dobi_bounds(x)
            momentum = self.reference_dobi_momentum
            self.x_min.data = self.x_min * momentum + x_min * (1.0 - momentum)
            self.x_max.data = self.x_max * momentum + x_max * (1.0 - momentum)
            self.delta, self.zero_point = self.set_param()
            return

        x_clone = x.clone().detach()
        if self.quantize_config.user == "weight":
            x_clone = x.clone().detach()
            x_max = self.get_tensor_maxmin(x_clone, type="max")
            x_max = self.channel_wise_reshape(x_max, x.shape)
            x_min = self.get_tensor_maxmin(x_clone, type="min")
            x_min = self.channel_wise_reshape(x_min, x.shape)
        else:
            x_max = x_clone.max().clone()
            x_min = x_clone.min().clone()

        self.x_min.data = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max.data = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        self.delta, self.zero_point = self.set_param()


class ManagedTensorwise2DQuantLayer(Basic2DquantLayer):
    """Ordinary affine QDQ with a scalar range at a managed attention site.

    The marker only changes which post-H AsyDoBi initialization sweep owns the
    range.  Forward quantization is inherited verbatim from
    :class:`Basic2DquantLayer`; in particular it does not replace or scalarize
    the reversible Q/K or W/V scale factors applied by the attention module.
    """

    _FAMILIES = {"paired_qk", "attention_wv"}
    _ROLES = {"q", "k", "w", "v"}

    def __init__(
        self,
        quantize_config: Quantize_config,
        *,
        family: str,
        role: str,
    ):
        super().__init__(quantize_config)
        family = str(family)
        role = str(role)
        if family not in self._FAMILIES:
            raise ValueError(f"unsupported managed tensorwise family: {family}")
        if role not in self._ROLES:
            raise ValueError(f"unsupported managed tensorwise role: {role}")
        if family == "paired_qk" and role not in {"q", "k"}:
            raise ValueError("paired_qk only supports q/k roles")
        if family == "attention_wv" and role not in {"w", "v"}:
            raise ValueError("attention_wv only supports w/v roles")

        self.managed_tensorwise_asydobi = True
        self.managed_tensorwise_family = family
        self.managed_tensorwise_role = role
        self.paired_qk_managed = family == "paired_qk"
        self.attention_wv_managed = family == "attention_wv"
        self.channelwise_axis = None
        self.channelwise_role = f"managed_{family}_{role}_per_tensor"


class PerHead2DquantLayer(Basic2DquantLayer):
    """Activation QDQ with one learnable range for every attention head.

    The attention implementation stores Q/K as ``[batch * heads, tokens,
    dim_head]``.  This quantizer temporarily restores the head axis, applies
    ranges shaped ``[heads]`` and returns the original layout.  It deliberately
    remains a ``Basic2DquantLayer`` subclass so PassionSR's existing
    weight-and-activation range reconstruction automatically optimizes its
    ``x_min``/``x_max`` parameters.
    """

    def __init__(self, quantize_config: Quantize_config, heads: int):
        super().__init__(quantize_config)
        self.heads = int(heads)
        if self.heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        parameter_dtype = self.x_min.dtype
        parameter_device = self.x_min.device
        self.x_min = nn.Parameter(
            torch.full(
                (self.heads,), -0.5, dtype=parameter_dtype, device=parameter_device
            )
        )
        self.x_max = nn.Parameter(
            torch.full(
                (self.heads,), 0.5, dtype=parameter_dtype, device=parameter_device
            )
        )
        # Ordinary scalar DoBi must not overwrite ranges calibrated by the
        # dedicated paired-Q/K path.
        self.paired_qk_managed = True

    def _head_view(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(
                "PerHead2DquantLayer expects [batch*heads, tokens, dim_head], "
                f"got shape={tuple(x.shape)}"
            )
        if x.shape[0] % self.heads != 0:
            raise ValueError(
                f"leading dimension {x.shape[0]} is not divisible by heads={self.heads}"
            )
        batch = x.shape[0] // self.heads
        return x.reshape(batch, self.heads, x.shape[1], x.shape[2])

    @staticmethod
    def _range_view(value: torch.Tensor):
        return value.reshape(1, value.numel(), 1, 1)

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        head_view = self._head_view(x)
        if not self.init_flag:
            self.Init_basicinfo(head_view)
        if self.running_stat:
            self.act_momentum_update(head_view)
        delta, zero_point = self.set_param()
        self.delta = delta.detach().clone()
        self.zero_point = zero_point.detach().clone()
        # The legacy range parameters are float64.  Cast the derived QDQ
        # operands to the activation dtype so Q/K do not promote the attention
        # matmul (and the following float32 projection) to float64.
        forward_delta = delta.to(dtype=head_view.dtype)
        forward_zero_point = zero_point.to(dtype=head_view.dtype)
        quantized = self.quantize(
            head_view,
            self._range_view(forward_delta),
            self._range_view(forward_zero_point),
        )
        quantized = self.dequantize(
            quantized,
            self._range_view(forward_delta),
            self._range_view(forward_zero_point),
        )
        return quantized.reshape(original_shape)

    def Init_basicinfo(self, x):
        self.init_quantization_scale(x)
        self.set_init_state()

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        del channel_wise
        head_view = x if x.dim() == 4 else self._head_view(x)
        x_clone = head_view.detach()
        x_min = x_clone.amin(dim=(0, 2, 3))
        x_max = x_clone.amax(dim=(0, 2, 3))
        self.set_head_ranges(x_min, x_max)

    def set_head_ranges(self, x_min: torch.Tensor, x_max: torch.Tensor):
        x_min = torch.as_tensor(
            x_min, device=self.x_min.device, dtype=self.x_min.dtype
        ).reshape(-1)
        x_max = torch.as_tensor(
            x_max, device=self.x_max.device, dtype=self.x_max.dtype
        ).reshape(-1)
        if x_min.numel() != self.heads or x_max.numel() != self.heads:
            raise ValueError(
                "Per-head range count mismatch: "
                f"heads={self.heads}, min={x_min.numel()}, max={x_max.numel()}"
            )
        if not torch.isfinite(x_min).all() or not torch.isfinite(x_max).all():
            raise ValueError("Per-head ranges contain NaN or Inf")
        invalid = x_max <= x_min
        if invalid.any():
            center = (x_max + x_min) / 2
            padding = torch.maximum(
                center.abs() * 1.0e-6,
                torch.full_like(center, torch.finfo(center.dtype).eps),
            )
            x_min = torch.where(invalid, center - padding, x_min)
            x_max = torch.where(invalid, center + padding, x_max)
        with torch.no_grad():
            self.x_min.copy_(x_min)
            self.x_max.copy_(x_max)
        self.set_init_state(True)
        self.delta, self.zero_point = (
            value.detach().clone() for value in self.set_param()
        )

    def act_momentum_update(
        self, x: torch.Tensor, act_range_momentum: float = 0.95
    ):
        if not self.init_flag:
            raise RuntimeError("Per-head quantizer must be initialized before update")
        head_view = x if x.dim() == 4 else self._head_view(x)
        x_clone = head_view.detach()
        x_min = x_clone.amin(dim=(0, 2, 3))
        x_max = x_clone.amax(dim=(0, 2, 3))
        with torch.no_grad():
            self.x_min.mul_(act_range_momentum).add_(
                x_min, alpha=1 - act_range_momentum
            )
            self.x_max.mul_(act_range_momentum).add_(
                x_max, alpha=1 - act_range_momentum
            )
        self.delta, self.zero_point = self.set_param()

class BasicQuantLayer(nn.Module):
    """
    A basic quantization layer for neural networks, supporting both symmetric and asymmetric quantization,
    as well as channel-wise and tensor-wise quantization. This layer can be used for quantizing weights or activations.
    Args:
        quantize_config (Quantize_config): Configuration object specifying quantization parameters such as
            number of bits, signed/unsigned, symmetric/asymmetric, and user type ("weight" or "activation").
        delta (float, optional): Initial quantization scale (step size). Default is 1.
        zero_point (float, optional): Initial zero point for quantization. Default is 0.
        load (bool, optional): If True, skips initialization of quantization parameters. Default is False.
    Attributes:
        quantize_config (Quantize_config): Stores the quantization configuration.
        quant_bits (int): Number of bits for quantization.
        sign (bool): Whether quantization is signed.
        sym (bool): Whether quantization is symmetric.
        delta (nn.Parameter): Learnable quantization scale.
        zero_point (nn.Parameter): Learnable zero point for asymmetric quantization.
        init_flag (bool): Indicates if the layer has been initialized.
        running_stat (bool): If True, updates quantization range using running statistics (momentum).
        x_min (nn.Parameter): Minimum value observed (used for scale calculation).
        x_max (nn.Parameter): Maximum value observed (used for scale calculation).
    Methods:
        set_init_state(flag=True): Sets the initialization flag.
        set_running_stat(flag): Enables or disables running statistics for activation range.
        forward(x): Applies quantization and dequantization to the input tensor.
        Init_basicinfo(x): Initializes quantization parameters based on input tensor statistics.
        init_quantization_scale(x, channel_wise=False): Initializes scale and zero point, optionally channel-wise.
        get_tensor_maxmin(x, type="max"): Computes channel-wise or tensor-wise min/max.
        channel_wise_reshape(value, shape): Reshapes min/max values for channel-wise quantization.
        set_param(): Updates delta and zero_point based on current min/max and quantization config.
        quantize(x): Quantizes the input tensor.
        dequantize(x): Dequantizes the quantized tensor.
        act_momentum_update(x, act_range_momentum=0.95): Updates min/max using exponential moving average.
    """
    def __init__(self, quantize_config: Quantize_config, delta=1, zero_point=0, load=False):
        super(BasicQuantLayer, self).__init__()
        self.quantize_config = quantize_config
        self.quant_bits = quantize_config.quant_bits
        self.sign = quantize_config.sign
        self.sym = quantize_config.sym
        self.delta, self.zero_point = nn.Parameter(torch.tensor(delta, dtype=float)), nn.Parameter(torch.tensor(zero_point, dtype=float))
        self.init_flag = load
        # self.x_min, self.x_max = None, None
        self.running_stat = False
        self.x_min, self.x_max =  nn.Parameter(torch.tensor(-0.5, dtype=float)), nn.Parameter(torch.tensor(0.5, dtype=float))

    def set_init_state(self, flag=True):
        self.init_flag = flag

    def set_running_stat(self, flag):
        self.running_stat = flag

    def forward(self, x: torch.Tensor):
        if not self.init_flag:
            self.Init_basicinfo(x)
        if self.running_stat:
            self.act_momentum_update(x)
        x = self.quantize(x)
        x = self.dequantize(x)
        return x

    def Init_basicinfo(self, x):
        x_clone = x.clone().detach()
        if self.quantize_config.user == "weight":
            channel_wise = True
        else:
            channel_wise = False
        self.init_quantization_scale(x, channel_wise)
        self.set_init_state()

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        x_clone = x.clone().detach()
        if channel_wise:
            x_clone = x.clone().detach()
            x_max = self.get_tensor_maxmin(x_clone, type="max")
            x_max = self.channel_wise_reshape(x_max, x.shape)
            x_min = self.get_tensor_maxmin(x_clone, type="min")
            x_min = self.channel_wise_reshape(x_min, x.shape)
        else:
            x_max = x_clone.max().clone()
            x_min = x_clone.min().clone()
        self.x_max.data, self.x_min.data = x_max, x_min
        self.set_param()

    def get_tensor_maxmin(self, x, type="max"):
        x_clone = x.clone().detach()
        if type == 'max':
            if len(x.shape) == 4:
                x_max = x_clone.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.max(dim=-1)[0]
            else:
                x_max = x_clone.max()
            return x_max
        elif type == 'min':
            if len(x.shape) == 4:
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 3:
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 2:
                x_min = x_clone.min(dim=-1)[0]
            else:
                x_min = x_clone.min()
            return x_min
        else:
            return None

    def channel_wise_reshape(self, value, shape):
        if len(shape) == 4:
            value = value.view(-1, 1, 1, 1)
        elif len(shape) == 3:
            value = value.view(-1, 1, 1)
        elif len(shape) == 2:
            value = value.view(-1, 1)
        else:
            pass
        return value

    def set_param(self):
        if self.sym:
            self.delta.data = torch.max(torch.abs(self.x_max), torch.abs(self.x_min)) / (2 ** (self.quant_bits - 1))
            self.zero_point.data = nn.Parameter(torch.zeros_like(self.delta))
        else:
            self.delta.data = (self.x_max - self.x_min)
            self.delta.data = self.delta / (2 ** (self.quant_bits) - 1)
            if self.sign == True:
                # self.zero_point.data = round_ste((self.x_max + self.x_min) / ( 2 * self.delta))
                self.zero_point.data = Round.apply((self.x_max + self.x_min) / ( 2 * self.delta))
            else:
                # self.zero_point.data = round_ste(self.x_min / self.delta)
                self.zero_point.data = Round.apply(self.x_min / self.delta)

    def quantize(self, x: torch.Tensor):
        x = x / self.delta
        # x = torch.round(x)
        # x = round_ste(x)
        x = Round.apply(x)
        if not self.sym:
            x = x - self.zero_point
        if self.sign == True:
            x = torch.clamp(x, -2 ** (self.quant_bits - 1), 2 ** (self.quant_bits - 1) - 1)
        else:
            x = torch.clamp(x, 0, 2 ** self.quant_bits - 1)
        return x
    
    def dequantize(self, x: torch.Tensor):
        if not self.sym:
            x = x + self.zero_point
        x = x * self.delta
        return x

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert(self.init_flag)

        x_clone = x.clone().detach()
        if self.quantize_config.user == "weight":
            x_clone = x.clone().detach()
            x_max = self.get_tensor_maxmin(x_clone, type="max")
            x_max = self.channel_wise_reshape(x_max, x.shape)
            x_min = self.get_tensor_maxmin(x_clone, type="min")
            x_min = self.channel_wise_reshape(x_min, x.shape)
        else:
            x_max = x_clone.max().clone()
            x_min = x_clone.min().clone()

        self.x_min.data = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max.data = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        self.set_param()

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

class QuantLayer(nn.Module):
    """
    A quantization wrapper layer for PyTorch modules (Conv2d, Conv1d, Linear).
    This layer replaces the original module's forward pass with a quantized version,
    supporting both weight and activation quantization, as well as optional channel splitting
    for separate quantization of input/weight channels.
    Args:
        org_module (Union[nn.Conv2d, nn.Linear, nn.Conv1d]): The original PyTorch module to be quantized.
        quantize_config (QuantizeModel_config): Configuration object specifying quantization parameters.
        device (str, optional): Device to use for computation. Default is "cuda".
    Attributes:
        quantize_config (QuantizeModel_config): Stores quantization configuration.
        device (str): Device for computation.
        quant_layer_type (type): Quantizer class determined by configuration.
        fwd_kwargs (dict): Forward function keyword arguments (stride, padding, etc.).
        fwd_func (callable): Forward function (F.conv2d, F.conv1d, or F.linear).
        weight (torch.Tensor): Quantized weight tensor.
        org_weight (torch.Tensor): Original weight tensor (for dequantization or reference).
        bias (torch.Tensor or None): Quantized bias tensor.
        org_bias (torch.Tensor or None): Original bias tensor.
        bias_quantizer (callable or None): Quantizer for bias.
        use_weight_quant (bool): Whether to apply weight quantization.
        use_act_quant (bool): Whether to apply activation quantization.
        weight_quantizer_0 (callable): Quantizer for weights (or first split).
        act_quantizer_0 (callable): Quantizer for activations (or first split).
        split (bool): Whether to split channels for separate quantization.
        weight_quantizer_1 (callable): Quantizer for second split of weights (if split enabled).
        act_quantizer_1 (callable): Quantizer for second split of activations (if split enabled).
        activation_function (callable): Activation function (StraightThrough).
    Methods:
        set_split(split): Enable or disable channel splitting for quantization.
        forward(input): Forward pass with quantization applied to weights and activations.
        set_quant_state(weight_quant, act_quant): Enable/disable weight and activation quantization.
        set_running_stat(running_stat): Set running statistics mode for activation quantizers.
    """
    def __init__(
        self,
        org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d],
        quantize_config : QuantizeModel_config,
        device = "cuda",
        # split : bool = False,
    ) -> None:
        super(QuantLayer, self).__init__()
        self.quantize_config = quantize_config
        self.device = device
        self.quant_layer_type = get_layer_type(quantize_config)

        # self.org_module = org_module
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight.data
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias.data
            self.org_bias = org_module.bias.data.clone()
            bias_quant_config = getattr(
                self.quantize_config, "bias_quant", {"enabled": True}
            )
            if isinstance(bias_quant_config, dict):
                quantize_bias = bool(bias_quant_config.get("enabled", True))
            else:
                quantize_bias = bool(bias_quant_config)
            self.bias_quantizer = (
                self.quant_layer_type(self.quantize_config.weight_config)
                if quantize_bias
                else None
            )
        else:
            self.bias = None
            self.org_bias = None
            self.bias_quantizer = None

        self.use_weight_quant = True
        self.use_act_quant = True
        self.weight_quantizer_0 = self.quant_layer_type(self.quantize_config.weight_config)
        self.act_quantizer_0 = self.quant_layer_type(self.quantize_config.act_config)
        self.set_split(quantize_config.split)

        post_h_reference_version = dict(
            getattr(self.quantize_config, "hadamard_dobi", {}).get(
                "post_h_reference_initializer", {}
            )
        ).get("version")
        tensorwise_reference = (
            post_h_reference_version == "posth_reference_tensorwise_native_v2"
        )
        # Channel-wise metadata belongs only to the older per-channel
        # initializers.  Native reference DOBI V2 is scalar per tensor from
        # construction, avoiding stale metadata in PTQ and inference models.
        channel_axis = None if tensorwise_reference else (
            -1 if self.fwd_func is F.linear else 1
        )
        channel_role = (
            "operator_activation_per_tensor"
            if tensorwise_reference
            else "operator_activation"
        )
        for act_quantizer in (
            self.act_quantizer_0,
            getattr(self, "act_quantizer_1", None),
        ):
            if act_quantizer is not None:
                act_quantizer.channelwise_axis = channel_axis
                act_quantizer.channelwise_role = channel_role

        self.activation_function = StraightThrough()

    def set_split(self, split):
        self.split = split
        if self.split:
            self.weight_quantizer_1 = self.quant_layer_type(self.quantize_config.weight_config)
            self.act_quantizer_1 = self.quant_layer_type(self.quantize_config.act_config)

    def forward(self, input: torch.Tensor):
        if self.use_act_quant:
            if self.split:
                split_divide_line = input.shape[1] // 2
                input_0 = self.act_quantizer_0(input[:, :split_divide_line, ...])
                input_1 = self.act_quantizer_1(input[:, split_divide_line:, ...])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                input = self.act_quantizer_0(input)
        if self.use_weight_quant:
            if self.split:
                split_divide_line = self.weight.shape[1] // 2
                weight_0 = self.weight_quantizer_0(self.weight[:, :split_divide_line, ...])
                weight_1 = self.weight_quantizer_1(self.weight[:, split_divide_line:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                weight = self.weight_quantizer_0(self.weight)
            # bias = self.bias
            if self.org_bias is not None:
                bias = (
                    self.bias_quantizer(self.bias)
                    if self.bias_quantizer is not None
                    else self.bias
                )
            else:
                bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
    
    def set_running_stat(self, running_stat: bool):
        self.act_quantizer_0.set_running_stat(running_stat)
        if self.split != 0:
            self.act_quantizer_1.set_running_stat(running_stat)


@torch.no_grad()
def project_quantizer_ranges_(
    model: nn.Module,
    minimum_gap: float = 1.0e-6,
    trainable_only: bool = True,
):
    """Project relevant ranges with batched foreach kernels per device/dtype."""

    minimum_gap = float(minimum_gap)
    if minimum_gap <= 0.0:
        raise ValueError("minimum_gap must be positive")
    quantizer_count = 0
    range_value_count = 0
    repaired_value_count = 0
    groups = {}
    for module in model.modules():
        if not isinstance(module, Basic2DquantLayer):
            continue
        if trainable_only and not (
            module.x_min.requires_grad or module.x_max.requires_grad
        ):
            continue
        quantizer_count += 1
        range_value_count += int(module.x_min.numel())
        key = (module.x_min.device, module.x_min.dtype)
        group = groups.setdefault(key, {"lower": [], "upper": []})
        group["lower"].append(module.x_min)
        group["upper"].append(module.x_max)

    for group in groups.values():
        learned_lower = group["lower"]
        learned_upper = group["upper"]
        gaps = torch._foreach_sub(learned_upper, learned_lower)
        flat_gaps = torch.cat([gap.reshape(-1) for gap in gaps])
        if not bool(torch.isfinite(flat_gaps).all()):
            raise FloatingPointError("quantizer range contains NaN or Inf")
        repaired_value_count += int((flat_gaps < minimum_gap).sum().item())

        lower = torch._foreach_minimum(learned_lower, learned_upper)
        upper = torch._foreach_maximum(learned_lower, learned_upper)
        center = torch._foreach_mul(torch._foreach_add(lower, upper), 0.5)
        half_span = torch._foreach_mul(torch._foreach_sub(upper, lower), 0.5)
        minimum_half_span = torch._foreach_mul(
            torch._foreach_abs(center), 1.0e-6
        )
        minimum_half_span = torch._foreach_maximum(
            minimum_half_span, minimum_gap * 0.5
        )
        half_span = torch._foreach_maximum(half_span, minimum_half_span)
        projected_lower = torch._foreach_sub(center, half_span)
        projected_upper = torch._foreach_add(center, half_span)
        torch._foreach_copy_(learned_lower, projected_lower)
        torch._foreach_copy_(learned_upper, projected_upper)

        projected_gaps = torch._foreach_sub(learned_upper, learned_lower)
        flat_projected_gaps = torch.cat(
            [gap.reshape(-1) for gap in projected_gaps]
        )
        if not bool((flat_projected_gaps > 0).all()):
            raise RuntimeError("failed to project quantizer ranges")
    return {
        "quantizers": quantizer_count,
        "range_values": range_value_count,
        "repaired_values": repaired_value_count,
    }


@torch.no_grad()
def validate_quantizer_ranges(model: nn.Module, component: str = "model"):
    """Fail before checkpoint save when a learned 2D range is not usable."""

    quantizer_count = 0
    range_value_count = 0
    failures = []
    for name, module in model.named_modules():
        if not isinstance(module, Basic2DquantLayer):
            continue
        quantizer_count += 1
        range_value_count += int(module.x_min.numel())
        if not bool(torch.isfinite(module.x_min).all()):
            failures.append(f"{name}:non_finite_x_min")
            continue
        if not bool(torch.isfinite(module.x_max).all()):
            failures.append(f"{name}:non_finite_x_max")
            continue
        invalid_count = int((module.x_max <= module.x_min).sum().item())
        if invalid_count:
            failures.append(f"{name}:invalid_ranges={invalid_count}")

    if failures:
        preview = ", ".join(failures[:8])
        if len(failures) > 8:
            preview += f", ... ({len(failures) - 8} more)"
        raise RuntimeError(
            f"Invalid {component} quantizer ranges before checkpoint save: "
            f"{preview}"
        )
    return {
        "component": str(component),
        "quantizers": quantizer_count,
        "range_values": range_value_count,
        "invalid_values": 0,
    }

def get_layer_type(quant_config: QuantizeModel_config):
    if quant_config.layer_type == "normal_quant":
        return BasicQuantLayer
    elif quant_config.layer_type == "2Dquant":
        return Basic2DquantLayer
    else:
        raise NotImplementedError(f"Quantization layer {quant_config.layer_type} not implemented")
