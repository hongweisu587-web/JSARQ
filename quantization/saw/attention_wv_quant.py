"""Context-coupled low-bit quantization for UNet attention W/V operands.

This module is intentionally dormant unless ``attention_wv_quantization`` is
enabled in the quantization configuration.  It provides:

* a zero-preserving, per-head logarithmic A4 quantizer for self-attention W;
* a per-head/per-group asymmetric A4 quantizer for cross-attention V, with an
  orthonormal H64 transform and an explicit inverse after W@V;
* auxiliary context, direction, and probability-mass losses consumed by the
  two established SAW calibration stages; and
* a post-scale range re-synchronization pass.

The floating reference used by the auxiliary loss keeps the already quantized
Q/K path.  Consequently the loss isolates the two W/V operand QDQ sites rather
than turning the calibration objective into a second full-precision teacher.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .hadamard_utils import deterministic_even_indices, normalized_block_fwht
from .quant_layer import Basic2DquantLayer, Round


ATTENTION_WV_VERSION = 1


def _safe_range(
    lower: torch.Tensor, upper: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    lower = lower.clone()
    upper = upper.clone()
    invalid = upper <= lower
    if invalid.any():
        center = (upper + lower) / 2
        padding = torch.maximum(
            center.abs() * 1.0e-6,
            torch.full_like(center, torch.finfo(center.dtype).eps),
        )
        lower = torch.where(invalid, center - padding, lower)
        upper = torch.where(invalid, center + padding, upper)
    return lower, upper


class _PerUnitObserverMixin:
    """Small deterministic CPU reservoir for per-head or per-group ranges."""

    observer_sample_limit: int
    observer_tail_probability: float

    def _observer_setup(self, unit_shape: Tuple[int, ...], config: dict):
        self._observer_unit_shape = tuple(int(value) for value in unit_shape)
        self.observer_sample_limit = int(
            config.get("observer_samples_per_unit", 8192)
        )
        if self.observer_sample_limit <= 0:
            raise ValueError("observer_samples_per_unit must be positive")
        self.observer_tail_probability = float(
            config.get("tail_probability", 0.001)
        )
        if not 0.0 <= self.observer_tail_probability < 0.5:
            raise ValueError("tail_probability must be in [0, 0.5)")
        self._observer_samples: List[torch.Tensor] = []
        self._observer_sample_count = 0

    def begin_range_observer(self):
        self._observer_samples = []
        self._observer_sample_count = 0
        self.running_stat = True

    def _observe_unit_values(self, values: torch.Tensor):
        if not self.running_stat:
            return
        if tuple(values.shape[: len(self._observer_unit_shape)]) != (
            self._observer_unit_shape
        ):
            raise ValueError(
                "Observer unit shape mismatch: "
                f"expected={self._observer_unit_shape}, got={tuple(values.shape)}"
            )
        remaining = self.observer_sample_limit - self._observer_sample_count
        if remaining <= 0:
            return
        flattened = values.detach().float().reshape(
            *self._observer_unit_shape, -1
        )
        take = min(int(flattened.shape[-1]), int(remaining))
        if take < flattened.shape[-1]:
            positions = deterministic_even_indices(
                flattened.shape[-1], take, device=flattened.device
            )
            flattened = flattened.index_select(-1, positions)
        self._observer_samples.append(flattened.to(device="cpu"))
        self._observer_sample_count += take

    def _observer_tensor(self) -> Optional[torch.Tensor]:
        if not self._observer_samples:
            return None
        return torch.cat(self._observer_samples, dim=-1)

    def end_range_observer(self):
        self.running_stat = False
        samples = self._observer_tensor()
        if samples is not None:
            self._finalize_observer(samples)
        self._observer_samples = []
        self._observer_sample_count = 0

    def set_running_stat(self, flag):
        flag = bool(flag)
        if flag and not self.running_stat:
            self.begin_range_observer()
        elif not flag and self.running_stat:
            self.end_range_observer()


class SoftmaxLog4Quantizer(_PerUnitObserverMixin, Basic2DquantLayer):
    """Per-head Softmax-W A4 with code 0 fixed to exact zero.

    Positive codes 1..15 are logarithmically spaced between ``x_min`` (tau)
    and ``x_max`` (upper bound).  Values below tau map to zero.  ``x_min`` and
    ``x_max`` remain ordinary Parameters so SAW's boundary stage optimizes them
    directly.
    """

    def __init__(self, quantize_config, heads: int, config: dict):
        super().__init__(quantize_config)
        if int(self.quant_bits) != 4:
            raise ValueError(
                f"SoftmaxLog4Quantizer requires A4, got A{self.quant_bits}"
            )
        if self.quantize_config.user != "act":
            raise ValueError("SoftmaxLog4Quantizer is activation-only")
        self.heads = int(heads)
        if self.heads <= 0:
            raise ValueError("heads must be positive")
        self.dynamic_span = float(config.get("dynamic_span", 1024.0))
        if self.dynamic_span <= 1.0:
            raise ValueError("Softmax-W dynamic_span must be greater than one")

        dtype = self.x_min.dtype
        device = self.x_min.device
        self.x_min = nn.Parameter(
            torch.full((self.heads,), 1.0 / self.dynamic_span, dtype=dtype, device=device)
        )
        self.x_max = nn.Parameter(
            torch.ones(self.heads, dtype=dtype, device=device)
        )
        self.register_buffer(
            "attention_wv_version",
            torch.tensor(ATTENTION_WV_VERSION, dtype=torch.int32),
        )
        self.register_buffer(
            "zero_is_exact", torch.tensor(True, dtype=torch.bool)
        )
        self.register_buffer(
            "range_anchor_min", self.x_min.detach().clone(), persistent=False
        )
        self.register_buffer(
            "range_anchor_max", self.x_max.detach().clone(), persistent=False
        )
        self.attention_wv_managed = True
        self.attention_wv_kind = "attn1_softmax_w_log4"
        self._observer_setup((self.heads,), config)

    def _head_view(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[0] % self.heads:
            raise ValueError(
                "SoftmaxLog4Quantizer expects [batch*heads, query, key], "
                f"got shape={tuple(x.shape)}, heads={self.heads}"
            )
        return x.reshape(
            x.shape[0] // self.heads, self.heads, x.shape[1], x.shape[2]
        )

    def _initialize_from_tensor(self, head_view: torch.Tensor):
        upper = head_view.detach().amax(dim=(0, 2, 3)).float()
        upper = upper.clamp_min(torch.finfo(upper.dtype).eps)
        tau = upper / self.dynamic_span
        with torch.no_grad():
            self.x_min.copy_(tau.to(device=self.x_min.device, dtype=self.x_min.dtype))
            self.x_max.copy_(
                upper.to(device=self.x_max.device, dtype=self.x_max.dtype)
            )
        self.set_init_state(True)

    def Init_basicinfo(self, x):
        self._initialize_from_tensor(self._head_view(x))

    def act_momentum_update(self, x, act_range_momentum=0.95):
        del act_range_momentum
        head_view = x if x.dim() == 4 else self._head_view(x)
        self._observe_unit_values(head_view.permute(1, 0, 2, 3))

    def _finalize_observer(self, samples: torch.Tensor):
        upper_q = 1.0 - self.observer_tail_probability
        upper = torch.quantile(samples, upper_q, dim=-1)
        upper = upper.clamp_min(torch.finfo(upper.dtype).eps)
        tau = upper / self.dynamic_span
        with torch.no_grad():
            self.x_min.copy_(tau.to(device=self.x_min.device, dtype=self.x_min.dtype))
            self.x_max.copy_(
                upper.to(device=self.x_max.device, dtype=self.x_max.dtype)
            )
        self.set_init_state(True)

    def capture_range_anchor(self):
        self.range_anchor_min.copy_(self.x_min.detach())
        self.range_anchor_max.copy_(self.x_max.detach())

    def range_anchor_loss(self) -> torch.Tensor:
        return (
            (self.x_min - self.range_anchor_min).float().square().mean()
            + (self.x_max - self.range_anchor_max).float().square().mean()
        )

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        head_view = self._head_view(x)
        if not self.init_flag:
            self._initialize_from_tensor(head_view)
        if self.running_stat:
            self.act_momentum_update(head_view)

        tau = self.x_min.to(dtype=head_view.dtype).clamp_min(
            torch.finfo(head_view.dtype).eps
        )
        upper = torch.maximum(
            self.x_max.to(dtype=head_view.dtype),
            tau * (1.0 + 1.0e-6),
        )
        tau = tau.reshape(1, self.heads, 1, 1)
        upper = upper.reshape(1, self.heads, 1, 1)
        positive = head_view >= tau
        clipped = head_view.clamp_min(torch.finfo(head_view.dtype).tiny)
        clipped = torch.minimum(clipped, upper)

        # Fifteen positive levels include both endpoints, hence 14 intervals.
        log_step = (torch.log(upper) - torch.log(tau)) / 14.0
        log_step = log_step.clamp_min(torch.finfo(head_view.dtype).eps)
        code = Round.apply((torch.log(clipped) - torch.log(tau)) / log_step)
        code = code.clamp(0, 14)
        quantized = tau * torch.exp(code * log_step)
        quantized = torch.where(positive, quantized, torch.zeros_like(quantized))
        return quantized.reshape(original_shape)


class PerHeadGroupVQuantizer(_PerUnitObserverMixin, Basic2DquantLayer):
    """Cross-attention V A4 after H64, with per-head/per-group ranges."""

    def __init__(
        self,
        quantize_config,
        heads: int,
        dim_head: int,
        config: dict,
    ):
        super().__init__(quantize_config)
        if int(self.quant_bits) != 4:
            raise ValueError(
                f"PerHeadGroupVQuantizer requires A4, got A{self.quant_bits}"
            )
        if self.quantize_config.user != "act":
            raise ValueError("PerHeadGroupVQuantizer is activation-only")
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.group_size = int(config.get("group_size", 16))
        self.hadamard_block_size = int(config.get("hadamard_block_size", 64))
        self.hadamard_enabled = bool(config.get("hadamard_enabled", True))
        if self.heads <= 0 or self.dim_head <= 0:
            raise ValueError("heads and dim_head must be positive")
        if self.dim_head % self.group_size:
            raise ValueError(
                f"group_size={self.group_size} does not divide dim_head={self.dim_head}"
            )
        if self.hadamard_enabled:
            if self.hadamard_block_size != self.dim_head:
                raise ValueError(
                    "The first HCC-WV version requires a full dim_head transform: "
                    f"H={self.hadamard_block_size}, dim_head={self.dim_head}"
                )
            if self.hadamard_block_size & (self.hadamard_block_size - 1):
                raise ValueError("hadamard_block_size must be a power of two")
        self.groups = self.dim_head // self.group_size

        dtype = self.x_min.dtype
        device = self.x_min.device
        self.x_min = nn.Parameter(
            torch.full((self.heads, self.groups), -0.5, dtype=dtype, device=device)
        )
        self.x_max = nn.Parameter(
            torch.full((self.heads, self.groups), 0.5, dtype=dtype, device=device)
        )
        self.register_buffer(
            "attention_wv_version",
            torch.tensor(ATTENTION_WV_VERSION, dtype=torch.int32),
        )
        self.register_buffer(
            "selected_group_size",
            torch.tensor(self.group_size, dtype=torch.int32),
        )
        self.register_buffer(
            "selected_hadamard_block_size",
            torch.tensor(
                self.hadamard_block_size if self.hadamard_enabled else 0,
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "range_anchor_min", self.x_min.detach().clone(), persistent=False
        )
        self.register_buffer(
            "range_anchor_max", self.x_max.detach().clone(), persistent=False
        )
        self.attention_wv_managed = True
        self.attention_wv_kind = "attn2_v_h64_group_a4"
        self._observer_setup((self.heads, self.groups), config)

    def _head_view(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[0] % self.heads:
            raise ValueError(
                "PerHeadGroupVQuantizer expects [batch*heads, key, dim_head], "
                f"got shape={tuple(x.shape)}, heads={self.heads}"
            )
        if x.shape[-1] != self.dim_head:
            raise ValueError(
                f"Expected dim_head={self.dim_head}, got {x.shape[-1]}"
            )
        return x.reshape(
            x.shape[0] // self.heads, self.heads, x.shape[1], self.dim_head
        )

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hadamard_enabled:
            return x
        return normalized_block_fwht(x, self.hadamard_block_size, dim=-1)

    def inverse_context(self, x: torch.Tensor) -> torch.Tensor:
        # The normalized Walsh-Hadamard transform is self-inverse.
        return self._transform(x)

    def _group_view(self, transformed: torch.Tensor) -> torch.Tensor:
        return transformed.reshape(
            transformed.shape[0],
            self.heads,
            transformed.shape[2],
            self.groups,
            self.group_size,
        )

    def _initialize_from_tensor(self, head_view: torch.Tensor):
        grouped = self._group_view(self._transform(head_view.detach())).float()
        lower = grouped.amin(dim=(0, 2, 4))
        upper = grouped.amax(dim=(0, 2, 4))
        lower, upper = _safe_range(lower, upper)
        with torch.no_grad():
            self.x_min.copy_(
                lower.to(device=self.x_min.device, dtype=self.x_min.dtype)
            )
            self.x_max.copy_(
                upper.to(device=self.x_max.device, dtype=self.x_max.dtype)
            )
        self.set_init_state(True)

    def Init_basicinfo(self, x):
        self._initialize_from_tensor(self._head_view(x))

    def act_momentum_update(self, x, act_range_momentum=0.95):
        del act_range_momentum
        head_view = x if x.dim() == 4 else self._head_view(x)
        grouped = self._group_view(self._transform(head_view.detach()))
        self._observe_unit_values(grouped.permute(1, 3, 0, 2, 4))

    def _finalize_observer(self, samples: torch.Tensor):
        tail = self.observer_tail_probability
        lower = torch.quantile(samples, tail, dim=-1)
        upper = torch.quantile(samples, 1.0 - tail, dim=-1)
        lower, upper = _safe_range(lower, upper)
        with torch.no_grad():
            self.x_min.copy_(
                lower.to(device=self.x_min.device, dtype=self.x_min.dtype)
            )
            self.x_max.copy_(
                upper.to(device=self.x_max.device, dtype=self.x_max.dtype)
            )
        self.set_init_state(True)

    def capture_range_anchor(self):
        self.range_anchor_min.copy_(self.x_min.detach())
        self.range_anchor_max.copy_(self.x_max.detach())

    def range_anchor_loss(self) -> torch.Tensor:
        return (
            (self.x_min - self.range_anchor_min).float().square().mean()
            + (self.x_max - self.range_anchor_max).float().square().mean()
        )

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        head_view = self._head_view(x)
        if not self.init_flag:
            self._initialize_from_tensor(head_view)
        if self.running_stat:
            self.act_momentum_update(head_view)

        grouped = self._group_view(self._transform(head_view))
        lower = self.x_min.to(dtype=grouped.dtype).reshape(
            1, self.heads, 1, self.groups, 1
        )
        upper = self.x_max.to(dtype=grouped.dtype).reshape(
            1, self.heads, 1, self.groups, 1
        )
        lower, upper = _safe_range(lower, upper)
        delta = ((upper - lower) / 15.0).clamp_min(
            torch.finfo(grouped.dtype).eps
        )
        zero_point = Round.apply(lower / delta)
        code = Round.apply(grouped / delta) - zero_point
        code = code.clamp(0, 15)
        quantized = (code + zero_point) * delta
        return quantized.reshape(original_shape)

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
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        loaded_group_size = int(self.selected_group_size.detach().cpu())
        loaded_hadamard = int(
            self.selected_hadamard_block_size.detach().cpu()
        )
        expected_hadamard = (
            self.hadamard_block_size if self.hadamard_enabled else 0
        )
        if loaded_group_size != self.group_size:
            error_msgs.append(
                f"{prefix}group_size mismatch: checkpoint={loaded_group_size}, "
                f"config={self.group_size}"
            )
        if loaded_hadamard != expected_hadamard:
            error_msgs.append(
                f"{prefix}Hadamard mismatch: checkpoint={loaded_hadamard}, "
                f"config={expected_hadamard}"
            )


def validate_attention_wv_config(config: dict, unet_config: Optional[dict] = None):
    if not config.get("enabled", False):
        return
    if str(config.get("version", "hcc_wv_v1")).lower() != "hcc_wv_v1":
        raise ValueError("Only attention_wv_quantization.version=hcc_wv_v1 is supported")
    if str(config.get("scope", "unet_only")).lower() != "unet_only":
        raise ValueError("HCC-WV v1 is intentionally restricted to scope=unet_only")
    if unet_config is not None:
        if int(unet_config.get("act_quant_bits", 0)) != 4:
            raise ValueError("HCC-WV v1 requires UNet activation quantization at 4 bits")
        if unet_config.get("layer_type") != "2Dquant":
            raise ValueError("HCC-WV v1 requires UNet layer_type=2Dquant")
    softmax_config = config.get("softmax_w", {})
    value_config = config.get("value_v", {})
    if not softmax_config.get("enabled", True):
        raise ValueError("HCC-WV v1 requires softmax_w.enabled=true")
    if softmax_config.get("target", "attn1") != "attn1":
        raise ValueError("HCC-WV v1 supports Softmax-W only on attn1")
    if softmax_config.get("quantizer", "zero_preserving_log") != (
        "zero_preserving_log"
    ):
        raise ValueError("HCC-WV v1 requires softmax_w.quantizer=zero_preserving_log")
    if int(softmax_config.get("bits", 4)) != 4:
        raise ValueError("HCC-WV v1 requires softmax_w.bits=4")
    if int(softmax_config.get("positive_levels", 15)) != 15:
        raise ValueError("HCC-WV v1 reserves one A4 code for zero and 15 for positives")
    if not value_config.get("enabled", True):
        raise ValueError("HCC-WV v1 requires value_v.enabled=true")
    if value_config.get("target", "attn2") != "attn2":
        raise ValueError("HCC-WV v1 supports grouped V only on attn2")
    if int(value_config.get("bits", 4)) != 4:
        raise ValueError("HCC-WV v1 requires value_v.bits=4")
    if not value_config.get("hadamard_enabled", True):
        raise ValueError("HCC-WV v1 requires value_v.hadamard_enabled=true")
    joint_loss_config = config.get("joint_loss", {})
    for stage_name in ("scale_stage", "boundary_stage"):
        stage_config = joint_loss_config.get(stage_name, {})
        for key, value in stage_config.items():
            if key.endswith("_weight") and float(value) < 0:
                raise ValueError(
                    f"joint_loss.{stage_name}.{key} must be non-negative"
                )
    if not joint_loss_config.get("streaming_backward", True):
        raise ValueError(
            "HCC-WV v1 requires joint_loss.streaming_backward=true "
            "to bound calibration memory"
        )
    memory_safety = config.get("memory_safety", {})
    required_memory_safety = (
        "force_transformer_checkpointing",
        "release_stage_state",
        "offload_unet_during_vae_encoder",
        "offload_vae_during_unet",
        "offload_unet_during_vae_decoder",
        "offload_all_before_checkpoint_save",
        "disable_unet_vae_monitoring",
    )
    for safety_flag in required_memory_safety:
        if not memory_safety.get(safety_flag, True):
            raise ValueError(
                "HCC-WV v1 requires "
                f"memory_safety.{safety_flag}=true"
            )
    sampling = joint_loss_config.get("sampling", {})
    max_attention_elements = int(
        sampling.get("max_attention_elements", 4_194_304)
    )
    if max_attention_elements <= 0:
        raise ValueError(
            "joint_loss.sampling.max_attention_elements must be positive"
        )
    for role, default_limit in (("attn1", 64), ("attn2", 128)):
        limit = int(sampling.get(f"{role}_query_tokens", default_limit))
        if limit <= 0:
            raise ValueError(
                f"joint_loss.sampling.{role}_query_tokens must be positive"
            )


def configure_attention_wv_runtime(attention, role: str, config: dict):
    attention._attention_wv_enabled = bool(config.get("enabled", False))
    attention._attention_wv_role = str(role)
    attention._attention_wv_aux_terms = None
    attention._attention_wv_cache_requested = False
    attention._attention_wv_cached_inputs = None
    attention._attention_wv_aux_query_limit = 64 if role == "attn1" else 128
    attention._attention_wv_aux_max_elements = 4_194_304
    attention._attention_wv_last_aux_query_tokens = 0
    attention._attention_wv_last_aux_key_tokens = 0
    attention._attention_wv_last_aux_attention_elements = 0


def store_attention_wv_auxiliary_terms(
    attention,
    reference_context: torch.Tensor,
    quantized_context: torch.Tensor,
    softmax_attention: torch.Tensor,
    quantized_w: torch.Tensor,
    vw_scale: Optional[torch.Tensor],
):
    if not getattr(attention, "_attention_wv_enabled", False):
        return
    reference = reference_context.detach()
    denominator = reference.float().square().mean().detach().clamp_min(1.0e-8)
    context_nmse = (
        (quantized_context.float() - reference.float()).square().mean()
        / denominator
    )
    reference_flat = reference.float().reshape(reference.shape[0], -1)
    quantized_flat = quantized_context.float().reshape(
        quantized_context.shape[0], -1
    )
    direction = (
        1.0
        - F.cosine_similarity(
            quantized_flat, reference_flat, dim=-1, eps=1.0e-8
        ).mean()
    )
    mass = context_nmse.new_zeros(())
    if getattr(attention, "_attention_wv_role", None) == "attn1":
        probability = quantized_w
        if vw_scale is not None:
            probability = probability / vw_scale.reshape(1, 1, -1)
        mass = (
            probability.float().sum(dim=-1)
            - softmax_attention.detach().float().sum(dim=-1)
        ).square().mean()
    attention._attention_wv_aux_terms = {
        "context": context_nmse,
        "direction": direction,
        "mass": mass,
    }


