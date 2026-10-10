"""Channel-wise range proposals for AsyDoBi variants.

Every feature channel independently selects one configured percentile range
by exact QDQ-MSE, then the complete quantizer vector is handled atomically.
Legacy channel-wise hierarchical mode may structurally accept or roll it back;
Post-H v2 commits every numerically valid vector as calibration initialization.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from .hadamard_utils import deterministic_even_indices


F3_ROOT = "input_blocks.2.1.transformer_blocks.0.attn2"
F3_PROJECTION_TARGETS = tuple(
    f"{F3_ROOT}.{suffix}" for suffix in ("to_q", "to_k", "to_v", "to_out.0")
)
STEM_W8A8_TARGET = "input_blocks.0.0"


def mark_channelwise_axis(quantizer: nn.Module, axis: Optional[int], role: str):
    """Attach explicit, non-persistent layout metadata to an activation QDQ."""

    quantizer.channelwise_axis = axis
    quantizer.channelwise_role = str(role)


def _set_quantizer_bits(
    quantizer: nn.Module, bits: int, *, fixed_precision_reason: str
):
    if quantizer is None:
        return
    local_config = copy.deepcopy(quantizer.quantize_config)
    local_config.quant_bits = int(bits)
    if hasattr(local_config, "config_dict"):
        local_config.config_dict = dict(local_config.config_dict)
        local_config.config_dict["quant_bits"] = int(bits)
    quantizer.quantize_config = local_config
    quantizer.quant_bits = int(bits)
    quantizer.fixed_precision = True
    quantizer.fixed_precision_reason = str(fixed_precision_reason)


def _apply_fixed_precision_override(
    model: nn.Module,
    *,
    targets: Sequence[str],
    bits: int,
    reason: str,
):
    modules = dict(model.named_modules())
    missing = [name for name in targets if name not in modules]
    if missing:
        raise KeyError(f"{reason} targets are absent: {missing}")

    report = []
    for name in targets:
        layer = modules[name]
        if not hasattr(layer, "weight_quantizer_0") or not hasattr(
            layer, "act_quantizer_0"
        ):
            raise TypeError(f"{reason} target is not a quantized layer: {name}")
        for attribute in (
            "weight_quantizer_0",
            "weight_quantizer_1",
            "act_quantizer_0",
            "act_quantizer_1",
        ):
            _set_quantizer_bits(
                layer._modules.get(attribute),
                bits,
                fixed_precision_reason=reason,
            )
        layer.quantize_config = copy.deepcopy(layer.quantize_config)
        layer.quantize_config.weight_config.quant_bits = int(bits)
        layer.quantize_config.act_config.quant_bits = int(bits)
        if hasattr(layer.quantize_config, "quantize_config"):
            layer.quantize_config.quantize_config = dict(
                layer.quantize_config.quantize_config
            )
            layer.quantize_config.quantize_config["weight_quant_bits"] = int(bits)
            layer.quantize_config.quantize_config["act_quant_bits"] = int(bits)
        layer.fixed_precision = True
        layer.fixed_precision_reason = str(reason)
        report.append(
            {
                "module": name,
                "weight_bits": int(layer.weight_quantizer_0.quant_bits),
                "activation_bits": int(layer.act_quantizer_0.quant_bits),
                "fixed_precision_reason": str(reason),
            }
        )
    return report


def apply_stem_w8a8(model: nn.Module, config: dict):
    """Apply the config-controlled construction-time Stem W8A8 override."""

    mixed = dict(config.get("mixed_precision", {}))
    stem = dict(mixed.get("stem_w8a8", {}))
    if not bool(stem.get("enabled", False)):
        return []
    target = str(stem.get("target", STEM_W8A8_TARGET))
    if target != STEM_W8A8_TARGET:
        raise ValueError(f"Stem W8A8 fixes target={STEM_W8A8_TARGET}")
    return _apply_fixed_precision_override(
        model,
        targets=(target,),
        bits=8,
        reason="stem_w8a8",
    )


def apply_f3_projection_w8a8(model: nn.Module, config: dict):
    """Apply the exact four-projection F3 W8A8 override.

    The override is run after module refactoring both in PTQ and inference
    construction.  Config objects are copied per layer so changing F3 cannot
    accidentally mutate every other quantizer that shares the base config.
    """

    mixed = dict(config.get("mixed_precision", {}))
    f3 = dict(mixed.get("f3_projection_w8a8", {}))
    if not bool(f3.get("enabled", False)):
        return []

    requested = tuple(f3.get("targets", F3_PROJECTION_TARGETS))
    if requested != F3_PROJECTION_TARGETS:
        raise ValueError("F3 W8A8 fixes the four canonical projection targets")
    return _apply_fixed_precision_override(
        model,
        targets=requested,
        bits=8,
        reason="f3_projection_w8a8",
    )


