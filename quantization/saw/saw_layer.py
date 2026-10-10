import math

import torch
import torch.nn as nn
from .quant_config import Quantize_config, QuantizeModel_config
from .quant_layer import BasicQuantLayer, QuantLayer
from tqdm import tqdm, trange
import numpy as np
from typing import Union
import torch.nn.functional as F
from .hadamard_utils import (
    deterministic_even_indices,
    flatten_channel_vectors,
    normalized_block_fwht,
    rotate_weight_input_axis,
)


class HadamardQuantMixin:
    """Runtime and calibration state shared by SAW Linear/1x1 Conv layers."""

    def _apply(self, fn, recurse=True):
        # Model-wide .half()/.to() must not round the preserved master factors.
        preserve = getattr(self, "_osr_mode", None) in {"precal_fp_preserved_subspace", "precal_joint_wa_input_projector"}
        factors = ({name: getattr(self, name).detach().clone()
                    for name in ("osr_lora_A", "osr_lora_B")} if preserve else {})
        # PyTorch 2.0.1 (the baseline environment) has no recurse keyword.
        # Omitting it for the normal recursive case also preserves newer behavior.
        result = super()._apply(fn) if recurse else super()._apply(fn, recurse=False)
        for name, original in factors.items():
            target = getattr(self, name)
            target.data = original.to(device=target.device, dtype=torch.float32)
        return result

    def _init_hadamard_state(self, quantize_config, channel_dim):
        self.hadamard_dobi_enabled = bool(
            getattr(quantize_config, "hadamard_dobi", {}).get("enabled", False)
        )
        self._hadamard_channel_dim = channel_dim
        self.register_buffer("hadamard_enabled", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("hadamard_block_size", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "_hadamard_weight_cache", torch.empty(0), persistent=False
        )
        self._hadamard_cached_scale_version = None
        self._hadamard_runtime_enabled = False
        self._hadamard_runtime_block_size = 0
        self._hadamard_collecting = False
        self._hadamard_max_vectors = 0
        self._hadamard_expected_calls = 1
        self._hadamard_calls_seen = 0
        self._hadamard_calls_with_samples = 0
        self._hadamard_cache_dtype = torch.float16
        self._hadamard_input_cache = []
        self._hadamard_cached_vectors = 0

    def _init_osr_state(self):
        """Initialize optional post-PTQ residual-scale/LoRA training state."""
        self.register_parameter("osr_residual_log_scale", None)
        self.register_parameter("osr_post_h_scale", None)
        # V4 stores a fixed positive post-H scale directly.  V5 keeps a
        # trainable log-scale instead so positivity is structural and the
        # numerical safety interval can be projected after every optimizer
        # update without changing the immutable H-First LET scale.
        self.register_parameter("osr_post_h_log_scale", None)
        self.register_parameter("osr_lora_A", None)
        self.register_parameter("osr_lora_B", None)
        self._osr_scale_kappa = 2.0
        self._osr_lora_alpha = 0.0
        self._osr_lora_rank = 0
        self._osr_lora_runtime_enabled = True
        self._osr_post_h_runtime_enabled = True
        self._osr_mode = "v3_pre_h_a4_branch"
        self._osr_post_h_log_min = math.log(0.25)
        self._osr_post_h_log_max = math.log(4.0)
        # V3 decomposes the weight *after* LET/H:
        #
        #   W_t = R + BA
        #   y   = Q_A(x_t) Q_W(R) + dequant(Q_A(x_t)) BA
        #
        # The V3 low-rank branch therefore never sees pre-A4 information and
        # is retained at deployment rather than merged back into W4.
        self._osr_detach_residual_for_lora = True
        # V5 keeps FP32 master factors for AdamW, but explicitly rounds both
        # factors (and the bypass activation) through FP16 in every forward so
        # training sees the same numerical branch that is saved for deployment.
        self._osr_simulate_fp16_factors = False
        self._osr_capture_auxiliary = False
        self._osr_auxiliary_losses = {}

    @property
    def osr_enabled(self):
        return any(
            parameter is not None
            for parameter in (
                self.osr_residual_log_scale,
                self.osr_post_h_scale,
                self.osr_post_h_log_scale,
                self.osr_lora_A,
                self.osr_lora_B,
            )
        )

    def configure_osr(
        self,
        *,
        use_residual_scale,
        rank,
        alpha,
        scale_kappa=2.0,
        initialize_random=True,
    ):
        """Install residual smoothing and a low-rank compensation branch."""
        rank = min(int(rank), int(self.weight.shape[0]), int(self.weight[0].numel()))
        if rank <= 0:
            raise ValueError(f"OSR rank must be positive, got {rank}")
        scale_kappa = float(scale_kappa)
        if scale_kappa <= 1.0:
            raise ValueError("OSR scale_kappa must be greater than 1")
        if int(self.fwd_kwargs.get("groups", 1)) != 1:
            raise ValueError(
                "Smooth-Residual V3 currently requires groups=1 so its "
                "factorized Conv branch is exactly equivalent to BA"
            )

        if use_residual_scale:
            self.osr_residual_log_scale = nn.Parameter(
                torch.zeros_like(self.scale_factor, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.osr_residual_log_scale = None
        self.osr_post_h_scale = None
        self.osr_post_h_log_scale = None

        flat_input_features = int(self.weight[0].numel())
        self.osr_lora_A = nn.Parameter(
            torch.empty(
                rank,
                flat_input_features,
                device=self.weight.device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.osr_lora_B = nn.Parameter(
            torch.zeros(
                int(self.weight.shape[0]),
                rank,
                device=self.weight.device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        if initialize_random:
            nn.init.kaiming_uniform_(self.osr_lora_A, a=math.sqrt(5))
        else:
            with torch.no_grad():
                self.osr_lora_A.zero_()
        self._osr_scale_kappa = scale_kappa
        self._osr_lora_alpha = float(alpha)
        self._osr_lora_rank = rank
        self._osr_lora_runtime_enabled = True
        self._osr_post_h_runtime_enabled = True
        self._osr_mode = "v3_pre_h_a4_branch"
        self._osr_detach_residual_for_lora = True
        self._osr_simulate_fp16_factors = False
        self.invalidate_hadamard_weight_cache()
        return rank

    def configure_posth_svd_residual(
        self,
        *,
        use_post_h_scale,
        rank,
        alpha,
    ):
        """Install V4's fixed post-H migration and retained FP16 branch.

        V4 deliberately does not reuse ``osr_residual_log_scale``: that
        parameter is multiplied into LET before Hadamard in V3, whereas this
        scale must operate in the real post-H activation coordinate system.
        """
        actual_rank = self.configure_osr(
            use_residual_scale=False,
            rank=rank,
            alpha=alpha,
            scale_kappa=2.0,
        )
        if use_post_h_scale:
            self.osr_post_h_scale = nn.Parameter(
                torch.ones_like(self.scale_factor, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.osr_post_h_scale = None
        self.osr_post_h_log_scale = None
        self._osr_mode = "v4_post_h_fp_bypass"
        self.invalidate_hadamard_weight_cache()
        return actual_rank

    def configure_precal_preserved_subspace(self, *, rank=32):
        """Install frozen FP32 factors in the pre-LET/pre-H master coordinate."""
        if not isinstance(self, (saw_Linear_QuantLayer, saw_Conv2d_QuantLayer)):
            raise TypeError("Precal preservation supports Linear and Conv2d 1x1")
        if self.weight.ndim == 4 and tuple(self.weight.shape[2:]) != (1, 1):
            raise ValueError("Precal preservation requires a 1x1 kernel")
        if int(rank) <= 0 or int(rank) > min(self.weight.shape[:2]):
            raise ValueError("Preservation rank exceeds master weight dimensions")
        actual_rank = self.configure_osr(
            use_residual_scale=False, rank=rank, alpha=rank, initialize_random=False
        )
        self._osr_mode = "precal_fp_preserved_subspace"
        self._osr_detach_residual_for_lora = False
        # Deterministic allocation: construction supplies the real factors.
        with torch.no_grad():
            self.osr_lora_A.zero_()
        self.set_osr_train_phase("frozen")
        return actual_rank

    def _osr_current_preserved_down_factor(self):
        # Keep the LET dependency in autograd even though master A/B are frozen.
        a = self.osr_lora_A
        if a.dtype != torch.float32 or self.osr_lora_B.dtype != torch.float32:
            raise TypeError("Precal preservation factors must remain FP32")
        if self.recon_flag:
            a = a * self.osr_effective_scale_factor().reshape(1, -1).float()
        if self.hadamard_runtime_enabled:
            a = rotate_weight_input_axis(a, self.hadamard_runtime_block_size)
        return a

    def configure_precal_joint_wa_input_projector(self, *, rank=32):
        actual_rank = self.configure_precal_preserved_subspace(rank=rank)
        self._osr_mode = "precal_joint_wa_input_projector"
        return actual_rank

    def _osr_current_input_projector_up_factor(self):
        """C_t = H S_t^-1 V, with the current LET dependency in autograd."""
        up_t = self.osr_lora_A
        if self.recon_flag:
            up_t = up_t / self.osr_effective_scale_factor().reshape(1, -1).float()
        if self.hadamard_runtime_enabled:
            up_t = rotate_weight_input_axis(up_t, self.hadamard_runtime_block_size)
        return up_t.T

    def precal_joint_activation_decomposition(self, value):
        """Return h and z_res before A4, without materializing a dense projector."""
        with torch.autocast(device_type=value.device.type, enabled=False):
            a = self._osr_current_preserved_down_factor()
            c = self._osr_current_input_projector_up_factor()
            if self.weight.ndim == 2:
                hidden = F.linear(value.float(), a)
                hard = F.linear(hidden, c)
            else:
                # Projection preserves input positions; operator stride/padding
                # applies later to the output branches, not to z_hard.
                hidden = F.conv2d(value.float(), a[:, :, None, None])
                hard = F.conv2d(hidden, c[:, :, None, None])
            return hidden, value.float() - hard

    def _osr_precal_joint_wa_input_forward(self, value, weight, bias):
        with torch.autocast(device_type=value.device.type, enabled=False):
            hidden, residual_input = self.precal_joint_activation_decomposition(value)
            tail_collector = getattr(
                self, "_expanded_two_set_tail_collector", None
            )
            if tail_collector is not None:
                tail_collector(value.detach(), residual_input.detach())
            a = self._osr_current_preserved_down_factor()
            residual = weight.float() - (self.osr_lora_B @ a).reshape_as(weight)
            # This is the only A4 call: AsyDoBi hooks observe z_res directly.
            quantized_input = self._osr_quantize_activation(residual_input)
            quantized_weight = self._osr_quantize_weight(residual)
            output = self.fwd_func(quantized_input, quantized_weight.to(quantized_input.dtype),
                                   bias, **self.fwd_kwargs)
            if self.weight.ndim == 2:
                branch = F.linear(hidden, self.osr_lora_B)
            else:
                branch = F.conv2d(hidden, self.osr_lora_B[:, :, None, None], None, **self.fwd_kwargs)
            return output + branch, residual, quantized_weight

    def precal_transformed_weights(self):
        """Return the current full, residual and preserved weights (FP32)."""
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            full = self.weight.float()
            if self.recon_flag:
                shape = (1, full.shape[1]) + (1,) * (full.ndim - 2)
                full = full * self.osr_effective_scale_factor().reshape(shape).float()
            if self.hadamard_runtime_enabled:
                full = rotate_weight_input_axis(full, self.hadamard_runtime_block_size)
            preserved = (self.osr_lora_B @ self._osr_current_preserved_down_factor()).reshape_as(full)
            return full, full - preserved, preserved

    def _osr_precal_preserved_forward(self, input, weight, bias, branch_input):
        if branch_input is None:
            raise RuntimeError("Preservation requires the pre-A4 activation")
        with torch.autocast(device_type=weight.device.type, enabled=False):
            a = self._osr_current_preserved_down_factor()
            b = self.osr_lora_B
            residual = weight.float() - (b @ a).reshape_as(weight)
            value = branch_input.float()
            if self.weight.ndim == 2:
                branch = F.linear(F.linear(value, a), b)
            else:
                hidden = F.conv2d(value, a[:, :, None, None], None, **self.fwd_kwargs)
                branch = F.conv2d(hidden, b[:, :, None, None])
        quantized = self._osr_quantize_weight(residual)
        output = self.fwd_func(input, quantized.to(input.dtype), bias, **self.fwd_kwargs)
        return output + branch, residual, quantized

    def configure_posth_v5_core(
        self,
        *,
        rank,
        alpha,
        scale_min=0.25,
        scale_max=4.0,
    ):
        """Install V5's trainable post-H transfer/decomposition unit.

        The original weight, bias, H-First LET/Hadamard state, and non-target
        quantizers remain untouched.  Only this log-scale, the target A4/W4
        ranges, and the retained rank-``rank`` factors are optimized by the
        V5 trainer.
        """
        scale_min = float(scale_min)
        scale_max = float(scale_max)
        if not (0.0 < scale_min < scale_max):
            raise ValueError(
                "V5 post-H scale bounds must satisfy 0 < min < max"
            )
        actual_rank = self.configure_osr(
            use_residual_scale=False,
            rank=rank,
            alpha=alpha,
            scale_kappa=2.0,
        )
        self.osr_post_h_scale = None
        self.osr_post_h_log_scale = nn.Parameter(
            torch.zeros_like(self.scale_factor, dtype=torch.float32),
            requires_grad=False,
        )
        self._osr_post_h_log_min = math.log(scale_min)
        self._osr_post_h_log_max = math.log(scale_max)
        self._osr_mode = "v5_post_h_fp_bypass"
        # Unlike V3/V4, V5 needs the -BA residual path and +BA FP16 path to
        # form one differentiable decomposition.  Their non-cancelling signal
        # is driven by X_s - Q_A4(X_s), the activation error V5 is designed to
        # absorb.
        self._osr_detach_residual_for_lora = False
        self._osr_simulate_fp16_factors = True
        self.invalidate_hadamard_weight_cache()
        return actual_rank

    def set_osr_train_phase(self, phase):
        """Freeze every OSR parameter except those belonging to ``phase``."""
        if self._osr_mode in {"precal_fp_preserved_subspace", "precal_joint_wa_input_projector"} and phase != "frozen":
            raise ValueError("Precal preservation factors cannot be trained")
        if phase not in {
            "scale",
            "lora",
            "residual",
            "joint",
            "v5_joint",
            "frozen",
        }:
            raise ValueError(f"Unknown OSR train phase {phase!r}")
        if self.osr_residual_log_scale is not None:
            self.osr_residual_log_scale.requires_grad_(phase == "scale")
        if self.osr_lora_A is not None:
            self.osr_lora_A.requires_grad_(
                phase in {"lora", "residual", "joint", "v5_joint"}
            )
        if self.osr_lora_B is not None:
            self.osr_lora_B.requires_grad_(
                phase in {"lora", "residual", "joint", "v5_joint"}
            )
        if self.osr_post_h_scale is not None:
            self.osr_post_h_scale.requires_grad_(False)
        if self.osr_post_h_log_scale is not None:
            self.osr_post_h_log_scale.requires_grad_(phase == "v5_joint")

    def osr_residual_scale(self):
        if self.osr_residual_log_scale is None:
            return torch.ones_like(self.scale_factor)
        log_kappa = math.log(self._osr_scale_kappa)
        return torch.exp(log_kappa * torch.tanh(self.osr_residual_log_scale))

    def osr_effective_scale_factor(self):
        return self.scale_factor * self.osr_residual_scale()

    def osr_post_h_scale_value(self):
        if self.osr_post_h_log_scale is not None:
            bounded = self.osr_post_h_log_scale.clamp(
                min=float(self._osr_post_h_log_min),
                max=float(self._osr_post_h_log_max),
            )
            return torch.exp(bounded)
        if self.osr_post_h_scale is None:
            return torch.ones_like(self.scale_factor)
        return self.osr_post_h_scale

    @property
    def osr_uses_post_h_fp_bypass(self):
        return self._osr_mode in {
            "precal_fp_preserved_subspace",
            "precal_joint_wa_input_projector",
            "v4_post_h_fp_bypass",
            "v5_post_h_fp_bypass",
        }

    @property
    def osr_is_v5_core(self):
        return self._osr_mode == "v5_post_h_fp_bypass"

    def set_osr_post_h_scale_(self, value):
        parameter = (
            self.osr_post_h_log_scale
            if self.osr_post_h_log_scale is not None
            else self.osr_post_h_scale
        )
        if parameter is None:
            raise RuntimeError("This layer has no post-H migration scale")
        value = value.detach().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        if value.shape != parameter.shape:
            value = value.reshape_as(parameter)
        if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
            raise ValueError("post-H migration scale must be finite and positive")
        with torch.no_grad():
            if self.osr_post_h_log_scale is not None:
                log_value = torch.log(value).clamp(
                    min=float(self._osr_post_h_log_min),
                    max=float(self._osr_post_h_log_max),
                )
                self.osr_post_h_log_scale.copy_(log_value)
            else:
                self.osr_post_h_scale.copy_(value)
        self.invalidate_hadamard_weight_cache()

    @torch.no_grad()
    def project_osr_post_h_log_scale_(self):
        """Project V5's trainable log-scale onto its safety interval."""
        if self.osr_post_h_log_scale is None:
            return False
        if not bool(torch.isfinite(self.osr_post_h_log_scale).all()):
            raise FloatingPointError("V5 post-H log-scale became non-finite")
        before = self.osr_post_h_log_scale.detach().clone()
        self.osr_post_h_log_scale.clamp_(
            min=float(self._osr_post_h_log_min),
            max=float(self._osr_post_h_log_max),
        )
        changed = not torch.equal(before, self.osr_post_h_log_scale)
        if changed:
            self.invalidate_hadamard_weight_cache()
        return changed

    def _apply_osr_post_h_scale(self, input, weight):
        if (
            not self.osr_uses_post_h_fp_bypass
            or not self._osr_post_h_runtime_enabled
            or (
                self.osr_post_h_scale is None
                and self.osr_post_h_log_scale is None
            )
        ):
            return input, weight
        scale = self.osr_post_h_scale_value().to(
            device=input.device, dtype=input.dtype
        )
        return input / scale, weight * scale.to(device=weight.device, dtype=weight.dtype)

    def _osr_forward_factors(self, compute_dtype):
        """Return differentiable deployment-simulated A/B factor values."""
        a = self.osr_lora_A
        b = self.osr_lora_B
        if self._osr_simulate_fp16_factors:
            # The two casts intentionally retain a FP32 master parameter while
            # making FP16 rounding part of the differentiable forward graph.
            a = a.to(torch.float16)
            b = b.to(torch.float16)
        return a.to(dtype=compute_dtype), b.to(dtype=compute_dtype)

    def osr_lora_delta(self):
        if (
            not self._osr_lora_runtime_enabled
            or self.osr_lora_A is None
            or self.osr_lora_B is None
        ):
            return torch.zeros_like(self.weight)
        scale = self._osr_lora_alpha / float(self._osr_lora_rank)
        compute_dtype = (
            torch.float16
            if self._osr_simulate_fp16_factors and self.weight.is_cuda
            else self.osr_lora_A.dtype
        )
        a, b = self._osr_forward_factors(compute_dtype)
        delta = (b @ a) * scale
        return delta.reshape_as(self.weight).to(dtype=self.weight.dtype)

    def osr_effective_weight(self):
        # BA lives in the deployed post-LET/post-H coordinate system and is a
        # decomposition of the transformed weight, not an update to the
        # pre-H master weight.
        return self.weight

    def set_osr_lora_runtime_enabled(self, enabled):
        """Temporarily include/exclude BA without changing its parameters.

        Paired V3 branches use this switch to keep their activation path
        identical while changing only the target weight representation.
        """
        self._osr_lora_runtime_enabled = bool(enabled)
        self.invalidate_hadamard_weight_cache()

    def set_posth_svd_residual_runtime_enabled(self, enabled):
        """Atomically enable or bypass a deployed V4 target.

        This switch is intentionally stricter than
        :meth:`set_osr_lora_runtime_enabled`.  A causal ablation target is
        ``off`` only when both the retained BA branch and its post-H migration
        are bypassed.  The caller must restore the matching H-First or V4
        A4/W4 range snapshot at the same time.
        """
        if not self.osr_uses_post_h_fp_bypass:
            raise RuntimeError("This layer is not configured as a Post-H V4/V5 target")
        enabled = bool(enabled)
        self._osr_lora_runtime_enabled = enabled
        self._osr_post_h_runtime_enabled = enabled
        self.invalidate_hadamard_weight_cache()

    def set_osr_residual_scale_(self, value):
        """Set the bounded residual scale from a positive tensor value."""
        if self.osr_residual_log_scale is None:
            raise RuntimeError("This layer has no OSR residual scale")
        value = value.detach().to(
            device=self.osr_residual_log_scale.device,
            dtype=self.osr_residual_log_scale.dtype,
        )
        if value.shape != self.osr_residual_log_scale.shape:
            value = value.reshape_as(self.osr_residual_log_scale)
        log_kappa = math.log(self._osr_scale_kappa)
        normalized = (torch.log(value.clamp_min(1.0e-12)) / log_kappa).clamp(
            min=-1.0 + 1.0e-6,
            max=1.0 - 1.0e-6,
        )
        with torch.no_grad():
            self.osr_residual_log_scale.copy_(torch.atanh(normalized))
        self.invalidate_hadamard_weight_cache()

    def merge_osr_lora_(self):
        """V3 retains its FP16 branch and never merges it back into W4."""
        raise RuntimeError(
            "Smooth-Residual V3 keeps its FP16 low-rank branch and must "
            "not be merged into the W4 master weight"
        )

    def merge_osr_scale_(self):
        """Fold only the residual scale into the existing LET scale."""
        if self.osr_residual_log_scale is None:
            return
        with torch.no_grad():
            merged_scale = self.osr_effective_scale_factor().detach().clone()
            self.scale_factor.copy_(merged_scale)
        self.osr_residual_log_scale = None
        self.invalidate_hadamard_weight_cache()

    @property
    def osr_smooth_residual_enabled(self):
        return bool(
            self._osr_lora_runtime_enabled
            and self.osr_lora_A is not None
            and self.osr_lora_B is not None
        )

    def osr_set_deploy_dtype_(self, dtype=torch.float16):
        """Materialize the retained V3 branch in its deployment dtype."""
        if self._osr_mode in {"precal_fp_preserved_subspace", "precal_joint_wa_input_projector"} and dtype != torch.float32:
            raise ValueError("Precal preservation deployment is FP32")
        for name in ("osr_lora_A", "osr_lora_B"):
            parameter = getattr(self, name)
            if parameter is not None and parameter.dtype != dtype:
                setattr(
                    self,
                    name,
                    nn.Parameter(
                        parameter.detach().to(dtype=dtype),
                        requires_grad=False,
                    ),
                )
        self.invalidate_hadamard_weight_cache()

    def _osr_quantize_activation(self, value):
        if not self.use_act_quant:
            return value
        if self.split:
            divide = value.shape[1] // 2
            return torch.cat(
                (
                    self.act_quantizer_0(value[:, :divide, ...]),
                    self.act_quantizer_1(value[:, divide:, ...]),
                ),
                dim=1,
            )
        return self.act_quantizer_0(value)

    def _osr_quantize_weight(self, value):
        if not self.use_weight_quant:
            return value
        if self.split:
            divide = value.shape[1] // 2
            return torch.cat(
                (
                    self.weight_quantizer_0(value[:, :divide, ...]),
                    self.weight_quantizer_1(value[:, divide:, ...]),
                ),
                dim=1,
            )
        return self.weight_quantizer_0(value)

    def _osr_factorized_residual_forward(self, value):
        """Evaluate BA without materializing a second dense weight operator."""
        if not self.osr_smooth_residual_enabled:
            return None
        # CUDA deploy uses FP16 A/B.  CPU unit tests retain the input dtype
        # because half-precision convolution support is backend dependent.
        compute_dtype = (
            torch.float16
            if self._osr_simulate_fp16_factors and value.is_cuda
            else (self.osr_lora_A.dtype if value.is_cuda else value.dtype)
        )
        branch_input = value
        if self._osr_simulate_fp16_factors:
            branch_input = branch_input.to(torch.float16)
        branch_input = branch_input.to(dtype=compute_dtype)
        scale = self._osr_lora_alpha / float(self._osr_lora_rank)
        a, b = self._osr_forward_factors(compute_dtype)
        if isinstance(self, saw_Linear_QuantLayer):
            hidden = F.linear(branch_input, a)
            output = F.linear(hidden, b) * scale
        elif isinstance(self, saw_Conv2d_QuantLayer):
            a = a.reshape(
                self._osr_lora_rank,
                self.weight.shape[1],
                *self.weight.shape[2:],
            )
            hidden = F.conv2d(branch_input, a, None, **self.fwd_kwargs)
            output = F.conv2d(
                hidden,
                b.reshape(self.weight.shape[0], self._osr_lora_rank, 1, 1),
                None,
            ) * scale
        elif isinstance(self, saw_Conv1d_QuantLayer):
            a = a.reshape(
                self._osr_lora_rank,
                self.weight.shape[1],
                self.weight.shape[2],
            )
            hidden = F.conv1d(branch_input, a, None, **self.fwd_kwargs)
            output = F.conv1d(
                hidden,
                b.reshape(self.weight.shape[0], self._osr_lora_rank, 1),
                None,
            ) * scale
        else:
            raise TypeError(f"Unsupported Smooth-Residual layer {type(self)!r}")
        return output.to(dtype=value.dtype)

    def _osr_weight_and_residual_forward(
        self, input, weight, bias, *, branch_input=None
    ):
        """Run the base path or a retained dual branch.

        V3 passes the same A4-dequantized activation to both branches. V4
        supplies the post-H/post-migration activation captured immediately
        before A4, so its low-rank subspace is a genuine high-precision bypass.
        """
        if self._osr_mode == "precal_joint_wa_input_projector" and self.osr_smooth_residual_enabled:
            return self._osr_precal_joint_wa_input_forward(input, weight, bias)
        if self._osr_mode == "precal_fp_preserved_subspace" and self.osr_smooth_residual_enabled:
            return self._osr_precal_preserved_forward(input, weight, bias, branch_input)
        if self.osr_smooth_residual_enabled and self.use_weight_quant:
            low_rank = self.osr_lora_delta().to(
                device=weight.device, dtype=weight.dtype
            )
            # Forward values always use the latest BA.  During A/B training the
            # main W4 branch is stop-gradient with respect to BA, preventing
            # the -BA and +BA paths around STE from cancelling each other.
            residual_low_rank = (
                low_rank.detach()
                if self._osr_detach_residual_for_lora
                else low_rank
            )
            residual_weight = weight - residual_low_rank
            quantized_weight = self._osr_quantize_weight(residual_weight)
            output = self.fwd_func(
                input, quantized_weight, bias, **self.fwd_kwargs
            )
            branch = self._osr_factorized_residual_forward(
                input if branch_input is None else branch_input
            )
            return output + branch, residual_weight, quantized_weight

        quantized_weight = self._osr_quantize_weight(weight)
        output = self.fwd_func(
            input, quantized_weight, bias, **self.fwd_kwargs
        )
        return output, weight, quantized_weight

    def merge_osr_(self):
        """Fold OSR parameters into the pre-H weight and existing LET scale."""
        self.merge_osr_lora_()
        self.merge_osr_scale_()

    def set_osr_auxiliary_capture(self, enabled):
        self._osr_capture_auxiliary = bool(enabled)
        self._osr_auxiliary_losses = {}

    @staticmethod
    def _osr_normalized_mse(value, reference, epsilon=1.0e-8):
        numerator = (value.float() - reference.float()).square().mean()
        denominator = reference.detach().float().square().mean().clamp_min(epsilon)
        return numerator / denominator

    @staticmethod
    def _osr_soft_clip_loss(value, quantizer, epsilon=1.0e-8):
        if not hasattr(quantizer, "x_min") or not hasattr(quantizer, "x_max"):
            return value.new_zeros((), dtype=torch.float32)
        lower = quantizer.x_min.detach().to(device=value.device, dtype=value.dtype)
        upper = quantizer.x_max.detach().to(device=value.device, dtype=value.dtype)
        overflow = torch.relu(value - upper).float().square().mean()
        underflow = torch.relu(lower - value).float().square().mean()
        denominator = value.detach().float().square().mean().clamp_min(epsilon)
        return (overflow + underflow) / denominator

    def _record_osr_auxiliary(
        self,
        activation_reference,
        activation_quantized,
        weight_reference,
        weight_quantized,
        operator_reference=None,
        operator_quantized=None,
    ):
        if not self._osr_capture_auxiliary:
            return
        activation_clip = []
        weight_clip = []
        if self.split:
            activation_split = activation_reference.shape[1] // 2
            weight_split = weight_reference.shape[1] // 2
            activation_clip.extend(
                (
                    self._osr_soft_clip_loss(
                        activation_reference[:, :activation_split, ...],
                        self.act_quantizer_0,
                    ),
                    self._osr_soft_clip_loss(
                        activation_reference[:, activation_split:, ...],
                        self.act_quantizer_1,
                    ),
                )
            )
            weight_clip.extend(
                (
                    self._osr_soft_clip_loss(
                        weight_reference[:, :weight_split, ...],
                        self.weight_quantizer_0,
                    ),
                    self._osr_soft_clip_loss(
                        weight_reference[:, weight_split:, ...],
                        self.weight_quantizer_1,
                    ),
                )
            )
        else:
            activation_clip.append(
                self._osr_soft_clip_loss(
                    activation_reference, self.act_quantizer_0
                )
            )
            weight_clip.append(
                self._osr_soft_clip_loss(
                    weight_reference, self.weight_quantizer_0
                )
            )
        self._osr_auxiliary_losses = {
            "activation_qdq": self._osr_normalized_mse(
                activation_quantized, activation_reference
            ),
            "weight_qdq": self._osr_normalized_mse(
                weight_quantized, weight_reference
            ),
            "soft_clip": torch.stack(
                [loss.float() for loss in activation_clip + weight_clip]
            ).mean(),
        }
        if operator_reference is not None and operator_quantized is not None:
            self._osr_auxiliary_losses["operator_qdq"] = (
                self._osr_normalized_mse(
                    operator_quantized,
                    operator_reference,
                )
            )

    def _osr_operator_reference(self, activation, weight, bias):
        """Detached live-input FP operator target used only during V5 training."""
        if not self._osr_capture_auxiliary:
            return None
        detached_bias = None if bias is None else bias.detach()
        return self.fwd_func(
            activation.detach(),
            weight.detach(),
            detached_bias,
            **self.fwd_kwargs,
        ).detach()

    def osr_auxiliary_losses(self):
        return dict(self._osr_auxiliary_losses)

    def invalidate_hadamard_weight_cache(self):
        self._hadamard_weight_cache = torch.empty(
            0, device=self.weight.device, dtype=self.weight.dtype
        )
        self._hadamard_cached_scale_version = None

    def _hadamard_weight_cache_version(self):
        # Tensors materialized by an inference-only device transfer do not
        # expose PyTorch's in-place version counter. Such tensors are immutable
        # outside inference mode, so object identity is a sufficient fallback
        # for cache invalidation.
        try:
            weight_version = int(self.weight._version)
        except RuntimeError as error:
            if "Inference tensors do not track version counter" not in str(error):
                raise
            weight_version = None
        versions = [id(self.weight), weight_version]
        if self.recon_flag and hasattr(self, "scale_factor"):
            versions.append(int(self.scale_factor._version))
        for parameter in (
            self.osr_residual_log_scale,
            self.osr_post_h_scale,
            self.osr_post_h_log_scale,
            self.osr_lora_A,
            self.osr_lora_B,
        ):
            versions.append(None if parameter is None else int(parameter._version))
        return tuple(versions)

    def _sync_hadamard_runtime_state(self):
        """Copy persistent checkpoint buffers to CPU-side forward state once."""
        self._hadamard_runtime_enabled = bool(self.hadamard_enabled.detach().cpu().item())
        self._hadamard_runtime_block_size = int(
            self.hadamard_block_size.detach().cpu().item()
        )

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
        if self._osr_mode in {"precal_fp_preserved_subspace", "precal_joint_wa_input_projector"}:
            for name in ("osr_lora_A", "osr_lora_B"):
                value = state_dict.get(prefix + name)
                if value is None or value.dtype != torch.float32 or not torch.isfinite(value).all():
                    error_msgs.append(f"Missing/non-FP32/non-finite preservation factor: {prefix}{name}")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._sync_hadamard_runtime_state()
        self._hadamard_weight_cache = torch.empty(
            0, device=self.weight.device, dtype=self.weight.dtype
        )
        self._hadamard_cached_scale_version = None

    @property
    def hadamard_runtime_enabled(self):
        return self._hadamard_runtime_enabled

    @property
    def hadamard_runtime_block_size(self):
        return self._hadamard_runtime_block_size

    def begin_hadamard_collection(
        self, max_vectors, expected_calls=1, cache_dtype=torch.float16
    ):
        self._hadamard_collecting = True
        self._hadamard_max_vectors = int(max_vectors)
        self._hadamard_expected_calls = max(1, int(expected_calls))
        self._hadamard_calls_seen = 0
        self._hadamard_calls_with_samples = 0
        self._hadamard_cache_dtype = cache_dtype
        self._hadamard_input_cache = []
        self._hadamard_cached_vectors = 0

    def end_hadamard_collection(self):
        self._hadamard_collecting = False

    def pop_hadamard_cache(self):
        if not self._hadamard_input_cache:
            return None
        cached = torch.cat(self._hadamard_input_cache, dim=0)
        self._hadamard_input_cache = []
        self._hadamard_cached_vectors = 0
        return cached

    def hadamard_collection_stats(self):
        return {
            "calls_seen": int(self._hadamard_calls_seen),
            "contributing_calls": int(self._hadamard_calls_with_samples),
        }

    def _record_hadamard_input(self, input):
        if not self._hadamard_collecting:
            return
        remaining = self._hadamard_max_vectors - self._hadamard_cached_vectors
        if remaining <= 0:
            return
        calls_remaining = max(
            1, self._hadamard_expected_calls - self._hadamard_calls_seen
        )
        per_call_quota = (remaining + calls_remaining - 1) // calls_remaining
        self._hadamard_calls_seen += 1
        vectors = flatten_channel_vectors(input.detach(), self._hadamard_channel_dim)
        take = min(int(vectors.shape[0]), int(per_call_quota), int(remaining))
        if take <= 0:
            return
        if vectors.shape[0] > take:
            positions = deterministic_even_indices(
                vectors.shape[0], take, device=vectors.device
            )
            vectors = vectors.index_select(0, positions)
        vectors = vectors.to(device="cpu", dtype=self._hadamard_cache_dtype)
        self._hadamard_input_cache.append(vectors)
        self._hadamard_cached_vectors += vectors.shape[0]
        self._hadamard_calls_with_samples += 1

    def configure_hadamard(self, enabled, block_size):
        enabled = bool(enabled)
        block_size = int(block_size) if enabled else 0
        self.hadamard_enabled.fill_(enabled)
        self.hadamard_block_size.fill_(block_size)
        self._hadamard_runtime_enabled = enabled
        self._hadamard_runtime_block_size = block_size
        self._hadamard_weight_cache = torch.empty(
            0, device=self.weight.device, dtype=self.weight.dtype
        )
        self._hadamard_cached_scale_version = None

    def _calculate_let_scale(self, input):
        """Calculate a legacy or HarmoQ LET scale along the input-channel axis."""
        config = getattr(
            self.quantize_config, "scale_init", {"mode": "legacy_maxabs"}
        )
        if isinstance(config, str):
            config = {"mode": config}
        if not isinstance(config, dict):
            raise TypeError("scale_init must be a mapping or mode string")

        mode = str(config.get("mode", "legacy_maxabs")).lower()
        epsilon = float(config.get("epsilon", 1.0e-5))
        if epsilon <= 0:
            raise ValueError("scale_init.epsilon must be positive")

        input_stats = input.detach().to(dtype=torch.float32)
        weight_stats = self.weight.detach().to(dtype=torch.float32)
        input_channel_dim = self._hadamard_channel_dim % input_stats.ndim
        weight_channel_dim = 1
        input_reduce_dims = tuple(
            dim for dim in range(input_stats.ndim) if dim != input_channel_dim
        )
        weight_reduce_dims = tuple(
            dim for dim in range(weight_stats.ndim) if dim != weight_channel_dim
        )

        input_maxabs = torch.amax(
            torch.abs(input_stats), dim=input_reduce_dims, keepdim=True
        ).clamp_min(epsilon)
        weight_maxabs = torch.amax(
            torch.abs(weight_stats), dim=weight_reduce_dims, keepdim=True
        ).clamp_min(epsilon)
        baseline_scale = (
            input_maxabs.pow(self.s_alpha) / weight_maxabs.pow(self.s_alpha)
        )

        if mode in {"legacy", "legacy_maxabs", "baseline"}:
            scale = baseline_scale
        elif mode == "harmoq_asymmetric_closed_form":
            input_range = (
                torch.amax(input_stats, dim=input_reduce_dims, keepdim=True)
                - torch.amin(input_stats, dim=input_reduce_dims, keepdim=True)
            ).clamp_min(epsilon)
            weight_range = (
                torch.amax(weight_stats, dim=weight_reduce_dims, keepdim=True)
                - torch.amin(weight_stats, dim=weight_reduce_dims, keepdim=True)
            ).clamp_min(epsilon)
            act_levels = float(
                (1 << int(self.quantize_config.act_config.quant_bits)) - 1
            )
            weight_levels = float(
                (1 << int(self.quantize_config.weight_config.quant_bits)) - 1
            )
            harmoq_scale = torch.sqrt(
                (input_range / act_levels) / (weight_range / weight_levels)
            )

            blend_ratio = float(config.get("blend_ratio", 1.0))
            if not 0.0 <= blend_ratio <= 1.0:
                raise ValueError("scale_init.blend_ratio must be in [0, 1]")
            scale = torch.exp(
                (1.0 - blend_ratio) * torch.log(baseline_scale)
                + blend_ratio * torch.log(harmoq_scale)
            )

            relative_clamp = config.get("relative_clamp")
            if relative_clamp is not None:
                if (
                    not isinstance(relative_clamp, (list, tuple))
                    or len(relative_clamp) != 2
                ):
                    raise ValueError(
                        "scale_init.relative_clamp must contain [min, max]"
                    )
                relative_min = float(relative_clamp[0])
                relative_max = float(relative_clamp[1])
                if relative_min <= 0 or relative_max < relative_min:
                    raise ValueError(
                        "scale_init.relative_clamp requires 0 < min <= max"
                    )
                scale = torch.maximum(
                    torch.minimum(scale, baseline_scale * relative_max),
                    baseline_scale * relative_min,
                )
        else:
            raise ValueError(f"Unsupported scale_init.mode={mode!r}")

        if scale.numel() != self.scale_factor.numel():
            raise RuntimeError(
                "LET scale statistics do not match scale_factor shape: "
                f"stats={tuple(scale.shape)}, parameter={tuple(self.scale_factor.shape)}"
            )
        return scale.reshape_as(self.scale_factor).to(
            device=self.scale_factor.device, dtype=self.scale_factor.dtype
        )

    def _apply_hadamard(self, input, weight):
        self._record_hadamard_input(input)
        if not self._hadamard_runtime_enabled:
            return input, weight
        block_size = self._hadamard_runtime_block_size
        input = normalized_block_fwht(input, block_size, dim=self._hadamard_channel_dim)
        # LET calibration optimizes scale_factor through both equivalent
        # branches: input / scale_factor and weight * scale_factor.  A detached
        # weight cache is valid only when no backward graph is being built.
        # During calibration (or any other grad-enabled training path), rotate
        # the effective weight directly so its scale-factor dependency reaches
        # autograd.
        if torch.is_grad_enabled() and weight.requires_grad:
            return input, rotate_weight_input_axis(weight, block_size)
        scale_version = self._hadamard_weight_cache_version()
        if (
            self._hadamard_weight_cache.numel() == 0
            or self._hadamard_weight_cache.device != weight.device
            or self._hadamard_weight_cache.dtype != weight.dtype
            or self._hadamard_weight_cache.shape != weight.shape
            or self._hadamard_cached_scale_version != scale_version
        ):
            self._hadamard_weight_cache = rotate_weight_input_axis(
                weight.detach(), block_size
            )
            self._hadamard_cached_scale_version = scale_version
        return input, self._hadamard_weight_cache

class saw_Conv1d_QuantLayer(HadamardQuantMixin, QuantLayer):
    """
    A quantized 1D convolutional layer with scale and offset adaptation for quantization-aware training.

    This layer extends a base QuantLayer to support quantization of Conv1d operations, including adaptive scaling and offsetting
    of inputs and weights for improved quantization performance. It supports optional LoRA (Low-Rank Adaptation), activation and 
    weight quantization, and running statistics for scale updating.

    Args:
        org_module (nn.Conv2d): The original convolutional module to be quantized.
        quantize_config: Configuration object for quantization parameters.
        device (str, optional): Device to place the layer parameters on. Default is "cuda".
        s_alpha (float, optional): Exponent for scale adaptation. Default is 0.5.

    Attributes:
        quant_config: Stores the quantization configuration.
        input_embedd: Placeholder for input embedding (if used).
        record_flag (bool): Flag to enable input recording.
        device (str): Device for computation.
        scale_factor (nn.Parameter): Learnable scale factor for input normalization.
        offset (nn.Parameter): Learnable offset for input normalization.
        recon_flag (bool): Flag to enable reconstruction mode.
        recon_inited (bool): Indicates if reconstruction has been initialized.
        s_alpha (float): Exponent for scale adaptation.
        running_stat (bool): Flag to enable running statistics for scale updating.

    Methods:
        init_scale(input): Initializes the scale factor based on input and weight statistics.
        set_running_stat(running_stat): Sets the running statistics flag.
        update_scale(input, update_rate=0.1): Updates the scale factor using running statistics.
        set_record(flag): Sets the input recording flag.
        forward(input): Forward pass with quantization, scaling, offsetting, and optional LoRA and reconstruction.
    """
    def __init__(self,
                 org_module: nn.Conv2d,
                 quantize_config,
                 device="cuda",
                 s_alpha = 0.5
                 ):
        super().__init__(org_module, quantize_config, device)
        self.quant_config = quantize_config
        self._init_hadamard_state(quantize_config, channel_dim=1)
        self.input_embedd = None
        self.record_flag = False
        self.device = device
        scale_factor = torch.ones(1, self.org_weight.shape[1], 1, requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.scale_factor = nn.Parameter(scale_factor, requires_grad=True)
        offset = torch.zeros(1, self.org_weight.shape[1], 1, requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.offset = nn.Parameter(offset, requires_grad=True)
        self.recon_flag = False
        self.recon_inited = False
        if hasattr(quantize_config, 's_alpha'):
            self.s_alpha = quantize_config.s_alpha
        else:
            self.s_alpha = s_alpha
        self.running_stat = False
        self._init_osr_state()

    def init_scale(self, input):
        self.recon_inited = True
        print("Input shape: ", input.shape)
        print("weight shape:", self.weight.shape)
        self.scale_factor.data.copy_(self._calculate_let_scale(input))
        return self.scale_factor

    def set_running_stat(self, running_stat):
        self.running_stat = running_stat

    def update_scale(self, input, update_rate=0.1):
        tmp = self._calculate_let_scale(input)
        self.scale_factor.data = self.scale_factor.data * (1 - update_rate) + tmp * update_rate

    def set_record(self, flag):
        self.record_flag = flag

    def forward(self, input):
        self._osr_auxiliary_losses = {}
        if self.record_flag:
            self.record_x(input)
        effective_weight = self.osr_effective_weight()
        if self.recon_flag:
            if not self.recon_inited:
                self.init_scale(input)
            elif self.running_stat:
                self.update_scale(input)
            effective_scale = self.osr_effective_scale_factor()
            offset = self.offset.expand_as(input)
            input = (input - offset) / effective_scale
            recond_bias = self.fwd_func(
                offset, effective_weight, None, **self.fwd_kwargs
            )
            recond_bias = recond_bias.squeeze(1)
            # weight = nn.Parameter(self.weight / self.scale_factor)
            weight = effective_weight * effective_scale
        else:
            input = input
            weight = effective_weight

        input, weight = self._apply_hadamard(input, weight)
        input, weight = self._apply_osr_post_h_scale(input, weight)
        activation_reference = input
        weight_reference = weight

        if not (self._osr_mode == "precal_joint_wa_input_projector" and self.osr_smooth_residual_enabled):
            input = self._osr_quantize_activation(input)

        if self.use_weight_quant:
            if self.org_bias is not None:
                bias = (
                    self.bias_quantizer(self.bias)
                    if self.bias_quantizer is not None
                    else self.bias
                )
            else:
                bias = self.bias
        else:
            bias = self.org_bias
        operator_reference = self._osr_operator_reference(
            activation_reference, weight_reference, bias
        )
        out, weight_reference_for_qdq, quantized_weight = (
            self._osr_weight_and_residual_forward(
                input,
                weight,
                bias,
                branch_input=(
                    activation_reference
                    if self.osr_uses_post_h_fp_bypass
                    else None
                ),
            )
        )
        self._record_osr_auxiliary(
            activation_reference,
            input,
            weight_reference_for_qdq,
            quantized_weight,
            operator_reference,
            out,
        )
        if self.recon_flag:
            out = out + recond_bias
        out = self.activation_function(out)
        return out

class saw_Conv2d_QuantLayer(HadamardQuantMixin, QuantLayer):
    """
    A quantized 2D convolutional layer with scale and offset adjustment for input reconstruction.

    This layer extends a standard quantized Conv2d layer by introducing learnable scale and offset parameters
    for each input channel, enabling input reconstruction and improved quantization performance. It supports
    both activation and weight quantization, as well as optional input recording for analysis.

    Args:
        org_module (nn.Conv2d): The original Conv2d module to be quantized.
        quantize_config: Configuration object for quantization parameters.
        device (str, optional): Device to place the layer parameters on. Default is "cuda".
        s_alpha (float, optional): Exponent for scale factor calculation. Default is 0.5.

    Attributes:
        quant_config: Stores the quantization configuration.
        input_embedd: Stores recorded input embeddings if recording is enabled.
        record_flag (bool): Flag to enable/disable input recording.
        device (str): Device for computation.
        scale_factor (nn.Parameter): Learnable scale factor for each input channel.
        offset (nn.Parameter): Learnable offset for each input channel.
        recon_flag (bool): Flag to enable/disable input reconstruction.
        recon_inited (bool): Indicates if reconstruction parameters have been initialized.
        s_alpha (float): Exponent used in scale factor calculation.
        running_stat (bool): Flag to enable/disable running statistics for scale update.

    Methods:
        init_scale(input): Initializes the scale factor based on input and weight statistics.
        set_running_stat(running_stat): Sets the running statistics flag.
        update_scale(input, update_rate=0.1): Updates the scale factor using running statistics.
        set_record(flag): Enables or disables input recording.
        forward(input): Forward pass with quantization, optional reconstruction, and activation.
        record_x(x): Records input embeddings for analysis.
    """
    def __init__(self,
                org_module: nn.Conv2d,
                quantize_config,
                device="cuda",
                s_alpha = 0.5,
                ):
        super().__init__(org_module, quantize_config, device)
        self.quant_config = quantize_config
        self._init_hadamard_state(quantize_config, channel_dim=1)
        self.input_embedd = None
        self.record_flag = False
        self.device = device
        scale_factor = torch.ones(1, self.org_weight.shape[1], 1, 1, requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.scale_factor = nn.Parameter(scale_factor, requires_grad=True)
        offset = torch.zeros(1, self.org_weight.shape[1], 1, 1, requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.offset = nn.Parameter(offset, requires_grad=True)
        self.recon_flag = False
        self.recon_inited = False
        if hasattr(quantize_config, 's_alpha'):
            self.s_alpha = quantize_config.s_alpha
        else:
            self.s_alpha = s_alpha
        self.running_stat = False
        self._init_osr_state()

    def init_scale(self, input):
        self.recon_inited = True
        self.scale_factor.data.copy_(self._calculate_let_scale(input))
        return self.scale_factor

    def set_running_stat(self, running_stat):
        self.running_stat = running_stat

    def update_scale(self, input, update_rate=0.1):
        tmp = self._calculate_let_scale(input)
        self.scale_factor.data = self.scale_factor.data * (1 - update_rate) + tmp * update_rate

    def set_record(self, flag):
        self.record_flag = flag

    def forward(self, input):
        self._osr_auxiliary_losses = {}
        if self.record_flag:
            self.record_x(input)
        effective_weight = self.osr_effective_weight()
        if self.recon_flag:
            if not self.recon_inited:
                self.init_scale(input)
            elif self.running_stat:
                self.update_scale(input)
            effective_scale = self.osr_effective_scale_factor()
            offset = self.offset.expand_as(input)
            input = (input - offset) / effective_scale
            recond_bias = self.fwd_func(
                offset, effective_weight, None, **self.fwd_kwargs
            )
            recond_bias = recond_bias.squeeze(1)
            # weight = nn.Parameter(self.weight / self.scale_factor)
            weight = effective_weight * effective_scale
        else:
            input = input
            weight = effective_weight
        input, weight = self._apply_hadamard(input, weight)
        input, weight = self._apply_osr_post_h_scale(input, weight)
        activation_reference = input
        weight_reference = weight
        if not (self._osr_mode == "precal_joint_wa_input_projector" and self.osr_smooth_residual_enabled):
            input = self._osr_quantize_activation(input)

        if self.use_weight_quant:
            if self.org_bias is not None:
                bias = (
                    self.bias_quantizer(self.bias)
                    if self.bias_quantizer is not None
                    else self.bias
                )
            else:
                bias = self.bias
        else:
            bias = self.org_bias
        operator_reference = self._osr_operator_reference(
            activation_reference, weight_reference, bias
        )
        out, weight_reference_for_qdq, quantized_weight = (
            self._osr_weight_and_residual_forward(
                input,
                weight,
                bias,
                branch_input=(
                    activation_reference
                    if self.osr_uses_post_h_fp_bypass
                    else None
                ),
            )
        )
        self._record_osr_auxiliary(
            activation_reference,
            input,
            weight_reference_for_qdq,
            quantized_weight,
            operator_reference,
            out,
        )
        if self.recon_flag:
            out = out + recond_bias
        out = self.activation_function(out)
        return out

    def record_x(self, x):
        x_clone = x.clone().detach().cpu()
        if self.input_embedd is None:
            self.input_embedd = x_clone
        else:
            self.input_embedd = torch.cat((self.input_embedd, x_clone), dim=0)

class saw_Linear_QuantLayer(HadamardQuantMixin, QuantLayer):
    """
    A quantized linear layer with scale and offset adaptation for weight and activation quantization.

    This layer wraps a standard nn.Linear module and applies quantization to its weights and activations,
    supporting adaptive scaling and offset for improved quantization accuracy. It provides mechanisms for
    recording input embeddings, updating scale factors based on running statistics, and reconstructing outputs
    with learned scale and offset parameters.

    Args:
        org_module (nn.Linear): The original linear module to be quantized.
        quantize_config (QuantizeModel_config): Configuration object specifying quantization parameters.
        device (str, optional): Device to place the quantized parameters on. Default is "cuda".
        s_alpha (float, optional): Exponent for scale factor computation. Default is 0.5.

    Attributes:
        quant_config: Stores the quantization configuration.
        input_embedd: Buffer for recorded input embeddings.
        record_flag (bool): Flag to enable input recording.
        device (str): Device for parameters.
        scale_factor (nn.Parameter): Learnable scale factor for input normalization.
        offset (nn.Parameter): Learnable offset for input normalization.
        recon_flag (bool): Flag to enable reconstruction mode.
        recon_inited (bool): Indicates if reconstruction parameters are initialized.
        s_alpha (float): Exponent for scale factor computation.
        running_stat (bool): Flag to enable running statistics for scale update.

    Methods:
        init_scale(input): Initializes the scale factor based on input and weight statistics.
        set_running_stat(running_stat): Sets the running statistics flag.
        update_scale(input, update_rate): Updates the scale factor using running statistics.
        set_record(flag): Enables or disables input recording.
        forward(input): Forward pass with quantization, scaling, and optional reconstruction.
        record_x(x): Records input embeddings for analysis or calibration.
    """
    def __init__(self,
                org_module: nn.Linear,
                quantize_config : QuantizeModel_config,
                device = "cuda",
                s_alpha = 0.5,
                ):
        super(saw_Linear_QuantLayer, self).__init__(org_module, quantize_config)
        self.quant_config = quantize_config
        self._init_hadamard_state(quantize_config, channel_dim=-1)
        self.input_embedd = None
        self.record_flag = False
        self.device = device
        scale_factor = torch.ones(1, self.org_weight.shape[1], requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.scale_factor = nn.Parameter(scale_factor, requires_grad=True)
        offset = torch.zeros(1, self.org_weight.shape[1], requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.offset = nn.Parameter(offset, requires_grad=True)
        self.recon_flag = False
        self.recon_inited = False
        if hasattr(quantize_config, 's_alpha'):
            self.s_alpha = quantize_config.s_alpha
        else:
            self.s_alpha = s_alpha
        self.running_stat = False
        self._init_osr_state()

    def init_scale(self, input):
        self.recon_inited = True
        self.scale_factor.data.copy_(self._calculate_let_scale(input))
        return self.scale_factor

    def set_running_stat(self, running_stat):
        self.running_stat = running_stat

    def update_scale(self, input, update_rate=0.1):
        tmp = self._calculate_let_scale(input)
        self.scale_factor.data = self.scale_factor.data * (1 - update_rate) + tmp * update_rate

    def set_record(self, flag):
        self.record_flag = flag

    def forward(self, input):
        self._osr_auxiliary_losses = {}
        if self.record_flag:
            self.record_x(input)
        effective_weight = self.osr_effective_weight()
        if self.recon_flag:
            if not self.recon_inited:
                self.init_scale(input)
            elif self.running_stat:
                self.update_scale(input)
            effective_scale = self.osr_effective_scale_factor()
            offset = self.offset.expand_as(input)
            input = (input - offset) / effective_scale
            recond_bias = effective_weight @ self.offset.T
            recond_bias = recond_bias.squeeze(1)
            weight = effective_weight * effective_scale
        else:
            input = input
            weight = effective_weight
        input, weight = self._apply_hadamard(input, weight)
        input, weight = self._apply_osr_post_h_scale(input, weight)
        activation_reference = input
        weight_reference = weight
        if not (self._osr_mode == "precal_joint_wa_input_projector" and self.osr_smooth_residual_enabled):
            input = self._osr_quantize_activation(input)

        if self.use_weight_quant:
            if self.org_bias is not None:
                bias = (
                    self.bias_quantizer(self.bias)
                    if self.bias_quantizer is not None
                    else self.bias
                )
            else:
                bias = self.bias
        else:
            bias = self.org_bias
        operator_reference = self._osr_operator_reference(
            activation_reference, weight_reference, bias
        )
        out, weight_reference_for_qdq, quantized_weight = (
            self._osr_weight_and_residual_forward(
                input,
                weight,
                bias,
                branch_input=(
                    activation_reference
                    if self.osr_uses_post_h_fp_bypass
                    else None
                ),
            )
        )
        self._record_osr_auxiliary(
            activation_reference,
            input,
            weight_reference_for_qdq,
            quantized_weight,
            operator_reference,
            out,
        )
        if self.recon_flag:
            out = out + recond_bias
        out = self.activation_function(out)
        return out

    def record_x(self, x):
        x_clone = x.clone().detach().cpu()
        if self.input_embedd is None:
            self.input_embedd = x_clone
        else:
            self.input_embedd = torch.cat((self.input_embedd, x_clone), dim=0)

class saw_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(saw_Loss, self).__init__()

    def forward(self, pred, target):
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        pred_std = torch.std(pred)
        target_std = torch.std(target)
        cov = torch.mean((pred - pred_mean) * (target - target_mean))
        correlation = cov / (pred_std * target_std + 1e-8)  # 避免除以0
        loss = 1 - correlation
        return loss
    
class PSNRLoss(nn.Module):
    def __init__(self, max_pixel_value=1.0, eps=1e-10):
        super(PSNRLoss, self).__init__()
        self.max_pixel_value = max_pixel_value
        self.eps = eps

    def forward(self, img1, img2):
        mse_loss = nn.functional.mse_loss(img1, img2)
        mse_loss = torch.clamp(mse_loss, min=self.eps)
        psnr = 10 * torch.log10(self.max_pixel_value ** 2 / mse_loss)
        return psnr

def get_loss_function(loss_name):
    loss_dict = {
        'mse': nn.MSELoss(),
        'cross_entropy': nn.CrossEntropyLoss(),
        'l1': nn.L1Loss(),
        'bce': nn.BCELoss(),
        'bce_with_logits': nn.BCEWithLogitsLoss(),
        'psnr': PSNRLoss()
    }
    
    if loss_name in loss_dict:
        return loss_dict[loss_name]
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
