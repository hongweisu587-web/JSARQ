import torch
import torch.nn as nn
from einops import rearrange, repeat
from types import MethodType
from .quant_config import Quantize_config, QuantizeModel_config
from .quant_layer import (
    BasicQuantLayer,
    QuantLayer,
    StraightThrough,
    get_layer_type,
    Basic2DquantLayer,
    ManagedTensorwise2DQuantLayer,
    PerHead2DquantLayer,
)
from .hadamard_utils import normalized_block_fwht
from .attention_wv_quant import (
    PerHeadGroupVQuantizer,
    SoftmaxLog4Quantizer,
    configure_attention_wv_runtime,
    validate_attention_wv_config,
)
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, checkpoint
from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul
from ldm.modules.attention import BasicTransformerBlock, CrossAttention
from ldm.modules.attention import exists, default


def _infer_attention_dim_head(attention):
    """Resolve ``dim_head`` before or after projection layers are quantized."""
    dim_head = getattr(attention, "dim_head", None)
    if dim_head is not None:
        return int(dim_head)

    heads = int(attention.heads)
    projection = attention.to_q
    out_features = getattr(projection, "out_features", None)
    if out_features is None:
        weight = getattr(projection, "weight", None)
        if weight is None or weight.ndim < 2:
            raise AttributeError(
                "Cannot infer attention dim_head: to_q exposes neither "
                "out_features nor a two-dimensional weight"
            )
        out_features = int(weight.shape[0])
    out_features = int(out_features)
    if heads <= 0 or out_features % heads != 0:
        raise ValueError(
            "Cannot infer attention dim_head from projection shape: "
            f"out_features={out_features}, heads={heads}"
        )
    return out_features // heads


def _infer_attention_query_dim(attention):
    query_dim = getattr(attention, "query_dim", None)
    if query_dim is not None:
        return int(query_dim)
    projection = attention.to_q
    in_features = getattr(projection, "in_features", None)
    if in_features is None:
        weight = getattr(projection, "weight", None)
        if weight is None or weight.ndim < 2:
            raise AttributeError(
                "Cannot infer attention query_dim: to_q exposes neither "
                "in_features nor a two-dimensional weight"
            )
        in_features = int(weight.shape[1])
    return int(in_features)


def _runtime_position_scale(scale, target_length, target_hw=None):
    """Adapt a calibrated position-wise attention scale to the runtime grid.

    VAE W/V equalization stores one paired scale per spatial position. The
    calibration and inference image sizes need not match, so the checkpoint
    vector cannot always be broadcast directly to the runtime attention map.
    Interpolating one shared scale and using it reciprocally for V and W keeps
    the reparameterization paired while preserving the calibrated spatial
    pattern in normalized image coordinates.
    """
    flat = scale.reshape(-1)
    target_length = int(target_length)
    if flat.numel() in (1, target_length):
        return flat
    if target_length <= 0:
        raise ValueError(f"attention target length must be positive, got {target_length}")

    source_side = int(round(flat.numel() ** 0.5))
    if target_hw is not None:
        target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
        if target_h * target_w != target_length:
            raise ValueError(
                f"attention target grid {target_h}x{target_w} does not match "
                f"target length {target_length}"
            )
    else:
        target_side = int(round(target_length ** 0.5))
        target_h = target_w = target_side

    original_dtype = flat.dtype
    if (
        source_side * source_side == flat.numel()
        and target_h * target_w == target_length
    ):
        resized = nn.functional.interpolate(
            flat.float().reshape(1, 1, source_side, source_side),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(-1)
    else:
        resized = nn.functional.interpolate(
            flat.float().reshape(1, 1, -1),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).reshape(-1)
    return resized.to(dtype=original_dtype).clamp_min(1.0e-6)
from ldm.modules.attention import MemoryEfficientCrossAttention
from ldm.modules.diffusionmodules.model import (
    ResnetBlock,
    AttnBlock,
    MemoryEfficientAttnBlock,
    nonlinearity,
)

class BaseQuantBlock(nn.Module):
    """
    BaseQuantBlock is a base class for quantization-aware neural network blocks.

    Args:
        quantize_config (QuantizeModel_config): Configuration object specifying quantization parameters.

    Attributes:
        use_weight_quant (bool): Flag indicating whether weight quantization is enabled.
        use_act_quant (bool): Flag indicating whether activation quantization is enabled.
        recon_flag (bool): Flag for reconstruction mode (default: False).
        quantize_config (QuantizeModel_config): Stores the quantization configuration.
        quant_layer_type (type): The quantization layer type determined by the configuration.
        activation_function (nn.Module): The activation function used, typically a straight-through estimator.

    Methods:
        set_quant_state(weight_quant: bool = False, act_quant: bool = False):
            Sets the quantization state for weights and activations for all QuantLayer modules in the block.

        set_running_stat(running_stat):
            Sets the running statistics for all QuantLayer modules in the block.
    """
    def __init__(self, quantize_config: QuantizeModel_config):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.recon_flag = False
        self.quantize_config = quantize_config
        self.quant_layer_type = get_layer_type(quantize_config)
        # self.act_quantizer = self.quant_layer_type(self.quantize_config.act_config)
        self.activation_function = StraightThrough()

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(weight_quant, act_quant)

    def set_running_stat(self, running_stat):
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_running_stat(running_stat)

def cross_attn_forward(self, x, context=None, mask=None):
    h = self.heads
    if getattr(self, "_attention_wv_cache_requested", False):
        self._attention_wv_cached_inputs = (
            x.detach(),
            None if context is None else context.detach(),
            None if mask is None else mask.detach(),
        )
    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)
    # print(q.shape)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    scale = 1.0 / q.shape[-1] ** 0.5
    event_qk_collector = getattr(self, "_event_qk_scale_collector", None)
    if event_qk_collector is not None:
        event_qk_collector.record(q, k)
    if self.use_act_quant:
        if self.recon_flag:
            if not self.scale_init_flag:
                self.init_scale_factor_qk(q, k, q.device)
            elif self.running_stat:
                self.unpdate_scale_factor_qk(q, k)
            q = torch.einsum("b i d, d -> b i d", q, self.scale_factor_qk)
            k = torch.einsum("b j d, d -> b j d", k, torch.reciprocal(self.scale_factor_qk))
        paired_collector = getattr(self, "_paired_qk_hadamard_collector", None)
        if paired_collector is not None:
            paired_collector.record(q, k)
        if getattr(self, "_paired_qk_hadamard_runtime_enabled", False):
            block_size = int(self._paired_qk_hadamard_runtime_block_size)
            if block_size != q.shape[-1] or block_size != k.shape[-1]:
                raise RuntimeError(
                    "Paired Q/K Hadamard requires a full dim_head transform: "
                    f"block={block_size}, q_dim={q.shape[-1]}, k_dim={k.shape[-1]}"
                )
            q = normalized_block_fwht(q, block_size, dim=-1)
            k = normalized_block_fwht(k, block_size, dim=-1)
        qk_collector = getattr(self, "_qk_logit_range_collector", None)
        if qk_collector is not None:
            qk_collector.record(q, k, v, mask)
        diagnostic_roles = getattr(
            self, "_diagnostic_disable_managed_roles", ()
        )
        quant_q = q if "qk" in diagnostic_roles else self.act_quantizer_q(q)
        quant_k = k if "qk" in diagnostic_roles else self.act_quantizer_k(k)
        # sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
        sim = torch.einsum('b i d, b j d -> b i j', quant_q, quant_k) * scale
    else:
        qk_collector = getattr(self, "_qk_logit_range_collector", None)
        if qk_collector is not None:
            qk_collector.record(q, k, v, mask)
        # sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)
    event_wv_collector = getattr(self, "_event_wv_scale_collector", None)
    if event_wv_collector is not None:
        event_wv_collector.record(v, attn)
    if self.use_act_quant:
        vw_scale = None
        if self.recon_flag:
            if not self.scale_init_flag:
                self.init_scale_factor_vw(v, attn, v.device)
                self.scale_init_flag = True
            elif self.running_stat:
                self.unpdate_scale_factor_vw(v, attn)
            vw_scale = _runtime_position_scale(
                self.scale_factor_vw, target_length=v.shape[1]
            )
            if getattr(self, "_attention_wv_enabled", False):
                # Log-W requires non-negative W*S. Keep the equivalent
                # reparameterization in its valid domain near zero.
                vw_scale = vw_scale.clamp_min(1.0e-6)
            attn = torch.einsum("b i j, j -> b i j", attn, vw_scale)
            v = torch.einsum(
                "b j d, j -> b j d", v, torch.reciprocal(vw_scale)
            )
        diagnostic_roles = getattr(
            self, "_diagnostic_disable_managed_roles", ()
        )
        quantized_w = (
            attn if "wv" in diagnostic_roles else self.act_quantizer_w(attn)
        )
        quantized_v = (
            v if "wv" in diagnostic_roles else self.act_quantizer_v(v)
        )
        out = torch.einsum(
            "b i j, b j d -> b i d", quantized_w, quantized_v
        )
        if isinstance(self.act_quantizer_v, PerHeadGroupVQuantizer):
            out = self.act_quantizer_v.inverse_context(out)
    else:
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)

class QuantBasicTransformerBlock(BaseQuantBlock):
    """
    A quantized version of a basic transformer block, enabling activation and weight quantization
    for attention and feed-forward layers.
    Args:
        tran (BasicTransformerBlock): The original transformer block to be quantized.
        quantize_config (QuantizeModel_config): Configuration for quantization, including
            activation and weight quantization settings.
    Attributes:
        attn1: The first attention layer, with quantization hooks.
        attn2: The second attention layer (cross-attention), with quantization hooks.
        ff: The feed-forward layer.
        norm1, norm2, norm3: Normalization layers.
        checkpoint (bool): Whether to use gradient checkpointing for memory efficiency.
        quant_layer_type: The quantization layer class/type, determined by config.
        use_weight_quant (bool): Whether to enable weight quantization.
        use_act_quant (bool): Whether to enable activation quantization.
    Methods:
        forward(x, context=None):
            Forward pass with optional context, using checkpointing if enabled.
        _forward(x, context=None):
            Internal forward pass: applies attention, cross-attention, and feed-forward layers
            with residual connections.
        set_quant_state(weight_quant=False, act_quant=False):
            Sets the quantization state for weights and activations for all quantized layers
            within the block.
    """
    def __init__(
        self, tran: BasicTransformerBlock,
        quantize_config: QuantizeModel_config,
        ):
        super().__init__(quantize_config)
        self.attn1 = tran.attn1
        self.ff = tran.ff
        self.attn2 = tran.attn2
        for attention in (self.attn1, self.attn2):
            if not hasattr(attention, "dim_head"):
                attention.dim_head = _infer_attention_dim_head(attention)

        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint = tran.checkpoint
        # self.checkpoint = False
        self.quant_layer_type = get_layer_type(quantize_config)
        # logger.info(f"quant attn matmul")
        self.attn1.act_quantizer_q = self.quant_layer_type(self.quantize_config.act_config)
        self.attn1.act_quantizer_k = self.quant_layer_type(self.quantize_config.act_config)
        self.attn1.act_quantizer_v = self.quant_layer_type(self.quantize_config.act_config)
        attention_wv_config = getattr(
            self.quantize_config, "attention_wv_quantization", {}
        )
        validate_attention_wv_config(
            attention_wv_config, self.quantize_config.quantize_config
        )
        self.attention_wv_available = bool(
            attention_wv_config.get("enabled", False)
        )

        blockwise_config = getattr(quantize_config, "hadamard_dobi", {}).get(
            "blockwise", {}
        )
        managed_tensorwise_config = blockwise_config.get(
            "managed_tensorwise_asydobi", {}
        )
        post_h_config = getattr(quantize_config, "hadamard_dobi", {})
        hybrid_reference_attention = (
            post_h_config.get("post_h_hybrid_initializer", {}).get("version")
            == "posth_hybrid_global_channelwise_reference_attention_v1"
            or post_h_config.get("post_h_reference_initializer", {}).get("version")
            in {
                "posth_reference_global_channelwise_native_attention_v1",
                "posth_reference_tensorwise_native_v2",
            }
            or post_h_config.get("post_h_protected_initializer", {}).get("version")
            in {
                "posth_protected_historical_channelwise_reference_attention_v1",
                "posth_runtime_trace_operator_grouped_lazy_cache_v10",
                "posth_channelwise_best_v4_dense32",
                "posth_channelwise_best_v4_dense32_managed262k",
                "posth_channelwise_best_v4_dense32_asydobi_then_joint_wa",
            }
        )
        managed_tensorwise_enabled = bool(
            managed_tensorwise_config.get("enabled", False)
        )
        if managed_tensorwise_enabled and not all(
            bool(managed_tensorwise_config.get(key, False))
            for key in ("paired_qk_managed", "attention_wv_managed")
        ):
            raise ValueError(
                "hybrid DOBI requires managed tensorwise Q/K and W/V together"
            )
        self.paired_qk_tensorwise_asydobi = bool(
            managed_tensorwise_enabled
            and managed_tensorwise_config.get("paired_qk_managed", False)
        )
        self.attention_wv_tensorwise_asydobi = bool(
            managed_tensorwise_enabled
            and managed_tensorwise_config.get("attention_wv_managed", False)
        )
        if managed_tensorwise_enabled:
            if blockwise_config.get("version") != "posth_channelwise_v2":
                raise ValueError(
                    "managed tensorwise attention AsyDoBi requires "
                    "blockwise.version=posth_channelwise_v2"
                )
            if self.quant_layer_type is not Basic2DquantLayer:
                raise TypeError(
                    "managed tensorwise attention AsyDoBi requires "
                    "layer_type=2Dquant"
                )
            if int(self.quantize_config.act_config.quant_bits) not in (4, 6):
                raise ValueError(
                    "managed tensorwise attention AsyDoBi supports A4 or A6"
                )

        paired_qk_config = getattr(quantize_config, "hadamard_dobi", {}).get(
            "paired_qk_hadamard", {}
        )
        self.paired_qk_hadamard_available = bool(
            paired_qk_config.get("enabled", False)
        )
        if self.paired_qk_tensorwise_asydobi and self.paired_qk_hadamard_available:
            raise ValueError(
                "paired_qk_managed is mutually exclusive with paired Q/K Hadamard"
            )
        if self.attention_wv_tensorwise_asydobi and self.attention_wv_available:
            raise ValueError(
                "attention_wv_managed is mutually exclusive with HCC-WV quantization"
            )
        if self.paired_qk_hadamard_available:
            if self.quant_layer_type is not Basic2DquantLayer:
                raise TypeError(
                    "Paired Q/K per-head ranges require layer_type=2Dquant"
                )
            self.attn2.act_quantizer_q = PerHead2DquantLayer(
                self.quantize_config.act_config, self.attn2.heads
            )
            self.attn2.act_quantizer_k = PerHead2DquantLayer(
                self.quantize_config.act_config, self.attn2.heads
            )
        elif self.paired_qk_tensorwise_asydobi:
            self.attn2.act_quantizer_q = ManagedTensorwise2DQuantLayer(
                self.quantize_config.act_config,
                family="paired_qk",
                role="q",
            )
            self.attn2.act_quantizer_k = ManagedTensorwise2DQuantLayer(
                self.quantize_config.act_config,
                family="paired_qk",
                role="k",
            )
        else:
            self.attn2.act_quantizer_q = self.quant_layer_type(
                self.quantize_config.act_config
            )
            self.attn2.act_quantizer_k = self.quant_layer_type(
                self.quantize_config.act_config
            )
        if self.attention_wv_available:
            self.attn2.act_quantizer_v = PerHeadGroupVQuantizer(
                self.quantize_config.act_config,
                self.attn2.heads,
                self.attn2.dim_head,
                attention_wv_config.get("value_v", {}),
            )
        elif self.attention_wv_tensorwise_asydobi:
            self.attn2.act_quantizer_v = ManagedTensorwise2DQuantLayer(
                self.quantize_config.act_config,
                family="attention_wv",
                role="v",
            )
        else:
            self.attn2.act_quantizer_v = self.quant_layer_type(
                self.quantize_config.act_config
            )

        if self.attention_wv_available:
            self.attn1.act_quantizer_w = SoftmaxLog4Quantizer(
                self.quantize_config.act_config,
                self.attn1.heads,
                attention_wv_config.get("softmax_w", {}),
            )
        elif self.attention_wv_tensorwise_asydobi:
            self.attn1.act_quantizer_w = ManagedTensorwise2DQuantLayer(
                self.quantize_config.act_config,
                family="attention_wv",
                role="w",
            )
        else:
            self.attn1.act_quantizer_w = self.quant_layer_type(
                self.quantize_config.act_config
            )
        self.attn2.act_quantizer_w = self.quant_layer_type(self.quantize_config.act_config)

        # Reference-aligned hybrid DOBI owns every internal Q/K/V/W site.
        # Keep projection-layer input quantizers ordinary, but make both
        # self- and cross-attention internal operands scalar-range managed.
        if hybrid_reference_attention and self.paired_qk_tensorwise_asydobi:
            for attention in (self.attn1, self.attn2):
                attention.act_quantizer_q = ManagedTensorwise2DQuantLayer(
                    self.quantize_config.act_config, family="paired_qk", role="q"
                )
                attention.act_quantizer_k = ManagedTensorwise2DQuantLayer(
                    self.quantize_config.act_config, family="paired_qk", role="k"
                )
        if hybrid_reference_attention and self.attention_wv_tensorwise_asydobi:
            for attention in (self.attn1, self.attn2):
                attention.act_quantizer_w = ManagedTensorwise2DQuantLayer(
                    self.quantize_config.act_config, family="attention_wv", role="w"
                )
                attention.act_quantizer_v = ManagedTensorwise2DQuantLayer(
                    self.quantize_config.act_config, family="attention_wv", role="v"
                )

        # Cross-attention Q/K/V are [batch*heads, tokens, dim_head], so their
        # natural feature granularity is the final dim_head axis.  Attention
        # probabilities remain per-tensor to avoid one range per token/key.
        for attention in (self.attn1, self.attn2):
            for suffix in ("q", "k", "v"):
                quantizer = getattr(attention, f"act_quantizer_{suffix}")
                if not getattr(quantizer, "paired_qk_managed", False) and not getattr(
                    quantizer, "attention_wv_managed", False
                ):
                    quantizer.channelwise_axis = -1
                    quantizer.channelwise_role = f"attention_{suffix}"
            if not getattr(
                attention.act_quantizer_w, "managed_tensorwise_asydobi", False
            ):
                attention.act_quantizer_w.channelwise_axis = None
                attention.act_quantizer_w.channelwise_role = (
                    "attention_w_per_tensor"
                )

        # CrossAttention is retained from the original model, so explicitly
        # attach the paired-scale implementation owned by QuantAttnBlock.
        # The finite-window initializer also reads these two metadata fields
        # to use exactly the same UNet reduction dimensions and exponent.
        for attention in (self.attn1, self.attn2):
            attention.is_vae_attention = False
            attention.s_alpha = (
                self.quantize_config.s_alpha
                if hasattr(self.quantize_config, "s_alpha")
                else 0.5
            )
            attention.init_scale_factor_qk = MethodType(
                QuantAttnBlock.init_scale_factor_qk, attention
            )
            attention.unpdate_scale_factor_qk = MethodType(
                QuantAttnBlock.unpdate_scale_factor_qk, attention
            )
            attention.init_scale_factor_vw = MethodType(
                QuantAttnBlock.init_scale_factor_vw, attention
            )
            attention.unpdate_scale_factor_vw = MethodType(
                QuantAttnBlock.unpdate_scale_factor_vw, attention
            )

        self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
        self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
        configure_attention_wv_runtime(
            self.attn1, "attn1", attention_wv_config
        )
        configure_attention_wv_runtime(
            self.attn2, "attn2", attention_wv_config
        )
        self.attn1.use_act_quant = False
        self.attn2.use_act_quant = False
        self.attn1.recon_flag = False
        self.attn2.recon_flag = False
        self.attn1.running_stat = False
        self.attn2.running_stat = False

        self.attn1.scale_factor_qk = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.attn1.scale_factor_vw = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.attn1.scale_init_flag = False

        self.attn2.scale_factor_qk = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.attn2.scale_factor_vw = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.attn2.scale_init_flag = False
        self.attn2.register_buffer(
            "paired_qk_hadamard_enabled",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.attn2.register_buffer(
            "paired_qk_hadamard_block_size",
            torch.tensor(0, dtype=torch.int32),
            persistent=True,
        )
        self.attn2.register_buffer(
            "paired_qk_hadamard_version",
            torch.tensor(0, dtype=torch.int32),
            persistent=True,
        )
        self.attn2.register_buffer(
            "paired_qk_activation_bits",
            torch.tensor(
                int(self.attn2.act_quantizer_q.quant_bits), dtype=torch.int32
            ),
            persistent=True,
        )
        self.attn2._paired_qk_hadamard_runtime_enabled = False
        self.attn2._paired_qk_hadamard_runtime_block_size = 0

    def configure_paired_qk_hadamard(self, enabled, block_size, version=1):
        enabled = bool(enabled)
        block_size = int(block_size) if enabled else 0
        version = int(version)
        self.attn2.paired_qk_hadamard_enabled.fill_(enabled)
        self.attn2.paired_qk_hadamard_block_size.fill_(block_size)
        self.attn2.paired_qk_hadamard_version.fill_(version)
        self.attn2._paired_qk_hadamard_runtime_enabled = enabled
        self.attn2._paired_qk_hadamard_runtime_block_size = block_size

    def sync_paired_qk_hadamard_runtime(self):
        enabled = bool(
            self.attn2.paired_qk_hadamard_enabled.detach().cpu().item()
        )
        block_size = int(
            self.attn2.paired_qk_hadamard_block_size.detach().cpu().item()
        )
        self.attn2._paired_qk_hadamard_runtime_enabled = enabled
        self.attn2._paired_qk_hadamard_runtime_block_size = block_size

    def forward(self, x, context=None):
        # print(f"x shape {x.shape} context shape {context.shape}")
        checkpoint_parameters = self.parameters()
        if self.attention_wv_available:
            # HCC-WV adds persistent topology buffers and dormant compatibility
            # parameters. The legacy checkpoint autograd helper cannot receive
            # tensors with requires_grad=False in its differentiated list.
            checkpoint_parameters = (
                parameter
                for parameter in checkpoint_parameters
                if parameter.requires_grad
            )
        return checkpoint(
            self._forward,
            (x, context),
            checkpoint_parameters,
            self.checkpoint,
        )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn1.use_act_quant = act_quant
        self.attn2.use_act_quant = act_quant

        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(weight_quant, act_quant)

class QuantAttnBlock(BaseQuantBlock):
    """
    Quantized Attention Block for neural network quantization.

    This class wraps an attention block with quantization-aware operations, enabling quantization of activations
    and scaling factors for efficient inference. It supports initialization and updating of scale factors for
    query-key and value-attention computations, and applies quantization to the attention mechanism.

    Args:
        attn (AttnBlock): The original attention block to be quantized.
        quantize_config (QuantizeModel_config): Configuration for quantization, including activation quantizer settings.
        s_alpha (float, optional): Exponent for scale factor computation. Defaults to 0.5.

    Attributes:
        in_channels (int): Number of input channels.
        heads (int): Number of attention heads.
        norm (nn.Module): Normalization layer (uses StraightThrough for quantization).
        to_q, to_k, to_v, to_out (nn.Module): Linear layers for query, key, value, and output projections.
        quant_layer_type (type): Quantization layer type from configuration.
        act_quantizer_q, act_quantizer_k, act_quantizer_v, act_quantizer_w (nn.Module): Activation quantizers for Q, K, V, and attention weights.
        scale_factor_qk (nn.Parameter): Learnable scale factor for query-key.
        scale_factor_vw (nn.Parameter): Learnable scale factor for value-attention.
        scale_init_flag (bool): Flag indicating if scale factors are initialized.
        s_alpha (float): Exponent for scale factor computation.
        running_stat (bool): Whether to use running statistics for quantization.

    Methods:
        set_running_stat(running_stat): Set the running statistics flag.
        init_scale_factor_qk(q, k, device): Initialize scale factor for query-key.
        unpdate_scale_factor_qk(q, k, update_rate=0.1): Update scale factor for query-key.
        init_scale_factor_vw(v, attn, device): Initialize scale factor for value-attention.
        unpdate_scale_factor_vw(v, attn, update_rate=0.1): Update scale factor for value-attention.
        forward(x): Forward pass with quantized attention computation.

    Forward Pass:
        - Applies normalization and projects input to Q, K, V.
        - Optionally applies scaling and quantization to Q, K, V, and attention weights.
        - Computes attention weights and applies them to values.
        - Projects output and adds residual connection.
    """
    def __init__(
        self,
        attn: AttnBlock,
        quantize_config:QuantizeModel_config,
        s_alpha = 0.5
        ):
        super().__init__(quantize_config)
        # LDM transparently replaces vanilla VAE attention with the xFormers
        # implementation when xFormers is installed.  Both classes expose the
        # same norm/q/k/v/proj_out topology and use the same single-head
        # attention math, so they must enter the same quantized wrapper.
        self.is_vae_attention = isinstance(
            attn, (AttnBlock, MemoryEfficientAttnBlock)
        )
        if self.is_vae_attention:
            self.in_channels = int(attn.in_channels)
            self.heads = 1
            self.dim_head = self.in_channels
            self.norm = attn.norm
            self.to_q = attn.q
            self.to_k = attn.k
            self.to_v = attn.v
            self.to_out = attn.proj_out
        else:
            self.in_channels = _infer_attention_query_dim(attn)
            self.heads = attn.heads
            self.dim_head = _infer_attention_dim_head(attn)
            self.norm = StraightThrough()
            self.to_q = attn.to_q
            self.to_k = attn.to_k
            self.to_v = attn.to_v
            self.to_out = attn.to_out

        self.quant_layer_type = get_layer_type(quantize_config)
        managed = getattr(quantize_config, "hadamard_dobi", {}).get(
            "blockwise", {}
        ).get("managed_tensorwise_asydobi", {})
        managed_enabled = bool(managed.get("enabled", False))
        if managed_enabled:
            if not all(
                bool(managed.get(key, False))
                for key in ("paired_qk_managed", "attention_wv_managed")
            ):
                raise ValueError(
                    "hybrid DOBI requires managed tensorwise Q/K and W/V together"
                )
            self.act_quantizer_q = ManagedTensorwise2DQuantLayer(
                self.quantize_config.act_config, family="paired_qk", role="q"
            )
            self.act_quantizer_k = ManagedTensorwise2DQuantLayer(
                self.quantize_config.act_config, family="paired_qk", role="k"
            )
            self.act_quantizer_v = ManagedTensorwise2DQuantLayer(
                self.quantize_config.act_config, family="attention_wv", role="v"
            )
            self.act_quantizer_w = ManagedTensorwise2DQuantLayer(
                self.quantize_config.act_config, family="attention_wv", role="w"
            )
        else:
            self.act_quantizer_q = self.quant_layer_type(self.quantize_config.act_config)
            self.act_quantizer_k = self.quant_layer_type(self.quantize_config.act_config)
            self.act_quantizer_v = self.quant_layer_type(self.quantize_config.act_config)
            self.act_quantizer_w = self.quant_layer_type(self.quantize_config.act_config)
            # VAE attention uses q=[B,HW,C], k/v=[B,C,HW].
            self.act_quantizer_q.channelwise_axis = -1
            self.act_quantizer_q.channelwise_role = "attention_q"
            for suffix in ("k", "v"):
                quantizer = getattr(self, f"act_quantizer_{suffix}")
                quantizer.channelwise_axis = 1
                quantizer.channelwise_role = f"attention_{suffix}"
            self.act_quantizer_w.channelwise_axis = None
            self.act_quantizer_w.channelwise_role = "attention_w_per_tensor"
        self.scale_factor_qk = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.scale_factor_vw = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.scale_init_flag = False
        if hasattr(quantize_config,'s_alpha'):
            self.s_alpha = quantize_config.s_alpha
        else:
            self.s_alpha = s_alpha
        self.running_stat = False

    def set_running_stat(self, running_stat):
        self.running_stat = running_stat

    def init_scale_factor_qk(self, q, k, device):
        self.scale_factor_qk.data = torch.ones(q.shape[2], requires_grad=True, \
                                    dtype=torch.float32, device=device)
        q_max = torch.amax(torch.abs(q), dim=(0,1)).clamp(min=1e-5).squeeze()
        k_dims = (0, 2) if self.is_vae_attention else (0, 1)
        k_max = torch.amax(torch.abs(k), dim=k_dims).clamp(min=1e-5).squeeze()
        tmp = k_max ** self.s_alpha / q_max ** self.s_alpha
        self.scale_factor_qk.data = tmp

    def unpdate_scale_factor_qk(self, q, k, update_rate=0.1):
        q_max = torch.amax(torch.abs(q), dim=(0,1)).clamp(min=1e-5).squeeze()
        k_dims = (0, 2) if self.is_vae_attention else (0, 1)
        k_max = torch.amax(torch.abs(k), dim=k_dims).clamp(min=1e-5).squeeze()
        tmp = k_max ** self.s_alpha / q_max ** self.s_alpha
        self.scale_factor_qk.data =  self.scale_factor_qk.data * (1 - update_rate) + tmp * update_rate

    def init_scale_factor_vw(self, v, attn, device):
        scale_length = v.shape[2] if self.is_vae_attention else v.shape[1]
        self.scale_factor_vw.data = torch.ones(scale_length, requires_grad=True, \
                                    dtype=torch.float32, device=device)
        v_dims = (0, 1) if self.is_vae_attention else (0, 2)
        attn_dims = (0, 2) if self.is_vae_attention else (0, 1)
        v_max = torch.amax(torch.abs(v), dim=v_dims).clamp(min=1e-5).squeeze()
        attn_max = torch.amax(torch.abs(attn), dim=attn_dims).clamp(min=1e-5).squeeze()
        self.scale_factor_vw.data =  v_max ** self.s_alpha / attn_max ** self.s_alpha
        # self.scale_init_flag = True

    def unpdate_scale_factor_vw(self, v, attn, update_rate=0.1):
        v_dims = (0, 1) if self.is_vae_attention else (0, 2)
        attn_dims = (0, 2) if self.is_vae_attention else (0, 1)
        v_max = torch.amax(torch.abs(v), dim=v_dims).clamp(min=1e-5).squeeze()
        attn_max = torch.amax(torch.abs(attn), dim=attn_dims).clamp(min=1e-5).squeeze()
        tmp = v_max ** self.s_alpha / attn_max ** self.s_alpha
        self.scale_factor_vw.data =  self.scale_factor_vw.data * (1 - update_rate) + tmp * update_rate

    def forward_attention(self, h_):
        """Run the attention core on an already-normalized VAE feature map.

        Tiled VAE inference has to execute GroupNorm across all tiles before
        it can run the remaining attention operations.  Exposing the core
        separately lets that path preserve the exact quantized Q/K/W/V math
        without reaching through this wrapper for the original ``q``/``k``/
        ``v`` attributes.
        """
        q = self.to_q(h_)
        k = self.to_k(h_)
        v = self.to_v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        event_qk_collector = getattr(self, "_event_qk_scale_collector", None)
        if event_qk_collector is not None:
            event_qk_collector.record(q, k)
        if self.recon_flag:
            if self.use_act_quant:
                if not self.scale_init_flag:
                    self.init_scale_factor_qk(q, k, q.device)
                elif self.running_stat:
                    self.unpdate_scale_factor_qk(q, k)
            q = torch.einsum("b h c, c -> b h c", q, self.scale_factor_qk)
            k = torch.einsum(
                "b c h, c -> b c h", k, torch.reciprocal(self.scale_factor_qk)
            )
            # q = q * self.scale_factor
            # k = k / self.scale_factor
        if self.use_act_quant and "qk" not in getattr(
            self, "_diagnostic_disable_managed_roles", ()
        ):
            q = self.act_quantizer_q(q)
            k = self.act_quantizer_k(k)
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        event_wv_collector = getattr(self, "_event_wv_scale_collector", None)
        if event_wv_collector is not None:
            event_wv_collector.record(v, w_)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        if self.recon_flag:
            if self.use_act_quant:
                if not self.scale_init_flag:
                    self.init_scale_factor_vw(v, w_, v.device)
                    self.scale_init_flag = True
                elif self.running_stat:
                    self.unpdate_scale_factor_vw(v, w_)
            vw_scale = _runtime_position_scale(
                self.scale_factor_vw,
                target_length=h * w,
                target_hw=(h, w),
            )
            v = torch.einsum(
                "b c h, h -> b c h", v, torch.reciprocal(vw_scale)
            )
            w_ = torch.einsum("b m n, m -> b m n", w_, vw_scale)
        if self.use_act_quant and "wv" not in getattr(
            self, "_diagnostic_disable_managed_roles", ()
        ):
            v = self.act_quantizer_v(v)
            w_ = self.act_quantizer_w(w_)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.to_out(h_)

        return h_

    def forward(self, x):
        h_ = self.forward_attention(self.norm(x))

        out = x + h_
        return out


def get_specials(quant_act=False, include_vae_attention=False):
    specials = {
        BasicTransformerBlock: QuantBasicTransformerBlock,
        CrossAttention: QuantAttnBlock,
        MemoryEfficientCrossAttention: QuantAttnBlock,
    }
    if include_vae_attention:
        specials[AttnBlock] = QuantAttnBlock
        specials[MemoryEfficientAttnBlock] = QuantAttnBlock
    return specials
