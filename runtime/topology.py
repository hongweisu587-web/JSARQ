from typing import Optional
from quantization.saw.saw_layer import saw_Conv2d_QuantLayer, saw_Linear_QuantLayer

def _normalize_candidate_module_name(module_name: str) -> str:
    normalized = str(module_name)
    for suffix in (".base_layer", ".quant_layer"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized

def _match_any(module_name: str, tokens: tuple[str, ...]) -> bool:
    normalized_name = _normalize_candidate_module_name(module_name)
    return any(normalized_name.endswith(token) for token in tokens)

def _is_conv1x1(module) -> bool:
    weight = getattr(module, "org_weight", getattr(module, "weight", None))
    return (
        isinstance(module, saw_Conv2d_QuantLayer)
        and weight is not None
        and tuple(weight.shape[2:]) == (1, 1)
    )

def _is_vae_mid(module_name: str) -> bool:
    normalized_name = _normalize_candidate_module_name(module_name)
    return (
        "encoder.mid_block" in normalized_name
        or "decoder.mid_block" in normalized_name
        or "encoder.mid." in normalized_name
        or "decoder.mid." in normalized_name
    )

def _classify_candidate(module_name: str, module, model_role: str):
    if isinstance(module, saw_Linear_QuantLayer):
        if model_role == "unet":
            if _match_any(module_name, (".to_q", ".to_k", ".to_v")):
                return "qkv", "linear"
            if _match_any(module_name, (".to_out.0", ".proj_in", ".proj_out")):
                return "proj", "linear"
            if _match_any(module_name, (".ff.net.0.proj", ".ff.net.0.0", ".mlp.fc1")):
                return "fc1", "linear"
            if _match_any(module_name, (".ff.net.2", ".mlp.fc2")):
                return "fc2", "linear"
        elif model_role == "vae" and _is_vae_mid(module_name):
            if _match_any(module_name, (".to_q", ".to_k", ".to_v")):
                return "qkv", "linear"
            if _match_any(module_name, (".to_out.0",)):
                return "proj", "linear"
        return None

    if not _is_conv1x1(module):
        return None
    if model_role == "unet":
        if _match_any(module_name, (".proj_in", ".proj_out")):
            return "proj", "conv2d_1x1"
        if _match_any(module_name, (".skip_connection", ".conv_shortcut", ".nin_shortcut")):
            return "resblock_shortcut", "conv2d_1x1"
        return None
    if model_role == "vae":
        if _match_any(module_name, (".nin_shortcut",)):
            return "resblock_shortcut", "conv2d_1x1"
        if _is_vae_mid(module_name):
            if _match_any(module_name, (".q", ".k", ".v")):
                return "qkv", "conv2d_1x1"
            if _match_any(module_name, (".proj_out",)):
                return "proj", "conv2d_1x1"
    return None

def _matches_stage_scope(module_name: str, stage_scope: Optional[str]) -> bool:
    if stage_scope in (None, "unet"):
        return True
    normalized = _normalize_candidate_module_name(module_name)
    if stage_scope == "encoder":
        return "encoder." in normalized or normalized.startswith("encoder.")
    if stage_scope == "decoder":
        return "decoder." in normalized or normalized.startswith("decoder.")
    raise ValueError(f"Unsupported low_rank_absorb stage scope {stage_scope!r}")

def _is_fixed_precision_candidate(module) -> bool:
    if bool(getattr(module, "fixed_precision", False)):
        return True
    weight_quantizer = getattr(module, "weight_quantizer_0", None)
    act_quantizer = getattr(module, "act_quantizer_0", None)
    if weight_quantizer is None or act_quantizer is None:
        return False
    return bool(
        int(getattr(weight_quantizer, "quant_bits", 0)) >= 8
        and int(getattr(act_quantizer, "quant_bits", 0)) >= 8
    )


def enumerate_candidates(model, stage):
    role = "unet" if stage == "unet" else "vae"
    scope = {"unet": None, "vae_encoder": "encoder", "vae_decoder": "decoder"}[stage]
    records = []
    for name, module in model.model.named_modules():
        if not _matches_stage_scope(name, scope) or _is_fixed_precision_candidate(module):
            continue
        classified = _classify_candidate(name, module, role)
        if classified is not None:
            records.append({"module_name": name, "layer_kind": classified[0]})
    return records
