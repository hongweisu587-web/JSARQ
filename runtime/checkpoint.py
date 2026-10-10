"""Load the two published JointWA checkpoint presets, without PTQ code."""
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from quantization.saw.quant_config import QuantizeModel_config
from quantization.saw.quant_model import QuantWrapper
from quantization.saw.quant_layer import validate_quantizer_ranges
from quantization.saw.channelwise_asydobi import F3_PROJECTION_TARGETS
from runtime.artifacts import MANIFEST_NAMES, read_manifest
from runtime.topology import enumerate_candidates


def _restore_structure(wrapper, manifest):
    model = wrapper.model
    stage = manifest["stage"]
    records = enumerate_candidates(model, stage)
    expected = {record["module_name"] for record in records}
    items = manifest["modules"]
    counts = dict(Counter(record["layer_kind"] for record in records))
    if ({item["module_name"] for item in items} != expected
            or len(records) != len(items)
            or manifest.get("kind_counts", counts) != counts):
        raise ValueError(f"JointWA coverage mismatch for {stage}")
    if stage == "unet":
        required_counts = {"qkv": 96, "proj": 64, "fc1": 16, "fc2": 16,
                           "resblock_shortcut": 14}
        if counts != required_counts or not set(F3_PROJECTION_TARGETS).issubset(expected):
            raise ValueError("The released UNet requires the 206-layer JointWA topology")
    modules = dict(model.model.named_modules())
    for item in items:
        module = modules[item["module_name"]]
        if list(module.weight.shape) != item["shape"] or module.osr_enabled:
            raise ValueError(f"JointWA structure mismatch: {item['module_name']}")
    for item in items:
        modules[item["module_name"]].configure_precal_joint_wa_input_projector(rank=32)
    if stage == "unet":
        model.precal_preservation_manifest = manifest
    else:
        manifests = dict(getattr(model, "precal_preservation_manifests", {}))
        manifests[stage] = manifest
        model.precal_preservation_manifests = manifests


def _restore_parameter_layout(wrapper, state):
    """Restore channel range shapes and checkpoint dtypes before copying values."""
    modules = dict(wrapper.named_modules())
    with torch.no_grad():
        for name, parameter in list(wrapper.named_parameters()):
            saved = state.get(name)
            if saved is None:
                continue
            if parameter.shape != saved.shape or parameter.dtype != saved.dtype:
                parent, attribute = name.rsplit(".", 1)
                replacement = nn.Parameter(
                    torch.empty(saved.shape, device=parameter.device, dtype=saved.dtype),
                    requires_grad=parameter.requires_grad,
                )
                setattr(modules[parent], attribute, replacement)


def _load_component(model, config, quant_config, component, device):
    component_config = dict(config)
    component_config["hadamard_dobi"] = quant_config["hadamard_dobi"]
    wrapper = QuantWrapper(model, QuantizeModel_config(component_config), device=device)
    wrapper.set_quant_state(weight_quant=True, act_quant=True)
    checkpoint = Path(config["quant_ckpt"])
    stages = ("unet",) if component == "unet" else ("vae_encoder", "vae_decoder")
    for stage in stages:
        manifest = read_manifest(checkpoint.parent, stage, config["act_quant_bits"])
        _restore_structure(wrapper, manifest)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError(f"Expected a tensor state_dict checkpoint: {checkpoint}")
    factor_names = [name for name, _ in wrapper.named_parameters()
                    if name.endswith(("osr_lora_A", "osr_lora_B"))]
    for name in factor_names:
        if name not in state or state[name].dtype != torch.float32:
            raise ValueError(f"Missing or non-FP32 JointWA factor: {name}")
    _restore_parameter_layout(wrapper, state)
    result = wrapper.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"Checkpoint structure mismatch ({component}): "
            f"missing={result.missing_keys[:12]}, unexpected={result.unexpected_keys[:12]}"
        )
    validate_quantizer_ranges(wrapper, component=component)
    wrapper.set_running_stat(False)
    wrapper.set_record(False)
    wrapper.set_all_init(True, "all")
    wrapper.set_all_recon(True)
    wrapper.set_all_recon_init(True)
    wrapper.set_quant_state(weight_quant=True, act_quant=True)
    wrapper.report_hadamard_state(component)
    wrapper.requires_grad_(False)
    wrapper.eval()
    return wrapper


def load_Quantmodel(model, quant_config, device):
    """Compatibility signature used by the extracted inference pipeline."""
    unet, vae = model
    unet = _load_component(unet, quant_config["Unet"], quant_config, "unet", device)
    vae = _load_component(vae, quant_config["Vae"], quant_config, "vae", device)
    return unet, vae
