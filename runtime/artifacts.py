"""Deployment file names and manifest validation; no GPU imports required."""
import json
from pathlib import Path

MANIFEST_NAMES = {
    "unet": "unet_precal_low_rank_preservation_manifest.json",
    "vae_encoder": "vae_encoder_precal_low_rank_preservation_manifest.json",
    "vae_decoder": "vae_decoder_precal_low_rank_preservation_manifest.json",
}
CHECKPOINT_NAMES = (
    "unet_ckpt_merge_saw_sep.pth", "vae_ckpt_merge_saw_sep.pth",
)


def read_manifest(directory, stage, bits):
    path = Path(directory) / MANIFEST_NAMES[stage]
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {
        "algorithm": "precal_joint_wa_input_hardness_v2",
        "branch_type": "precal_joint_wa_input_projector",
        "stage": stage, "rank": 32, "factor_dtype": "float32",
        "master_coordinate": "master_centered_input",
        "runtime_input": f"post_let_post_h_residual_pre_a{bits}",
        "statistics_sample_count": 480,
        "master_down_factor": "Vt", "master_up_factor": "WV",
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in required.items() if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Incompatible manifest {path.name}: {mismatches}")
    items = manifest.get("modules")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Manifest has no modules: {path}")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Malformed module entries in {path}")
    names = [item.get("module_name") for item in items]
    if (any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
            or manifest.get("candidate_count") != len(items)):
        raise ValueError(f"Invalid module count or duplicate names in {path}")
    return manifest


def required_files(options):
    base = options["basic_config"]
    checkpoint_dir = Path(options["quantize_config"]["Unet"]["quant_ckpt"]).parent
    return [
        Path(base[key]) for key in
        ("config", "ckpt", "lora_weights_path", "context_embedding_path")
    ] + [
        Path(base["pretrained_model_name_or_path"]) / "scheduler/scheduler_config.json",
    ] + [checkpoint_dir / name for name in (*CHECKPOINT_NAMES, *MANIFEST_NAMES.values())]


def check_artifacts(options):
    missing = [path for path in required_files(options) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model artifacts:\n" + "\n".join(str(p) for p in missing))
    config = options["quantize_config"]
    directory = Path(config["Unet"]["quant_ckpt"]).parent
    bits = config["Unet"]["act_quant_bits"]
    for stage in MANIFEST_NAMES:
        read_manifest(directory, stage, bits)
