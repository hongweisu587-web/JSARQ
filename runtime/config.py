"""Public inference configuration and project-relative path resolution."""
from pathlib import Path
import yaml
from runtime.presets import quantization_preset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path(value, root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_options(config_path, *, model_dir=None, device=None, root=None):
    root = PROJECT_ROOT if root is None else Path(root).resolve()
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The inference YAML must contain a mapping")
    quantization = config.get("quantization", {})
    if set(quantization) != {"bits", "checkpoint_dir", "jointwa_rank"}:
        raise ValueError("quantization accepts only bits, checkpoint_dir and jointwa_rank")
    bits = quantization["bits"]
    if type(bits) is not int or bits not in (4, 6):
        raise ValueError("quantization.bits must be 4 or 6")
    if quantization["jointwa_rank"] != 32:
        raise ValueError("Released checkpoints require jointwa_rank=32")
    upscale = config.get("upscale", 4)
    if upscale != 4:
        raise ValueError("These checkpoints support upscale=4")
    process_size = config.get("process_size", 128)
    if type(process_size) is not int or process_size < 4:
        raise ValueError("process_size must be an integer >= 4")
    align_method = config.get("align_method", "adain")
    if align_method not in ("adain", "wavelet", "nofix"):
        raise ValueError("align_method must be adain, wavelet or nofix")
    warmup = config.get("warmup", True)
    if type(warmup) is not bool:
        raise ValueError("warmup must be true or false")
    tiling = config["tiling"]
    for key in ("vae_encoder", "vae_decoder", "latent_size", "latent_overlap"):
        if type(tiling[key]) is not int or tiling[key] < 0:
            raise ValueError(f"tiling.{key} must be a nonnegative integer")
    if min(tiling[key] for key in ("vae_encoder", "vae_decoder", "latent_size")) == 0:
        raise ValueError("Tile sizes must be positive")
    if tiling["latent_overlap"] >= tiling["latent_size"]:
        raise ValueError("latent_overlap must be smaller than latent_size")
    models = config["models"]
    checkpoint_dir = _path(model_dir or quantization["checkpoint_dir"], root)
    return {
        "device": device or config.get("device", "cuda:0"),
        "warmup": warmup,
        "basic_config": {
            "seed": config.get("seed", 42), "upscale": upscale,
            "process_size": process_size, "align_method": align_method,
            "merge_lora": True,
            "config": str(_path(models["architecture"], root)),
            "ckpt": str(_path(models["checkpoint"], root)),
            "lora_weights_path": str(_path(models["lora"], root)),
            "context_embedding_path": str(_path(models["context"], root)),
            "pretrained_model_name_or_path": str(_path(models["scheduler"], root)),
        },
        "tile_config": {
            "vae_encoder_tiled_size": tiling["vae_encoder"],
            "vae_decoder_tiled_size": tiling["vae_decoder"],
            "latent_tiled_size": tiling["latent_size"],
            "latent_tiled_overlap": tiling["latent_overlap"],
        },
        "quantize_config": quantization_preset(bits, checkpoint_dir),
    }
