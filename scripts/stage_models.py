"""Copy deployment artifacts from the original server project, without editing it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.artifacts import CHECKPOINT_NAMES, MANIFEST_NAMES, read_manifest

PREFIX = "blockwise_hadamard_asymmetric_dobi_hfirst_pipeline_joint_wa_best_v4_dense32_managed262k_stem_w8_f3_w"


def source_path(project, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def copy_plan(project, destination):
    pairs = []
    shared = None
    for bits in (4, 6):
        config_path = project / "scripts/inference/config" / f"{PREFIX}{bits}_jointwa" / "UV" / f"saw_U_W{bits}A{bits}_Stem_W8A8_F3_W{bits}A{bits}_V_W{bits}A{bits}.yaml"
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        quant = config["quantize_config"]
        unet_path = source_path(project, quant["Unet"]["quant_ckpt"])
        vae_path = source_path(project, quant["Vae"]["quant_ckpt"])
        if unet_path.parent != vae_path.parent:
            raise ValueError("Expected UNet and VAE artifacts in the same PTQ directory")
        model_output = destination / f"w{bits}a{bits}"
        pairs.extend([(unet_path, model_output / CHECKPOINT_NAMES[0]),
                      (vae_path, model_output / CHECKPOINT_NAMES[1])])
        for stage, name in MANIFEST_NAMES.items():
            # Verify that the right version's manifests are being exported.
            read_manifest(unet_path.parent, stage, bits)
            pairs.append((unet_path.parent / name, model_output / name))
        base = config["basic_config"]
        mapping = {
            "config": "v2-inference-v-fp32.yaml", "ckpt": "v2-1_512-ema-pruned.ckpt",
            "lora_weights_path": "osediff.ckpt", "context_embedding_path": "empty_context_embedding.pt",
        }
        common = [(source_path(project, base[key]), destination / "base" / filename)
                  for key, filename in mapping.items()]
        common.append((source_path(project, base["pretrained_model_name_or_path"]) / "scheduler/scheduler_config.json",
                       destination / "base/sd21/scheduler/scheduler_config.json"))
        if shared is not None and common != shared:
            raise ValueError("The two presets reference different shared base artifacts")
        shared = common
    return pairs + shared


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", required=True)
    parser.add_argument("--weights-dir", default=str(ROOT / "weights"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = Path(args.source_project).expanduser().resolve()
    destination = Path(args.weights_dir).expanduser().resolve()
    try:
        if destination == project or project in destination.parents:
            raise ValueError("Choose a weights directory outside the original source project")
        pairs = copy_plan(project, destination)
        missing = [src for src, _ in pairs if not src.is_file()]
        if missing:
            raise FileNotFoundError("Missing source artifacts:\n" + "\n".join(map(str, missing)))
        for src, dst in pairs:
            print(f"{src} -> {dst}")
        if args.dry_run:
            return
        existing = [dst for _, dst in pairs if dst.exists()]
        if existing and not args.overwrite:
            raise FileExistsError("Target artifacts already exist; use --overwrite if intended")
        records = []
        for src, dst in pairs:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            records.append({"file": str(dst.relative_to(destination)),
                            "bytes": dst.stat().st_size, "sha256": sha256(dst)})
        (destination / "artifacts.json").write_text(
            json.dumps({"files": records}, indent=2) + "\n", encoding="utf-8")
        print("Deployment files copied; original source files were not modified.")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
