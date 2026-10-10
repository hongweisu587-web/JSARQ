"""Run extracted PassionSR inference on one image or an image directory."""
import argparse
from pathlib import Path
from runtime.artifacts import check_artifacts, required_files
from runtime.config import PROJECT_ROOT, load_options

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def collect_images(path):
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if path.is_dir():
        images = sorted(p for p in path.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            return images
    raise ValueError(f"No PNG/JPG/JPEG images found at {path}")


def prepare_image(image, basic_config):
    """Keep the original resize order, filters, and output-size convention."""
    original_width, original_height = image.size
    upscale = basic_config["upscale"]
    minimum = basic_config["process_size"] // upscale
    resized = original_width < minimum or original_height < minimum
    if resized:
        scale = minimum / min(original_width, original_height)
        image = image.resize((int(scale * original_width), int(scale * original_height)))
    image = image.resize((image.width * upscale, image.height * upscale))
    from PIL import Image
    image = image.resize((image.width - image.width % 8, image.height - image.height % 8),
                         Image.LANCZOS)
    restore_size = (upscale * original_width, upscale * original_height) if resized else None
    return image, restore_size


def run(options, images, output_dir, overwrite=False):
    output_dir = Path(output_dir).expanduser().resolve()
    if any(output_dir == image.parent for image in images):
        raise ValueError("Choose an output directory different from the input directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [image for image in images if overwrite or not (output_dir / image.name).exists()]
    if not pending:
        print("All output images already exist; use --overwrite to process them again.")
        return
    import torch
    from PIL import Image
    from torchvision import transforms
    from pytorch_lightning import seed_everything
    from tqdm import tqdm
    from runtime.pipeline import InferencePipeline
    from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

    device = options["device"]
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This reference implementation requires an available CUDA device")
    seed_everything(options["basic_config"]["seed"])
    pipeline = InferencePipeline(options).to(device).eval()
    for path in tqdm(pending, desc="Super-resolution", unit="image"):
        with Image.open(path) as source:
            image, restore_size = prepare_image(source.convert("RGB"), options["basic_config"])
        with torch.no_grad():
            tensor = transforms.ToTensor()(image).unsqueeze(0).to(device) * 2 - 1
            output = pipeline(tensor)
            result = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
            alignment = options["basic_config"]["align_method"]
            if alignment == "adain":
                result = adain_color_fix(target=result, source=image)
            elif alignment == "wavelet":
                result = wavelet_color_fix(target=result, source=image)
            if restore_size is not None:
                result = result.resize(restore_size)
        result.save(output_dir / path.name)
    print(f"Saved {len(pending)} image(s) to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/w4a4.yaml"))
    parser.add_argument("--input", help="Input image or directory")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--model-dir", help="Override the PTQ model directory")
    parser.add_argument("--device", help="Override the CUDA device, for example cuda:0")
    parser.add_argument("--vae-decoder-tile", type=int,
                        help="Override decoder tile size; use 96 for the FloodNet preset")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="Validate model files and manifests without importing PyTorch")
    args = parser.parse_args()
    try:
        options = load_options(args.config, model_dir=args.model_dir, device=args.device)
        if args.vae_decoder_tile is not None:
            if args.vae_decoder_tile <= 0:
                raise ValueError("--vae-decoder-tile must be positive")
            options["tile_config"]["vae_decoder_tiled_size"] = args.vae_decoder_tile
        if not args.check and (not args.input or not args.output):
            parser.error("--input and --output are required unless --check is used")
        check_artifacts(options)
        if args.check:
            print("Model files and JointWA manifest metadata are present and valid.")
            for path in required_files(options):
                print(path)
            return
        run(options, collect_images(args.input), args.output, args.overwrite)
    except (FileNotFoundError, ValueError, RuntimeError, ModuleNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
