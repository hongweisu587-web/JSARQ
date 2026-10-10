# JSARQ

Inference code and W4A4 / W6A6 calibrated models based on [PassionSR](https://github.com/libozhu03/PassionSR).

## Installation

Follow the PassionSR baseline environment: Python 3.10, PyTorch 2.0.1, and CUDA 11.8.

```bash
git clone https://github.com/hongweisu587-web/JSARQ.git
cd JSARQ
conda create -n jsarq python=3.10
conda activate jsarq
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements-inference.txt
```

## Pretrained Models

Download [JSARQ.zip](https://github.com/hongweisu587-web/JSARQ/releases/download/v0.1.0/JSARQ.zip) and extract its `weights/` directory into the project root. The package also contains the inference code and configurations.

| Model / file | Preparation |
|---|---|
| W4A4 calibrated UNet and VAE | Included in the release package under `weights/w4a4/`, together with the three JointWA manifests. |
| W6A6 calibrated UNet and VAE | Included in the release package under `weights/w6a6/`, together with the three JointWA manifests. |
| OSEDiff adapter and empty context embedding | Included in the release package as `weights/base/osediff.ckpt` and `weights/base/empty_context_embedding.pt`. |
| SD2.1 | Prepare the base model using the [PassionSR instructions](https://github.com/libozhu03/PassionSR#-download-pretrained-models-and-datasets) and [SD2.1 model repository](https://huggingface.co/stabilityai/stable-diffusion-2-1-base). |

Place the SD2.1 checkpoint, model configuration, and scheduler configuration in `weights/base/`:

```text
weights/base/
├── v2-1_512-ema-pruned.ckpt
├── v2-inference-v-fp32.yaml
├── osediff.ckpt
├── empty_context_embedding.pt
└── sd21/
    └── scheduler/
        └── scheduler_config.json
```

Model paths can be changed in `configs/w4a4.yaml` and `configs/w6a6.yaml`.

## Inference

W4A4:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/w4a4.yaml --input data/LR --output outputs/w4a4
```

W6A6:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/w6a6.yaml --input data/LR --output outputs/w6a6
```

`--input` accepts a single image or a directory of PNG/JPG/JPEG images. Add `--overwrite` to replace existing outputs.

For FloodNet with a decoder tile size of 96:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/w6a6.yaml --input data/FloodNet --output outputs/FloodNet --vae-decoder-tile 96
```
