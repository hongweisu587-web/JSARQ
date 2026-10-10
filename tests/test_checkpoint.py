"""Real tensor round-trip checks on a small VAE-shaped model; no SD download."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tests.test_release import manifest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointTests(unittest.TestCase):
    def test_baseline_module_apply_signature(self):
        from runtime.presets import quantization_preset
        from quantization.saw.quant_config import QuantizeModel_config
        from quantization.saw.quant_model import QuantWrapper
        original_apply = nn.Module._apply

        def baseline_apply(module, fn):
            # PyTorch 2.0.1 accepts fn only, without the newer recurse keyword.
            return original_apply(module, fn)

        config = quantization_preset(4, Path("unused"))
        component = dict(config["Vae"])
        component["hadamard_dobi"] = config["hadamard_dobi"]
        with patch.object(nn.Module, "_apply", baseline_apply):
            model = QuantWrapper(self._toy_vae(), QuantizeModel_config(component), "cpu")
            layer = model.model.model.encoder.block.nin_shortcut
            layer.configure_precal_joint_wa_input_projector(rank=32)
            model.to(dtype=torch.float64)
            self.assertEqual(layer.osr_lora_A.dtype, torch.float32)
            self.assertEqual(layer.osr_lora_B.dtype, torch.float32)

    def _toy_vae(self):
        class ToyVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.block = nn.Module()
                self.encoder.block.nin_shortcut = nn.Conv2d(64, 48, 1)
                self.decoder = nn.Module()
                self.decoder.block = nn.Module()
                self.decoder.block.nin_shortcut = nn.Conv2d(48, 64, 1)

            def forward(self, x, timesteps=None, context=None):
                return self.decoder.block.nin_shortcut(self.encoder.block.nin_shortcut(x))
        return ToyVAE()

    def test_factors_ranges_and_hadamard_round_trip_for_both_presets(self):
        from runtime.presets import quantization_preset
        from runtime.checkpoint import _load_component
        from runtime.artifacts import MANIFEST_NAMES
        from runtime.topology import enumerate_candidates
        from quantization.saw.quant_config import QuantizeModel_config
        from quantization.saw.quant_model import QuantWrapper
        from quantization.saw.quant_layer import Basic2DquantLayer
        for bits in (4, 6):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temp:
                torch.manual_seed(42)
                config = quantization_preset(bits, Path(temp))
                component = dict(config["Vae"])
                component["hadamard_dobi"] = config["hadamard_dobi"]
                original = QuantWrapper(self._toy_vae(), QuantizeModel_config(component), "cpu")
                original.set_quant_state(True, True)
                model = original.model
                modules = dict(model.model.named_modules())
                for stage in ("vae_encoder", "vae_decoder"):
                    items = []
                    for record in enumerate_candidates(model, stage):
                        layer = modules[record["module_name"]]
                        layer.configure_precal_joint_wa_input_projector(rank=32)
                        layer.configure_hadamard(True, 16)
                        with torch.no_grad():
                            basis = torch.linalg.qr(torch.randn(layer.weight.shape[1], 32))[0].T
                            layer.osr_lora_A.copy_(basis)
                            layer.osr_lora_B.copy_(layer.weight.reshape(layer.weight.shape[0], -1) @ basis.T)
                        items.append({**record, "shape": list(layer.weight.shape)})
                    payload = manifest(stage, bits, items)
                    payload["kind_counts"] = {"resblock_shortcut": 1}
                    (Path(temp) / MANIFEST_NAMES[stage]).write_text(json.dumps(payload))
                # Exercise range expansion and dtype preservation, not just scalar reloads.
                for layer in modules.values():
                    if not hasattr(layer, "act_quantizer_0"):
                        continue
                    for attr, shape, dtype in (
                        ("act_quantizer_0", (1, layer.weight.shape[1], 1, 1), torch.float64),
                        ("weight_quantizer_0", (layer.weight.shape[0], 1, 1, 1), torch.float32),
                    ):
                        quantizer = getattr(layer, attr)
                        quantizer.x_min = nn.Parameter(torch.full(shape, -1.25, dtype=dtype))
                        quantizer.x_max = nn.Parameter(torch.full(shape, 1.25, dtype=dtype))
                original.set_running_stat(False)
                original.set_all_init(True, "all")
                original.set_all_recon(True)
                original.set_all_recon_init(True)
                original.eval()
                torch.save(original.state_dict(), component["quant_ckpt"])
                # The runtime reconstructs from the same SD2.1/OSEDiff base.
                # Source layers keep original-weight/bias tensors outside state_dict.
                torch.manual_seed(42)
                restored = _load_component(self._toy_vae(), config["Vae"], config, "vae", "cpu")
                expected = original.state_dict()
                actual = restored.state_dict()
                self.assertEqual(set(expected), set(actual))
                for name in expected:
                    self.assertEqual(expected[name].dtype, actual[name].dtype, name)
                    self.assertTrue(torch.equal(expected[name], actual[name]), name)
                inputs = torch.randn(1, 64, 3, 3)
                with torch.no_grad():
                    self.assertTrue(torch.equal(original(inputs), restored(inputs)))
                self.assertTrue(all(not p.requires_grad for p in restored.parameters()))

    def test_checkpoint_construction_keeps_stem_policy_outside_yaml(self):
        from runtime.presets import quantization_preset
        from quantization.saw.quant_config import QuantizeModel_config
        from quantization.saw.quant_model import QuantWrapper
        for bits in (4, 6):
            toy = nn.Module()
            toy.input_blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(4, 64, 3, padding=1))])
            toy.time_embed = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
            config = quantization_preset(bits, Path("unused"))
            component = dict(config["Unet"])
            component["hadamard_dobi"] = config["hadamard_dobi"]
            wrapped = QuantWrapper(toy, QuantizeModel_config(component), "cpu")
            model = wrapped.model.model
            self.assertEqual(model.input_blocks[0][0].act_quantizer_0.quant_bits, 8)
            self.assertEqual(model.time_embed[0].weight_quantizer_0.quant_bits, 8)
            self.assertEqual(model.time_embed[2].weight_quantizer_0.quant_bits, bits)


if __name__ == "__main__":
    unittest.main()
