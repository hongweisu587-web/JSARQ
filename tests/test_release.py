import copy
import json
from pathlib import Path
import tempfile
import unittest
import yaml
from runtime.artifacts import MANIFEST_NAMES, read_manifest
from runtime.config import PROJECT_ROOT, load_options
from inference import collect_images, prepare_image


def manifest(stage, bits, modules):
    return {
        "algorithm": "precal_joint_wa_input_hardness_v2",
        "branch_type": "precal_joint_wa_input_projector",
        "stage": stage, "rank": 32, "factor_dtype": "float32",
        "master_coordinate": "master_centered_input",
        "runtime_input": f"post_let_post_h_residual_pre_a{bits}",
        "statistics_sample_count": 480, "master_down_factor": "Vt",
        "master_up_factor": "WV", "candidate_count": len(modules), "modules": modules,
    }


class ReleaseTests(unittest.TestCase):
    def test_two_presets_and_project_relative_paths(self):
        for bits in (4, 6):
            options = load_options(PROJECT_ROOT / f"configs/w{bits}a{bits}.yaml")
            self.assertEqual(options["quantize_config"]["Unet"]["weight_quant_bits"], bits)
            self.assertEqual(options["quantize_config"]["Vae"]["act_quant_bits"], bits)
            self.assertEqual(Path(options["basic_config"]["config"]).parent,
                             PROJECT_ROOT / "weights/base")
            self.assertEqual(options["tile_config"]["latent_tiled_overlap"], 32)

    def test_public_configs_do_not_expose_precision_overrides(self):
        for path in (PROJECT_ROOT / "configs").glob("*.yaml"):
            config = yaml.safe_load(path.read_text())
            self.assertEqual(set(config["quantization"]),
                             {"bits", "checkpoint_dir", "jointwa_rank"})
            text = path.read_text().lower()
            for forbidden in ("mixed_precision", "stem_w8", "f3_projection", "w8a8",
                              "cali_img_path", "/home/", "cali_epochs"):
                self.assertNotIn(forbidden, text)

    def test_invalid_tile_overlap_and_unknown_quantization_field_rejected(self):
        config = yaml.safe_load((PROJECT_ROOT / "configs/w4a4.yaml").read_text())
        for mutation in ("overlap", "override"):
            bad = copy.deepcopy(config)
            if mutation == "overlap":
                bad["tiling"]["latent_overlap"] = bad["tiling"]["latent_size"]
            else:
                bad["quantization"]["mixed_precision"] = {}
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "bad.yaml"
                path.write_text(yaml.safe_dump(bad))
                with self.assertRaises(ValueError):
                    load_options(path)

    def test_manifest_must_match_activation_precision_and_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            item = {"module_name": "encoder.block.nin_shortcut", "shape": [48, 64, 1, 1]}
            path = Path(temp) / MANIFEST_NAMES["vae_encoder"]
            path.write_text(json.dumps(manifest("vae_encoder", 6, [item])))
            self.assertEqual(read_manifest(temp, "vae_encoder", 6)["rank"], 32)
            with self.assertRaises(ValueError):
                read_manifest(temp, "vae_encoder", 4)
            bad = manifest("vae_decoder", 6, [item])
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                read_manifest(temp, "vae_encoder", 6)

    def test_duplicate_manifest_entries_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            item = {"module_name": "encoder.block.nin_shortcut", "shape": [48, 64, 1, 1]}
            path = Path(temp) / MANIFEST_NAMES["vae_encoder"]
            path.write_text(json.dumps(manifest("vae_encoder", 4, [item, item])))
            with self.assertRaises(ValueError):
                read_manifest(temp, "vae_encoder", 4)

    def test_single_image_and_folder_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("b.JPG", "a.png", "notes.txt"):
                (root / name).touch()
            self.assertEqual([p.name for p in collect_images(root)], ["a.png", "b.JPG"])
            self.assertEqual(collect_images(root / "b.JPG"), [root / "b.JPG"])

    def test_resize_matches_original_output_convention(self):
        from PIL import Image
        basic = {"upscale": 4, "process_size": 128}
        image, restore = prepare_image(Image.new("RGB", (16, 24)), basic)
        self.assertEqual(image.size, (128, 192))
        self.assertEqual(restore, (64, 96))
        image, restore = prepare_image(Image.new("RGB", (101, 103)), basic)
        self.assertEqual(image.size, (400, 408))
        self.assertIsNone(restore)


if __name__ == "__main__":
    unittest.main()
