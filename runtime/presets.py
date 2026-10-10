"""Checkpoint construction metadata, deliberately kept out of public YAMLs.

Bit-width attributes are not serialized by state_dict. These two released
checkpoints therefore require their original construction-time precision rules.
This module creates structure only; it does not calibrate or select ranges.
"""


def quantization_preset(bits, checkpoint_dir):
    if bits not in (4, 6):
        raise ValueError("Only the released 4-bit and 6-bit presets are supported")
    defaults = {
        "quantype": "PTQ", "method": "saw_sep", "only_weight": False,
        "weight_quant_bits": bits, "weight_sym": False, "weight_sign": False,
        "act_quant_bits": bits, "act_sign": False, "act_sym": False,
        "layer_type": "2Dquant", "s_alpha": 0.3,
    }
    return {
        "quantize": True,
        "only_Unet": False,
        "Unet": {
            **defaults, "split": True,
            "quant_ckpt": str(checkpoint_dir / "unet_ckpt_merge_saw_sep.pth"),
            "mixed_precision": {
                "stem_w8a8": {"enabled": True, "target": "input_blocks.0.0"},
                "f3_projection_w8a8": {"enabled": False},
            },
            "low_rank_absorb": {"enabled": False},
        },
        "Vae": {
            **defaults, "split": False,
            "quant_ckpt": str(checkpoint_dir / "vae_ckpt_merge_saw_sep.pth"),
            "low_rank_absorb": {
                "encoder": {"enabled": False}, "decoder": {"enabled": False},
            },
        },
        "precal_low_rank_preservation": {
            "enabled": True, "version": "precal_joint_wa_input_hardness_v2",
            "scope": "pipeline_wise", "rank": 32,
        },
        "hadamard_dobi": {
            "enabled": True,
            "blockwise": {
                "enabled": True, "version": "posth_channelwise_v2",
                "managed_tensorwise_asydobi": {
                    "enabled": True, "paired_qk_managed": True,
                    "attention_wv_managed": True,
                },
            },
            "post_h_protected_initializer": {
                "version": "posth_channelwise_best_v4_dense32_managed262k",
            },
        },
    }
