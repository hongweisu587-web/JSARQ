# W6A6 PTQ artifacts

This directory includes the following five files from the W6A6 `PTQ/` result:

1. `unet_ckpt_merge_saw_sep.pth`
2. `vae_ckpt_merge_saw_sep.pth`
3. `unet_precal_low_rank_preservation_manifest.json`
4. `vae_encoder_precal_low_rank_preservation_manifest.json`
5. `vae_decoder_precal_low_rank_preservation_manifest.json`

Use this version's own manifests: their runtime activation coordinate is
`post_let_post_h_residual_pre_a6`. Do not reuse the W4A4 manifests.
