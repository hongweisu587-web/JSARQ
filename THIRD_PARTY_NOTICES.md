# Third-party sources

This inference extraction is based on the provided PassionSR-main checkout.

- PassionSR: https://github.com/libozhu03/PassionSR
- Stable Diffusion/LDM: upstream references and existing notices remain in the
  copied `ldm/` source files.
- OSEDiff: adapter-loading and weight-conversion utilities remain in `ldm/lora/`.
- VAE tile implementation: the existing author and MIT-license notice remains
  at the beginning of `my_utils/vaehook.py`.

No project-wide LICENSE file was present in the source checkout during this
extraction. No new license has been assigned to third-party code or model
weights. The owner should include the applicable upstream license texts and
model distribution terms before public publication.

The full release package includes the supplied calibrated checkpoints and the
OSEDiff adapter/context artifacts. The SD2.1 base model is prepared separately.
This extraction does not claim ownership of the original code or models.
File-level source provenance is available in `docs/source_provenance.json`.
