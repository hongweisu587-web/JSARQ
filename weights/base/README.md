# Shared model artifacts

The OSEDiff adapter and empty context embedding are included. Prepare the
following SD2.1 files using the PassionSR baseline model setup:

- `v2-inference-v-fp32.yaml`: the original SD2.1 LDM architecture configuration.
- `v2-1_512-ema-pruned.ckpt`: the original base model checkpoint.
- `osediff.ckpt`: the OSEDiff adapter checkpoint.
- `empty_context_embedding.pt`: the precomputed empty context embedding.
- `sd21/scheduler/scheduler_config.json`: the scheduler configuration, copied
  from the original `pretrained_model_name_or_path/scheduler/` directory.

The SD2.1 checkpoint, architecture configuration and scheduler configuration
are external pretrained-model dependencies. The text encoder is not used.
