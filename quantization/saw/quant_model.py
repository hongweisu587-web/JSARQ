import torch.nn as nn

from .quant_config import Quantize_config, QuantizeModel_config

from .quant_block import get_specials, BaseQuantBlock

from .quant_block import QuantBasicTransformerBlock

from .quant_block import QuantBasicTransformerBlock, QuantAttnBlock

from .quant_layer import QuantLayer, StraightThrough, BasicQuantLayer, Basic2DquantLayer, get_layer_type

from .saw_layer import saw_Linear_QuantLayer, saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer

from .channelwise_asydobi import apply_f3_projection_w8a8, apply_stem_w8a8

from ldm.modules.attention import BasicTransformerBlock

from ldm.modules.attention import MemoryEfficientCrossAttention

import torch

import copy

unet_first_layer_name = ["time_embed.0", "out.2"]

vae_first_layer_name = ["quant_conv", "post_quant_conv", "encoder.conv_out", "decoder.conv_out"]

first_layer_name = unet_first_layer_name + vae_first_layer_name

class QuantWrapper(nn.Module):
    """
    A wrapper module for quantizing neural network models.
    This class wraps a given PyTorch model with quantization logic using the QuantModel class.
    It allows for easy integration of quantization into existing models, and delegates attribute
    access to the underlying quantized model or the original model as needed.
    Args:
        model (nn.Module): The PyTorch model to be quantized.
        quant_config: Configuration object or dictionary specifying quantization parameters.
        device (str, optional): Device to place the quantized model on. Defaults to "cuda".
        exclude (optional): List or set of module names to exclude from quantization.
    Methods:
        forward(x, timesteps=None, encoder_hidden_states=None):
            Performs a forward pass through the quantized model.
        __getattr__(name):
            Delegates attribute access to the quantized model or the original model if the attribute
            is not found in the wrapper itself.
    """
    def __init__(self, model: nn.Module, quant_config, device="cuda", exclude=None):
        nn.Module.__init__(self)
        self.model = QuantModel(model, quant_config, device, exclude)

    def forward(self, x, timesteps=None, encoder_hidden_states=None):
        return self.model(x, timesteps, encoder_hidden_states)
    
    def __getattr__(self, name):
        if name == "model":
            return super().__getattr__(name)
        
        if hasattr(self.model, name):
            return getattr(self.model, name)
        
        return getattr(self.model.model, name)

class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, quant_config: QuantizeModel_config, device="cuda", exclude=None):
        super().__init__()
        self.register_buffer(
            "qk_lars_applied", torch.tensor(False, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "qk_lars_version", torch.tensor(0, dtype=torch.int32), persistent=True
        )
        self.register_buffer(
            "paired_qk_hadamard_calibrated",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "paired_qk_hadamard_version",
            torch.tensor(0, dtype=torch.int32),
            persistent=True,
        )
        self.model = model
        self.device = device
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        if hasattr(model, 'in_channels'):
            self.in_channels = model.in_channels
        hybrid_version = quant_config.hadamard_dobi.get(
            "post_h_hybrid_initializer", {}
        ).get("version")
        reference_version = quant_config.hadamard_dobi.get(
            "post_h_reference_initializer", {}
        ).get("version")
        protected_version = quant_config.hadamard_dobi.get(
            "post_h_protected_initializer", {}
        ).get("version")
        self.specials = get_specials(
            True,
            include_vae_attention=(
                hybrid_version
                == "posth_hybrid_global_channelwise_reference_attention_v1"
                or reference_version
                in {
                    "posth_reference_global_channelwise_native_attention_v1",
                    "posth_reference_tensorwise_native_v2",
                }
                or protected_version in {
                    "posth_protected_historical_channelwise_reference_attention_v1",
                    "posth_runtime_trace_operator_grouped_lazy_cache_v10",
                    "posth_channelwise_best_v4_dense32",
                    "posth_channelwise_best_v4_dense32_managed262k",
                    "posth_channelwise_best_v4_dense32_asydobi_then_joint_wa",
                }
            ),
        )

        self.exclude = None
        self.quant_config = quant_config
        self.low_rank_absorb_manifest = []
        self.low_rank_absorb_records = []
        self.low_rank_absorb_records_encoder = []
        self.low_rank_absorb_records_decoder = []
        self.first_layer_quant_config = copy.deepcopy(quant_config)
        self.first_layer_quant_config.weight_config.quant_bits = 8
        self.first_layer_quant_config.act_config.quant_bits = 8
        self.quant_module_refactor(self.model, quant_config)
        self.quant_block_refactor(self.model, quant_config)
        self.stem_precision_overrides = apply_stem_w8a8(
            self.model, quant_config.quantize_config
        )
        if self.stem_precision_overrides:
            print(
                "[Stem-W8A8] fixed layers="
                + ",".join(
                    entry["module"] for entry in self.stem_precision_overrides
                )
            )
        # Python-side bit-width properties are not stored in state_dict, so
        # applying this at construction makes PTQ and inference loading obey
        # the same exact four-projection W8A8 topology.
        self.f3_precision_overrides = apply_f3_projection_w8a8(
            self.model, quant_config.quantize_config
        )
        if self.f3_precision_overrides:
            print(
                "[F3-W8A8] fixed projections="
                + ",".join(
                    entry["module"] for entry in self.f3_precision_overrides
                )
            )
        self.quant_layer_type = get_layer_type(quant_config)

        # ``quant_block_refactor`` creates attention quantizers, LET scale
        # parameters and Paired-Q/K state buffers after the source model has
        # already been moved to ``device``.  Newly constructed torch modules
        # default to CPU, so without this final migration the original UNet
        # projections live on CUDA while (in particular) the per-head Q/K
        # ranges remain on CPU.  Move the complete, final module tree once all
        # replacements are installed so every newly created parameter/buffer
        # follows the same device contract as the wrapped model.
        self.to(device)

    def set_qk_lars_state(self, applied, version=2):
        self.qk_lars_applied.fill_(bool(applied))
        self.qk_lars_version.fill_(int(version))

    def get_qk_lars_state(self):
        return {
            "applied": bool(self.qk_lars_applied.detach().cpu().item()),
            "version": int(self.qk_lars_version.detach().cpu().item()),
        }

    def set_paired_qk_hadamard_state(self, calibrated, version=1):
        self.paired_qk_hadamard_calibrated.fill_(bool(calibrated))
        self.paired_qk_hadamard_version.fill_(int(version))

    def get_paired_qk_hadamard_state(self):
        return {
            "calibrated": bool(
                self.paired_qk_hadamard_calibrated.detach().cpu().item()
            ),
            "version": int(
                self.paired_qk_hadamard_version.detach().cpu().item()
            ),
        }

    def sync_paired_qk_hadamard_runtime(self):
        for module in self.model.modules():
            if isinstance(module, QuantBasicTransformerBlock):
                module.sync_paired_qk_hadamard_runtime()

    def report_paired_qk_hadamard_state(self, component="unet"):
        modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, QuantBasicTransformerBlock):
                modules.append(
                    {
                        "module": f"{name}.attn2",
                        "selected": bool(
                            module.attn2._paired_qk_hadamard_runtime_enabled
                        ),
                        "block_size": int(
                            module.attn2._paired_qk_hadamard_runtime_block_size
                        ),
                        "heads": int(module.attn2.heads),
                    }
                )
        selected = sum(entry["selected"] for entry in modules)
        state = self.get_paired_qk_hadamard_state()
        print(
            "[Paired-QK-H][load] "
            f"{component} calibrated={state['calibrated']} "
            f"version={state['version']} selected={selected}/{len(modules)}"
        )
        return modules

    def quant_module_refactor(self, module: nn.Module, quant_config: QuantizeModel_config, module_name="model"):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """

        for name, child_module in module.named_children():
            whole_name = module_name + "." + name
            if self.exclude is not None and self.exclude in name:
                    continue
            elif isinstance(child_module, nn.Linear):
                if any(substring in whole_name for substring in first_layer_name):
                    print(f"{whole_name} is first layer, set precision to 8")
                    setattr(module, name, saw_Linear_QuantLayer(
                        child_module, self.first_layer_quant_config, self.device))
                else:
                    setattr(module, name, saw_Linear_QuantLayer(
                        child_module, quant_config, self.device))
            elif isinstance(child_module, nn.Conv1d):
                if any(substring in whole_name for substring in first_layer_name):
                    print(f"{whole_name} is first layer, set precision to 8")
                    setattr(module, name, saw_Conv1d_QuantLayer(
                        child_module, self.first_layer_quant_config, self.device))
                else:
                    setattr(module, name, saw_Conv1d_QuantLayer(
                        child_module, quant_config, self.device))
            elif isinstance(child_module, nn.Conv2d):
                if any(substring in whole_name for substring in first_layer_name):
                    print(f"{whole_name} is first layer, set precision to 8")
                    setattr(module, name, saw_Conv2d_QuantLayer(
                        child_module, self.first_layer_quant_config, self.device))
                else:
                    setattr(module, name, saw_Conv2d_QuantLayer(
                        child_module, quant_config, self.device))
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, quant_config, module_name=whole_name)

    def quant_block_refactor(self, module: nn.Module, quant_config: QuantizeModel_config):
        for name, child_module in module.named_children():
            if (
                type(child_module) in self.specials
                and self.specials[type(child_module)] is QuantAttnBlock
            ):
                setattr(module, name, self.specials[type(child_module)](child_module,
                        quant_config))
            else:
                self.quant_block_refactor(child_module, quant_config)
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        quant_config))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        quant_config))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        quant_config))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        quant_config))
            else:
                self.quant_block_refactor(child_module, quant_config)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantLayer, QuantAttnBlock,
                              saw_Linear_QuantLayer, QuantBasicTransformerBlock,
                              saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)

    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.set_running_stat(running_stat)
                    m.attn2.act_quantizer_w.set_running_stat(running_stat)
                else:
                    m.attn1.act_quantizer_q.set_running_stat(running_stat)
                    m.attn1.act_quantizer_k.set_running_stat(running_stat)
                    m.attn1.act_quantizer_v.set_running_stat(running_stat)
                    m.attn1.act_quantizer_w.set_running_stat(running_stat)
                    m.attn2.act_quantizer_q.set_running_stat(running_stat)
                    m.attn2.act_quantizer_k.set_running_stat(running_stat)
                    m.attn2.act_quantizer_v.set_running_stat(running_stat)
                    m.attn2.act_quantizer_w.set_running_stat(running_stat)
                m.attn1.running_stat = running_stat
                m.attn2.running_stat = running_stat
            if isinstance(m, QuantLayer):
                m.set_running_stat(running_stat)
            if isinstance(m, (saw_Linear_QuantLayer, saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer)):
                m.set_running_stat(running_stat)
            if isinstance(m, QuantAttnBlock):
                m.set_running_stat(running_stat)

    def set_record(self, record: bool):
        for m in self.model.modules():
            if isinstance(m, saw_Linear_QuantLayer):
                m.set_record(record)

    def set_all_init(self, flag, user="all"):
        for name, child_module in self.model.named_modules():
            if user == "all" or user == 'quant':
                if isinstance(child_module, self.quant_layer_type):
                    if isinstance(child_module, BasicQuantLayer):
                        if child_module.delta.numel() == 1:
                            if child_module.delta.data.item() == 1.0:
                                child_module.set_init_state(False)
                        else:
                            child_module.set_init_state(flag)
                    elif isinstance(child_module, Basic2DquantLayer):
                        if child_module.x_max.numel() == 1:
                            if child_module.x_max.data.item() == 0.5 and child_module.x_min.data.item() == -0.5:
                                child_module.set_init_state(False)
                                print(name)
                        else:
                            child_module.set_init_state(flag)
                    child_module.set_init_state(flag)

    def set_all_recon_init(self, flag):
        for name, child_module in self.model.named_modules():
            if isinstance(child_module, (saw_Linear_QuantLayer,
                            saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer)):
                child_module.recon_inited = flag
            if isinstance(child_module, QuantAttnBlock):
                child_module.scale_init_flag = flag
            if isinstance(child_module, QuantBasicTransformerBlock):
                child_module.attn1.scale_init_flag = flag
                child_module.attn2.scale_init_flag = flag

    def set_all_recon(self, flag):
        for name, child_module in self.model.named_modules():
            if isinstance(child_module, (saw_Linear_QuantLayer,
                                        saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer)):
                child_module.recon_flag = flag
            if isinstance(child_module, QuantAttnBlock):
                child_module.recon_flag = flag
            if isinstance(child_module, QuantBasicTransformerBlock):
                child_module.recon_flag = flag
                child_module.attn1.recon_flag = flag
                child_module.attn2.recon_flag = flag

    def report_hadamard_state(self, component="model"):
        selected = []
        for name, child_module in self.model.named_modules():
            if isinstance(child_module, (saw_Linear_QuantLayer,
                                         saw_Conv1d_QuantLayer,
                                         saw_Conv2d_QuantLayer)):
                if child_module.hadamard_runtime_enabled:
                    selected.append(
                        (name, child_module.hadamard_runtime_block_size)
                    )
        print(f"[Hadamard+DoBi][load] {component} selected Hadamard layers={len(selected)}")
        return selected

