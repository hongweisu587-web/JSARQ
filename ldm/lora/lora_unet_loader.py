from peft import LoraConfig, get_peft_model
import torch
from ldm.lora.lora_weight import convert_unet_target_module, convert_unet_state_dict, convert_vae_target_module, convert_vae_state_dict

def load_lora_unet_OSE(model, lora_weights_path, merge_lora=False):
    model_ckpt = torch.load(lora_weights_path, map_location="cpu")
    rank_unet = model_ckpt["rank_unet"]
    unet_lora_encoder_modules = convert_unet_target_module(model_ckpt["unet_lora_encoder_modules"])
    unet_lora_decoder_modules = convert_unet_target_module(model_ckpt["unet_lora_decoder_modules"])
    unet_lora_others_modules = convert_unet_target_module(model_ckpt["unet_lora_others_modules"])
    
    unet_lora_encoder_modules.extend(["input_blocks.3.0.op", "input_blocks.6.0.op", "input_blocks.9.0.op"])
    # i don't understand why missing three blocks

    # load unet lora
    lora_conf_encoder = LoraConfig(r=rank_unet, init_lora_weights="gaussian", target_modules=unet_lora_encoder_modules)
    lora_conf_decoder = LoraConfig(r=rank_unet, init_lora_weights="gaussian", target_modules=unet_lora_decoder_modules)
    lora_conf_others = LoraConfig(r=rank_unet, init_lora_weights="gaussian", target_modules=unet_lora_others_modules)
    
    model = get_peft_model(model, lora_conf_encoder, adapter_name="default_encoder").base_model.model
    model = get_peft_model(model, lora_conf_decoder, adapter_name="default_decoder").base_model.model
    peft_model = get_peft_model(model, lora_conf_others, adapter_name="default_others")
    model = peft_model.base_model.model

    unet_state_dict = model_ckpt["unet_state_dict"]
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    
    # lora_state_dict = {}
    # for key, value in unet_state_dict.items():
    #     if "lora" in key:
    #         lora_state_dict[key] = value
    lora_state_dict = unet_state_dict
    
    state_dict_info = model.load_state_dict(lora_state_dict, strict=False)
    
    if len(state_dict_info.unexpected_keys) == 0:
        print("OSEDiff Unet is loaded successfully!")
    else:
        print("Unexpected keys:", state_dict_info.unexpected_keys)
        
    if merge_lora:
        model = peft_model.merge_and_unload(
            adapter_names=["default_encoder", "default_decoder", "default_others"]
        )
    return model

def load_lora_vae_OSE(model, lora_weights_path, merge_lora=False):
    model_ckpt = torch.load(lora_weights_path, map_location="cpu")
    rank_vae = model_ckpt["rank_vae"]
    # print(model_ckpt["vae_lora_encoder_modules"])
    vae_lora_encoder_modules = convert_vae_target_module(model_ckpt["vae_lora_encoder_modules"])
    vae_lora_decoder_modules = convert_vae_target_module(model_ckpt["vae_lora_decoder_modules"])

    # print(vae_lora_encoder_modules)
    vae_lora_conf_encoder = LoraConfig(r=rank_vae, init_lora_weights="gaussian", target_modules=vae_lora_encoder_modules)
    vae_lora_conf_decoder = LoraConfig(r=rank_vae, init_lora_weights="gaussian", target_modules=vae_lora_decoder_modules)
    
    peft_model = get_peft_model(model, vae_lora_conf_encoder, adapter_name="default_encoder")
    model = peft_model.base_model.model
    # model = get_peft_model(model, vae_lora_conf_decoder, adapter_name="default_decoder").base_model.model
    vae_state_dict = model_ckpt["vae_state_dict"]
    vae_state_dict = convert_vae_state_dict(vae_state_dict)
    
    lora_state_dict = {}
    for key, value in vae_state_dict.items():
        # if "lora" or "input_blocks.0.0" in key:
        #     lora_state_dict[key] = value
        if "default_decoder" in key:
            continue
        else:
            lora_state_dict[key] = value
    
    # lora_state_dict = vae_state_dict
    state_dict_info = model.load_state_dict(lora_state_dict, strict=False)
    if len(state_dict_info.unexpected_keys) == 0:
        print("OSEDiff vae is loaded successfully!")
    else:
        print("Unexpected keys:", state_dict_info.unexpected_keys)
    
    if merge_lora:
        model = peft_model.merge_and_unload(adapter_names=["default_encoder"])
    return model
