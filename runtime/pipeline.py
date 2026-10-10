"""Single-step SR pipeline extracted from PassionSR; tensor operations retained."""
import torch
import numpy as np
from omegaconf import OmegaConf
from diffusers import DDIMScheduler
from ldm.lora.load_model import load_model_from_config
from ldm.lora.lora_unet_loader import load_lora_unet_OSE, load_lora_vae_OSE
from my_utils.vaehook import VAEHook
from runtime.checkpoint import load_Quantmodel

class InferencePipeline(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        config_path = opt['basic_config']['config']
        ckpt_path = opt['basic_config']['ckpt']
        config = OmegaConf.load(config_path)
        # PassionSR supplies its precomputed empty context embedding directly to
        # the UNet, so the OpenCLIP text encoder is not needed for inference.
        config.model.params.cond_stage_config = "__is_unconditional__"

        self.model = load_model_from_config(config, ckpt_path)
        self.vae = self.model.first_stage_model
        self.unet = self.model.model.diffusion_model
        self.unet_config = self.model.model.diffusion_model_config["params"]
        self.encode_scaling_factor = self.model.scale_factor
        self.decode_scaling_factor = self.model.scale_factor
        del self.model
        torch.cuda.empty_cache()

        self.device = opt['device']
        self.unet = self.unet.to(self.device)
        self.vae = self.vae.to(self.device)
        self.opt = opt
        self.weight_dtype = torch.float32
        self.timesteps = torch.tensor([999], device=self.device).long()
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            opt['basic_config']['pretrained_model_name_or_path'], subfolder="scheduler"
        )
        self.noise_scheduler.set_timesteps(1, device=self.device)
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self.load_context_embedding(opt)
        self._init_tiled_vae(encoder_tile_size=opt["tile_config"]["vae_encoder_tiled_size"], decoder_tile_size=opt["tile_config"]["vae_decoder_tiled_size"])

        lora_weights_path = opt['basic_config']['lora_weights_path']
        merge_lora = opt['basic_config']['merge_lora']
        self.load_lora_ckpt(lora_weights_path, merge_lora)

        if opt['quantize_config']['quantize']:
            self.quant_config = opt['quantize_config']
            if self.quant_config["only_Unet"]:
                self.unet = load_Quantmodel(self.unet, self.quant_config, self.device)
            else:
                self.unet, self.vae = load_Quantmodel((self.unet, self.vae), self.quant_config, self.device)
        self.unet, self.vae = self.unet.to(self.device), self.vae.to(self.device)      
        if opt.get("warmup", True):
            self.preheat()

    def preheat(self,):
        input = torch.randn(1, 3, 512, 512, device=self.device, dtype=self.weight_dtype)
        _ = self.forward(input)

    def load_context_embedding(self, opt):
        empty_context_embedding = torch.load(opt['basic_config']["context_embedding_path"]).to(self.device)
        self.empty_context_embedding = empty_context_embedding
        # self.empty_context_embedding = torch.rand_like(empty_context_embedding, dtype=torch.float)
    
    def load_lora_ckpt(self, lora_weights_path, merge_lora):
        self.unet = load_lora_unet_OSE(self.unet, lora_weights_path, merge_lora)
        # save_model_to_txt(self.unet, "results/ldm/Unet.txt")
        self.vae = load_lora_vae_OSE(self.vae, lora_weights_path, merge_lora)
        # save_model_to_txt(self.vae, "results/ldm/vae.txt")

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet_config["in_channels"], 1, 1))

    @torch.no_grad()
    def forward(self, lq):
        # lq_latent_model = self.model.get_first_stage_encoding(self.model.encode_first_stage(lq))
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).sample() * self.encode_scaling_factor
        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.opt["tile_config"]["latent_tiled_size"], self.opt["tile_config"]["latent_tiled_overlap"])
        if h * w <= tile_size * tile_size:
            # print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            model_pred = self.unet(lq_latent, self.timesteps, self.empty_context_embedding)
        else:
            # print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        model_out = self.unet(input_list_t, self.timesteps, self.empty_context_embedding)
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        x_0 = get_x0_from_noise(
            lq_latent.double(),
            model_pred.double(), self.alphas_cumprod.double(), self.timesteps
        ).float()
        output_image = (self.vae.decode(x_0.to(self.weight_dtype) / self.decode_scaling_factor)).clamp(-1, 1)
        return output_image

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample
