"""QDM H3 RGBA VAE support without modifying ComfyUI's global loader."""
import torch
import comfy.sd
import comfy.ops
import comfy.utils
import comfy.model_management as mm
import comfy.model_patcher
import folder_paths


def validate_h3_rgba_state(state):
    expected = {'encoder.conv_in.weight': (128, 4, 3, 3, 3),
                'decoder.proj_out.weight': (4096, 2048),
                'decoder.proj_out.bias': (4096,),
                'post_quant_conv.weight': (24, 24, 1, 1, 1)}
    for key, shape in expected.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f'Not a native H3 RGBA VAE: {key} must have shape {shape}')


class H3RGBAVAE(comfy.sd.VAE):
    def __init__(self, state, dtype=None, device=None):
        validate_h3_rgba_state(state)
        try:
            from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
        except ImportError as exc:
            raise RuntimeError('Update ComfyUI to a version with native MiniMax H3 support') from exc
        # Initialize the public wrapper defaults, then supply the RGBA core.
        # Empty weights stop the stock constructor before architecture loading.
        super().__init__(sd={})
        self.first_stage_model = MiniMaxH3VideoVAE(in_channels=4, out_ch=4,
                                                  operations=comfy.ops.disable_weight_init)
        model = self.first_stage_model
        model.pixel_mean = torch.cat((model.pixel_mean, torch.full((1, 1, 1, 1, 1), .5)), dim=1)
        model.pixel_std = torch.cat((model.pixel_std, torch.full((1, 1, 1, 1, 1), .5)), dim=1)
        # Normalization statistics are mandatory in QDM H3 exports.
        model.load_state_dict(state, strict=True, assign=True)
        self.latent_channels, self.latent_dim, self.output_channels = 24, 3, 4
        self.pad_channel_value = 1.0
        self.upscale_ratio = (lambda n: max(1, (n - 2) // 5 * 17 + 5), 16, 16)
        self.downscale_ratio = (lambda n: max(1, (n - 5) // 17 * 5 + 2) if n > 1 else 1, 16, 16)
        self.upscale_index_formula = self.downscale_index_formula = (4, 16, 16)
        self.working_dtypes = [torch.float16, torch.float32]
        self.handles_tiling = True
        self.process_output = lambda pixels: pixels
        self.device = device if device is not None else mm.vae_device()
        self.vae_dtype = dtype or mm.vae_dtype(self.device, self.working_dtypes)
        model.eval().requires_grad_(False).to(dtype=self.vae_dtype)
        if hasattr(mm, 'archive_model_dtypes'):
            mm.archive_model_dtypes(model)
        self.output_device = mm.intermediate_device()
        # Static patcher is also supported by Comfy builds without dynamic VRAM.
        self.patcher = comfy.model_patcher.ModelPatcher(model, load_device=self.device,
                                                       offload_device=mm.vae_offload_device())
        self.memory_used_encode = lambda shape, dt: (1_300_000_000 + 13 * min(shape[2], 17) * shape[3] * shape[4]) * mm.dtype_size(dt)
        self.memory_used_decode = lambda shape, dt: (300_000_000 + 13 * min(self.upscale_ratio[0](shape[2]), 30) * shape[3] * shape[4] * 256) * mm.dtype_size(dt)
        self.model_size()


class H3RGBAVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'vae_name': (folder_paths.get_filename_list('vae'),),
                             'precision': (['auto', 'fp32', 'fp16'],)}}

    RETURN_TYPES = ('VAE',)
    FUNCTION = 'load'
    CATEGORY = 'QDM/Transparency/H3'
    DESCRIPTION = 'Loads a trained H3 RGBA VAE. Shared by normal H3 and Ref2VA; retains 24 latent channels.'

    def load(self, vae_name, precision='auto'):
        path = folder_paths.get_full_path_or_raise('vae', vae_name)
        state = comfy.utils.load_torch_file(path, safe_load=True)
        return (H3RGBAVAE(state, dtype={'auto': None, 'fp32': torch.float32, 'fp16': torch.float16}[precision]),)


class H3RGBAEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'pixels': ('IMAGE',), 'vae': ('VAE',)},
                'optional': {'mask': ('MASK',)}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'encode'
    CATEGORY = 'QDM/Transparency/H3'
    DESCRIPTION = 'Encode RGBA frames. Optional Comfy mask uses 1=transparent, 0=opaque; a supplied mask overrides image alpha.'

    def encode(self, pixels, vae, mask=None):
        if not isinstance(vae, H3RGBAVAE):
            raise ValueError('Connect the QDM H3 RGBA VAE Loader')
        if mask is not None:
            if mask.shape != pixels.shape[:-1]:
                raise ValueError('Mask and image frame count/size must match')
            pixels = torch.cat((pixels[..., :3], 1 - mask.to(pixels).unsqueeze(-1)), dim=-1)
        return ({'samples': vae.encode(pixels)},)


class H3RGBADecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples': ('LATENT',), 'vae': ('VAE',)}}

    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK')
    RETURN_NAMES = ('rgba', 'rgb', 'mask')
    FUNCTION = 'decode'
    CATEGORY = 'QDM/Transparency/H3'
    DESCRIPTION = 'Decode H3 video or joint audio/video latents. Returns RGBA frames and a Comfy transparency mask; the audio stream is left unchanged.'

    def decode(self, samples, vae):
        if not isinstance(vae, H3RGBAVAE):
            raise ValueError('Connect the QDM H3 RGBA VAE Loader')
        latent = samples['samples']
        if getattr(latent, 'is_nested', False):
            streams = getattr(latent, 'tensors', ())
            if len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24:
                raise ValueError('Expected an H3 joint video/audio latent')
            latent = streams[0]
        pixels = vae.decode(latent)
        if pixels.ndim == 5:
            pixels = pixels.flatten(0, 1)  # Comfy IMAGE is [frames,H,W,C]
        if pixels.shape[-1] != 4:
            raise ValueError('H3 VAE did not produce RGBA frames')
        return pixels, pixels[..., :3], 1 - pixels[..., 3]


NODE_CLASS_MAPPINGS = {'QDMH3RGBAVAELoader': H3RGBAVAELoader,
                       'QDMH3RGBAEncode': H3RGBAEncode, 'QDMH3RGBADecode': H3RGBADecode}
NODE_DISPLAY_NAME_MAPPINGS = {'QDMH3RGBAVAELoader': 'H3 RGBA VAE Loader (QDM)',
                              'QDMH3RGBAEncode': 'H3 RGBA Encode (QDM)',
                              'QDMH3RGBADecode': 'H3 RGBA Decode (QDM)'}
