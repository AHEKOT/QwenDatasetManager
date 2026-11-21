"""Joint QI2 layer diffusion. Height concatenation is storage, never VAE geometry."""

import torch
from .qwen_image_2 import QwenImage2Model
from toolkit.layered_psd import composite_layers


class QwenImage2LayeredModel(QwenImage2Model):
    arch = "qwen_image_2_layered"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_space_version = "qwen_image_2_layered_v1"
        _ = self.target_layer_count  # validate before weights are loaded

    def load_model(self):
        from toolkit.layered_psd import _psd_api
        _psd_api()  # report a missing PSD dependency before allocating model weights
        super().load_model()

    @property
    def target_layer_count(self):
        count = self.model_config.model_kwargs.get("layer_slots", 4)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
            raise ValueError("layer_slots must be an integer from 1 to 20")
        return count

    @property
    def load_rgba(self):
        return True

    @property
    def timestep_embedding(self):
        # Existing saved PSD jobs used a bf16 frequency buffer. Keep resumes
        # consistent; newly created jobs explicitly select the corrected fp32 mode.
        if 'timestep_embedding' not in self.model_config.model_kwargs:
            return 'legacy_bf16'
        return super().timestep_embedding

    def get_text_embedding_space_version(self):
        return super().get_text_embedding_space_version() + "_layers_v1"

    def encode_images(self, image_list, device=None, dtype=None):
        count = self.target_layer_count
        outputs = []
        for image in image_list:
            if image.ndim != 3 or image.shape[0] != 4 or image.shape[1] % count:
                raise ValueError("Layer targets must be RGBA canvases stacked bottom-to-top")
            # Encode each canvas independently, keeping VAE memory bounded.
            encoded = [super(QwenImage2LayeredModel, self).encode_images(
                [layer], device=device, dtype=dtype
            ) for layer in image.chunk(count, dim=1)]
            outputs.append(torch.cat(encoded, dim=2))
        return torch.cat(outputs, dim=0)

    def decode_latents(self, latents, device=None, dtype=None):
        if latents.shape[-2] % self.target_layer_count:
            raise ValueError("Invalid layered latent shape")
        return torch.cat([
            self._decode_rgba(layer, device=device, dtype=dtype)
            for layer in latents.chunk(self.target_layer_count, dim=2)
        ], dim=2)

    def decode_to_images(self, latents):
        images = []
        for sample in self.decode_latents(latents):
            layers = [self.image_tensor_to_pil(layer) for layer in sample.chunk(self.target_layer_count, dim=1)]
            preview = composite_layers(layers)
            # GenerateImageConfig writes this stack as a PSD next to the preview.
            preview.info["qdm_psd_layers"] = layers
            images.append(preview)
        return images
