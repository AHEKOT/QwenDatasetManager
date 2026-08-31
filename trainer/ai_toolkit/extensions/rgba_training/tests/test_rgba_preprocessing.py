import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import RescaleTransform
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.rgba_utils import (
    ensure_normalized_rgba_tensor,
    prepare_rgba_image,
    resize_rgba_alpha_safe,
    rgba_tensor_to_rgb_control,
)


class RGBAPreprocessingTests(unittest.TestCase):
    def test_hidden_green_is_removed_but_visible_rgb_is_preserved(self):
        source = np.array(
            [[[0, 255, 0, 0], [255, 0, 0, 255], [25, 200, 30, 128]]],
            dtype=np.uint8,
        )
        result = np.asarray(prepare_rgba_image(Image.fromarray(source, "RGBA")))
        np.testing.assert_array_equal(result[0, 0], [0, 0, 0, 0])
        np.testing.assert_array_equal(result[0, 1], source[0, 1])
        np.testing.assert_array_equal(result[0, 2], source[0, 2])

    def test_alpha_safe_resize_does_not_bleed_hidden_green(self):
        source = np.zeros((2, 2, 4), dtype=np.uint8)
        source[:, 0] = [255, 0, 0, 255]
        source[:, 1] = [0, 255, 0, 0]
        image = prepare_rgba_image(Image.fromarray(source, "RGBA"))
        resized = np.asarray(resize_rgba_alpha_safe(image, (32, 2)))
        transparent_or_edge = resized[..., 3] < 255
        self.assertLessEqual(int(resized[..., 1][transparent_or_edge].max(initial=0)), 1)

    def test_known_green_matte_can_be_removed_from_partial_alpha(self):
        # 50% red foreground precomposited over green, with alpha retained.
        source = np.array([[[128, 127, 0, 128]]], dtype=np.uint8)
        result = np.asarray(prepare_rgba_image(
            Image.fromarray(source, "RGBA"),
            unblend_background=[0, 255, 0],
        ))
        self.assertGreaterEqual(int(result[0, 0, 0]), 250)
        self.assertLessEqual(int(result[0, 0, 1]), 2)

    def test_nearest_opaque_edge_correction_replaces_green_matte(self):
        source = np.array([[[255, 0, 0, 255], [0, 255, 0, 128], [0, 255, 0, 0]]], dtype=np.uint8)
        result = np.asarray(prepare_rgba_image(
            Image.fromarray(source, "RGBA"),
            edge_color_correction="nearest_opaque",
        ))
        np.testing.assert_array_equal(result[0, 1], [255, 0, 0, 128])
        np.testing.assert_array_equal(result[0, 2], [0, 0, 0, 0])

    def test_matte_despill_replaces_opaque_and_partial_green_boundary(self):
        source = np.zeros((9, 9, 4), dtype=np.uint8)
        source[1:8, 1:8] = [0, 255, 0, 255]
        source[2:7, 2:7] = [255, 0, 0, 255]
        source[0, 4] = [0, 255, 0, 128]
        result = np.asarray(prepare_rgba_image(
            Image.fromarray(source, "RGBA"),
            edge_color_correction="matte_despill",
            edge_matte_color=[0, 255, 0],
            edge_width=2,
        ))
        self.assertGreater(int(result[1, 4, 0]), int(result[1, 4, 1]))
        self.assertGreater(int(result[0, 4, 0]), int(result[0, 4, 1]))
        self.assertEqual(int(result[0, 4, 3]), 128)

    def test_rgb_tensor_gets_opaque_alpha(self):
        rgb = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
        rgba = ensure_normalized_rgba_tensor(rgb)
        self.assertEqual(tuple(rgba.shape), (2, 4, 4, 5))
        self.assertTrue(torch.equal(rgba[:, 3], torch.ones_like(rgba[:, 3])))

        video_rgb = torch.zeros((1, 3, 1, 4, 5), dtype=torch.float32)
        video_rgba = ensure_normalized_rgba_tensor(video_rgb)
        self.assertEqual(tuple(video_rgba.shape), (1, 4, 1, 4, 5))
        self.assertTrue(torch.equal(video_rgba[:, 3], torch.ones_like(video_rgba[:, 3])))

    def test_control_composite_ignores_hidden_rgb(self):
        rgba = torch.tensor(
            [[[-1.0]], [[1.0]], [[-1.0]], [[-1.0]]],
            dtype=torch.float32,
        )
        control = rgba_tensor_to_rgb_control(rgba, [255, 255, 255])
        self.assertTrue(torch.equal(control, torch.ones_like(control)))

    def test_dataset_path_preserves_alpha_and_generates_rgb_control(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "character.png"
            source = np.zeros((8, 8, 4), dtype=np.uint8)
            source[:] = [0, 255, 0, 0]
            source[2:6, 2:6] = [255, 0, 0, 255]
            Image.fromarray(source, "RGBA").save(path)

            config = DatasetConfig(
                folder_path=temp_dir,
                resolution=8,
                buckets=True,
                pixel_channels="rgba",
                rgba_generate_control=True,
                rgba_control_background=[255, 255, 255],
                num_workers=0,
            )
            transform = transforms.Compose([transforms.ToTensor(), RescaleTransform()])
            item = FileItemDTO(
                sd=None,
                path=str(path),
                dataset_config=config,
                dataloader_transforms=transform,
                size_database={},
                dataset_root=temp_dir,
                scale_to_width=8,
                scale_to_height=8,
                crop_width=8,
                crop_height=8,
            )
            # This is also the path used while caching latents/text embeddings.
            item.load_and_process_image(transform, only_load_latents=True)

            self.assertEqual(tuple(item.tensor.shape), (4, 8, 8))
            self.assertEqual(tuple(item.control_tensor.shape), (3, 8, 8))
            # Hidden green is black in the normalized RGBA target.
            self.assertTrue(torch.equal(item.tensor[:3, 0, 0], torch.full((3,), -1.0)))
            # The generated visual control composites that transparent pixel to white.
            self.assertTrue(torch.equal(item.control_tensor[:, 0, 0], torch.ones(3)))

    def test_generation_dataset_builds_empty_black_control(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "character.png"
            source = np.zeros((8, 8, 4), dtype=np.uint8)
            source[2:6, 2:6] = [255, 0, 0, 255]
            Image.fromarray(source, "RGBA").save(path)

            config = DatasetConfig(
                folder_path=temp_dir,
                resolution=8,
                buckets=True,
                pixel_channels="rgba",
                rgba_generate_control=True,
                rgba_control_mode="generation",
                num_workers=0,
            )
            transform = transforms.Compose([transforms.ToTensor(), RescaleTransform()])
            item = FileItemDTO(
                sd=None,
                path=str(path),
                dataset_config=config,
                dataloader_transforms=transform,
                size_database={},
                dataset_root=temp_dir,
                scale_to_width=8,
                scale_to_height=8,
                crop_width=8,
                crop_height=8,
            )
            item.load_and_process_image(transform, only_load_latents=True)

            self.assertEqual(tuple(item.tensor.shape), (4, 8, 8))
            self.assertEqual(tuple(item.control_tensor.shape), (3, 8, 8))
            self.assertTrue(torch.equal(item.control_tensor, torch.zeros_like(item.control_tensor)))
            item.encode_control_in_text_embeddings = True
            item.caption = "Generate a character"
            cache_info = item.get_text_embedding_info_dict()
            self.assertEqual(cache_info["rgba_control_mode"], "generation")
            self.assertNotIn("rgba_control_backgrounds", cache_info)

    def test_generation_control_mode_requires_generated_rgba_control(self):
        with self.assertRaisesRegex(ValueError, "requires rgba_generate_control"):
            DatasetConfig(
                folder_path="unused",
                pixel_channels="rgba",
                rgba_generate_control=False,
                rgba_control_mode="generation",
            )

    def test_rgba_dataset_rejects_rgb_only_files(self):
        with self.assertRaisesRegex(ValueError, "alpha channel"):
            prepare_rgba_image(Image.new("RGB", (2, 2)), require_alpha=True)


if __name__ == "__main__":
    unittest.main()
