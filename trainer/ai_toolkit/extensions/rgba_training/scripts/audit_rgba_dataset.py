from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from toolkit.rgba_utils import image_has_alpha


def main():
    parser = argparse.ArgumentParser(description="Audit alpha coverage and hidden RGB in an RGBA dataset")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--top", type=int, default=10, help="show files with the most green-dominant edge pixels")
    parser.add_argument(
        "--edge-width",
        type=float,
        default=3.0,
        help="foreground boundary width used to detect baked-in chroma spill",
    )
    args = parser.parse_args()

    paths = sorted(
        path for path in args.dataset.rglob("*")
        if path.suffix.lower() in {".png", ".webp"}
    )
    if not paths:
        raise SystemExit(f"No PNG/WebP images found under {args.dataset}")

    missing_alpha = 0
    total_pixels = hidden_pixels = partial_pixels = 0
    hidden_nonzero = hidden_green_dominant = 0
    suspicious_partial_green = 0
    suspicious_files = []
    boundary_pixels = suspicious_boundary_green = 0
    boundary_suspicious_files = []
    threshold_u8 = int(round(args.alpha_threshold * 255.0))

    for path in paths:
        with Image.open(path) as image:
            if not image_has_alpha(image):
                missing_alpha += 1
                continue
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]
        hidden = alpha <= threshold_u8
        partial = (alpha > threshold_u8) & (alpha < 255)
        foreground = alpha > threshold_u8
        total_pixels += alpha.size
        hidden_pixels += int(hidden.sum())
        partial_pixels += int(partial.sum())
        if hidden.any():
            hidden_rgb = rgb[hidden]
            hidden_rgb_i16 = hidden_rgb.astype(np.int16)
            hidden_nonzero += int((hidden_rgb.max(axis=1) > 0).sum())
            hidden_green_dominant += int(
                ((hidden_rgb_i16[:, 1] > hidden_rgb_i16[:, 0] + 20) &
                 (hidden_rgb_i16[:, 1] > hidden_rgb_i16[:, 2] + 20)).sum()
            )
        if partial.any():
            partial_rgb = rgb[partial]
            partial_rgb_i16 = partial_rgb.astype(np.int16)
            suspicious_mask = (
                (partial_rgb_i16[:, 1] > partial_rgb_i16[:, 0] + 40)
                & (partial_rgb_i16[:, 1] > partial_rgb_i16[:, 2] + 40)
            )
            suspicious_count = int(suspicious_mask.sum())
            suspicious_partial_green += suspicious_count
            if suspicious_count:
                suspicious_files.append((
                    suspicious_count / int(partial.sum()),
                    suspicious_count,
                    int(partial.sum()),
                    path,
                ))
        if foreground.any():
            distance_inside = distance_transform_edt(foreground)
            boundary = foreground & (distance_inside <= args.edge_width)
            boundary_rgb_i16 = rgb[boundary].astype(np.int16)
            boundary_suspicious_mask = (
                (boundary_rgb_i16[:, 1] > boundary_rgb_i16[:, 0] + 40)
                & (boundary_rgb_i16[:, 1] > boundary_rgb_i16[:, 2] + 40)
            )
            boundary_count = int(boundary.sum())
            boundary_suspicious_count = int(boundary_suspicious_mask.sum())
            boundary_pixels += boundary_count
            suspicious_boundary_green += boundary_suspicious_count
            if boundary_suspicious_count:
                boundary_suspicious_files.append((
                    boundary_suspicious_count / boundary_count,
                    boundary_suspicious_count,
                    boundary_count,
                    path,
                ))

    def percent(value, denominator):
        return 100.0 * value / max(1, denominator)

    print(f"Images: {len(paths)}")
    print(f"Missing alpha: {missing_alpha}")
    print(f"Fully transparent pixels: {hidden_pixels} ({percent(hidden_pixels, total_pixels):.2f}%)")
    print(f"Partially transparent pixels: {partial_pixels} ({percent(partial_pixels, total_pixels):.2f}%)")
    print(f"Hidden pixels with retained RGB: {hidden_nonzero} ({percent(hidden_nonzero, hidden_pixels):.2f}%)")
    print(
        "Hidden green-dominant pixels: "
        f"{hidden_green_dominant} ({percent(hidden_green_dominant, hidden_pixels):.2f}%)"
    )
    print(
        "Suspicious green among partial-alpha pixels: "
        f"{suspicious_partial_green} ({percent(suspicious_partial_green, partial_pixels):.2f}%)"
    )
    print(
        f"Suspicious green in foreground boundary (width {args.edge_width:g}): "
        f"{suspicious_boundary_green} ({percent(suspicious_boundary_green, boundary_pixels):.2f}%)"
    )
    if args.top > 0 and suspicious_files:
        print("Top files by suspicious partial-alpha green ratio:")
        for ratio, count, partial_count, path in sorted(suspicious_files, reverse=True)[:args.top]:
            print(f"  {ratio * 100:6.2f}% ({count}/{partial_count})  {path}")
    if args.top > 0 and boundary_suspicious_files:
        print("Top files by suspicious foreground-boundary green ratio:")
        for ratio, count, boundary_count, path in sorted(boundary_suspicious_files, reverse=True)[:args.top]:
            print(f"  {ratio * 100:6.2f}% ({count}/{boundary_count})  {path}")
    if missing_alpha:
        raise SystemExit("Audit failed: some files do not contain alpha")


if __name__ == "__main__":
    main()
