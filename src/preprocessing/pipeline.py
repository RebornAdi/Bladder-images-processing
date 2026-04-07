"""
src/preprocessing/pipeline.py  (UPDATED — v2)
===============================================
Changes from v1:
  1. normalize() uses "percentile_clip" by default — fixes dark normalized image
  2. visualize_pipeline() uses cv2.NORM_MINMAX for PNG — correct for all methods
  3. GEXDclearLayout defaults: bottom=280, right=220 (removes overlay + sidebar)
"""

import cv2
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger("preprocess")


@dataclass
class ImageMetadata:
    original_filename:   str
    original_shape:      tuple
    crop_shape:          tuple
    processed_shape:     tuple
    panel_side:          str
    label:               str
    pixel_spacing_mm:    float
    original_spacing_mm: float
    normalization:       str
    mean_intensity:      float
    std_intensity:       float
    percentile_low_val:  float = 0.0
    percentile_high_val: float = 0.0


class GEXDclearLayout:
    """
    Crop offsets for GE XDclear screenshots.

    Tuning guide — re-run visualize_pipeline() after each change:
      "L x.xx mm" still visible   →  increase bottom (try +20 at a time)
      "ADM"/"Direct Rep." visible →  increase right  (try +20 at a time)
      Top bar still showing       →  increase top    (try +10 at a time)
      Scan content being cut off  →  decrease that value
    """
    HEADER_TOP    = 80
    FOOTER_BOTTOM = 280    # increased from 210
    RIGHT_SIDEBAR = 220    # increased from 150
    LEFT_MENU     = 0

    @classmethod
    def has_left_menu(cls, img: np.ndarray) -> bool:
        return float(img[:, :180].mean()) > 30


class UltrasoundPreprocessor:

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.pre_cfg = self.cfg["preprocessing"]
        self.target_h, self.target_w = self.pre_cfg["target_size"]
        self.default_spacing = self.cfg["pixel_spacing"]["default_mm_per_pixel"]

    def load(self, path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot load: {path}")
        return img

    def crop_ge_ui(self, img: np.ndarray, left_menu: bool = False) -> np.ndarray:
        c = self.pre_cfg["crop"]
        h, w = img.shape[:2]
        top    = c["top"]
        bottom = max(h - c["bottom"], top + 100)
        left   = 180 if left_menu else c["left"]
        right  = max(w - c["right"], left + 100)
        return img[top:bottom, left:right]

    def split_panels(self, img: np.ndarray, keep: str = "both") -> dict:
        h, w = img.shape[:2]
        mid = w // 2
        panels = {"left": img[:, :mid], "right": img[:, mid:]}
        return panels if keep == "both" else {keep: panels[keep]}

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (self.target_w, self.target_h),
                          interpolation=cv2.INTER_AREA)

    def denoise(self, img: np.ndarray) -> np.ndarray:
        d = self.pre_cfg["denoise"]
        if d["method"] == "bilateral":
            return cv2.bilateralFilter(img, d=d["bilateral_d"],
                sigmaColor=d["bilateral_sigma_color"],
                sigmaSpace=d["bilateral_sigma_space"])
        return cv2.medianBlur(img, d["median_ksize"])

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        c = self.pre_cfg["clahe"]
        clahe = cv2.createCLAHE(clipLimit=c["clip_limit"],
                                 tileGridSize=tuple(c["tile_grid"]))
        return clahe.apply(img)

    def normalize(self, img: np.ndarray) -> tuple:
        """
        Normalize pixel intensities for DL training.

        RECOMMENDED: "percentile_clip"
          Clips bottom p_low% and top p_high% of pixels, then rescales to [0,1].
          Why: ultrasound has a large dark background (very low values) and
          bright specular reflections (very high values). These outliers collapse
          the useful intensity range when using simple minmax or zscore.
          Clipping them first expands the contrast in the diagnostically
          important scan region (bladder wall, prostate boundary).

        "minmax": divide by 255 — simple, scan must be well-exposed
        "zscore": mean-zero unit-variance — correct for DL but PNG looks dark

        Returns: (normalized_float32, mean, std, p_low_value, p_high_value)
        """
        img_f = img.astype(np.float32)
        mean  = float(img_f.mean())
        std   = float(img_f.std()) + 1e-8
        p_low_val = p_high_val = 0.0

        method = self.pre_cfg.get("normalization", "percentile_clip")

        if method == "percentile_clip":
            p_low  = self.pre_cfg.get("percentile_low",  1.0)
            p_high = self.pre_cfg.get("percentile_high", 99.0)
            p_low_val  = float(np.percentile(img_f, p_low))
            p_high_val = float(np.percentile(img_f, p_high))
            img_c = np.clip(img_f, p_low_val, p_high_val)
            normalized = (img_c - p_low_val) / ((p_high_val - p_low_val) + 1e-8)

        elif method == "minmax":
            normalized = img_f / 255.0

        elif method == "zscore":
            normalized = (img_f - mean) / std

        else:
            raise ValueError(f"Unknown normalization: {method}")

        return normalized.astype(np.float32), mean, std, p_low_val, p_high_val

    @staticmethod
    def _to_uint8_vis(normalized: np.ndarray) -> np.ndarray:
        """Convert any float32 (incl. negatives from zscore) to uint8 for PNG."""
        return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def compute_pixel_spacing(self, original_shape, crop_shape):
        orig = self.default_spacing
        return orig * (crop_shape[0] / self.target_h), orig

    def process_image(self, input_path, output_dir, label,
                      panel="both", save=True) -> list:
        input_path = Path(input_path)
        output_dir = Path(output_dir) / label
        output_dir.mkdir(parents=True, exist_ok=True)

        raw            = self.load(input_path)
        original_shape = raw.shape[:2]
        has_menu       = GEXDclearLayout.has_left_menu(raw)
        cropped        = self.crop_ge_ui(raw, left_menu=has_menu)
        crop_shape     = cropped.shape[:2]
        panels         = self.split_panels(cropped, keep=panel)

        results = []
        for side, panel_img in panels.items():
            gray      = self.to_grayscale(panel_img)
            resized   = self.resize(gray)
            denoised  = self.denoise(resized)
            enhanced  = self.enhance_contrast(denoised)
            normalized, mean_i, std_i, p_low, p_high = self.normalize(enhanced)
            new_sp, orig_sp = self.compute_pixel_spacing(original_shape, crop_shape)

            stem = input_path.stem
            meta = ImageMetadata(
                original_filename   = input_path.name,
                original_shape      = original_shape,
                crop_shape          = crop_shape,
                processed_shape     = (self.target_h, self.target_w),
                panel_side          = side,
                label               = label,
                pixel_spacing_mm    = round(new_sp, 6),
                original_spacing_mm = round(orig_sp, 6),
                normalization       = self.pre_cfg.get("normalization", "percentile_clip"),
                mean_intensity      = round(mean_i, 4),
                std_intensity       = round(std_i, 4),
                percentile_low_val  = round(p_low, 4),
                percentile_high_val = round(p_high, 4),
            )
            result = {"image": normalized, "metadata": meta, "saved_path": None}

            if save:
                npy_path = output_dir / f"{stem}_{side.upper()}.npy"
                np.save(str(npy_path), normalized)

                vis_uint8 = self._to_uint8_vis(normalized)
                cv2.imwrite(str(output_dir / f"{stem}_{side.upper()}_vis.png"), vis_uint8)

                with open(output_dir / f"{stem}_{side.upper()}_meta.json", "w") as f:
                    json.dump(asdict(meta), f, indent=2)

                result["saved_path"] = npy_path
                log.info(f"Saved: {npy_path.name}")

            results.append(result)
        return results

    def batch_process(self, raw_dir, output_dir, label,
                      extensions=(".jpg", ".jpeg", ".png"),
                      panel="both") -> list:
        raw_dir = Path(raw_dir)
        files   = sorted([f for f in raw_dir.iterdir()
                          if f.suffix.lower() in extensions])
        log.info(f"[{label}] Found {len(files)} images in {raw_dir}")
        all_paths, failed = [], []
        for idx, p in enumerate(files):
            try:
                res = self.process_image(p, output_dir, label, panel=panel, save=True)
                all_paths.extend([r["saved_path"] for r in res if r["saved_path"]])
                if (idx + 1) % 20 == 0:
                    log.info(f"  Progress: {idx+1}/{len(files)}")
            except Exception as e:
                log.error(f"  FAILED {p.name}: {e}")
                failed.append(p.name)
        log.info(f"[{label}] Done: {len(all_paths)} saved, {len(failed)} failed")
        return all_paths

    def visualize_pipeline(self, input_path, output_path):
        """5-panel strip: Raw → Cropped → Denoised → CLAHE → Normalized"""
        input_path = Path(input_path)
        raw      = self.load(input_path)
        has_menu = GEXDclearLayout.has_left_menu(raw)
        cropped  = self.crop_ge_ui(raw, left_menu=has_menu)
        panel_l  = self.split_panels(cropped, keep="left")["left"]

        gray      = self.to_grayscale(panel_l)
        resized   = self.resize(gray)
        denoised  = self.denoise(resized)
        enhanced  = self.enhance_contrast(denoised)
        norm, _, _, _, _ = self.normalize(enhanced)
        normed_vis = self._to_uint8_vis(norm)    # always correct for PNG

        stages = [
            ("Raw",        cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), (256, 256))),
            ("Cropped",    cv2.resize(gray,       (256, 256))),
            ("Denoised",   cv2.resize(denoised,   (256, 256))),
            ("CLAHE",      cv2.resize(enhanced,   (256, 256))),
            ("Normalized", cv2.resize(normed_vis, (256, 256))),
        ]
        labeled = []
        for title, img in stages:
            h,w=img.shape
            canvas = np.zeros((h+30, w), dtype=np.uint8)
            canvas[30:30+h] = img
            cv2.putText(canvas, title, (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, 255, 1)
            labeled.append(canvas)
        strip = np.concatenate(labeled, axis=1)
        cv2.imwrite(str(output_path), strip)
        log.info(f"Pipeline visualization saved: {output_path}")
        return strip


if __name__ == "__main__":
    p = UltrasoundPreprocessor("configs/config.yaml")
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    for cat in ["ipp", "dwt", "mixed", "volume"]:
        raw = Path(cfg["paths"]["raw_data"]) / cat
        if raw.exists():
            p.batch_process(raw, cfg["paths"]["processed"], label=cat)