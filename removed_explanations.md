# Explanations Removed from the Cleaned Code

The cleaned `.py` file keeps the original executable statements and processing logic. This file collects the longer explanations removed from comments/docstrings to reduce clutter. Short operational comments and section labels remain in the code.

## 1. Calibration improvements

**Source:** Large `DROP-IN CALIBRATION PATCH` comment immediately before `_detect_ruler_tick_spacing()` and `get_pixel_spacing_auto()`.

- **Grid fit:** Estimate tick spacing from ticks that align with a regular grid using least squares, reducing sensitivity to missing/spurious ticks and half/double-spacing errors.
- **Both edges:** Detect ruler ticks on the left and right edges. If their scales agree within 5%, confidence is higher; if not, prefer the edge with stronger grid support and warn.
- **Depth sanity check:** Reject detected scales whose implied scan depth is outside the configured 4–30 cm range.
- **Fallbacks:** Keep OCR of the `D ##.#` depth field and the 14 cm default as fallback paths.

## 2. Tick-spacing detector

**Source:** Docstring of `_detect_ruler_tick_spacing()`.

Detects the pixel spacing between minor depth-scale ticks on either image edge without OCR. Returns the spacing, detected tick centers, and a support count used as a confidence signal. Returns no spacing when it cannot find a consistent set.

## 3. Automatic pixel calibration

**Source:** Docstring of `get_pixel_spacing_auto()`.

Calibration priority:
1. Use `depth_cm` directly when provided.
2. Detect and cross-check ruler ticks on both edges, with a plausible-depth check.
3. Fall back to OCR of the depth field.
4. If OCR also fails, use the 14 cm default and warn.

The `minor_tick_cm` parameter describes the real-world spacing between minor ruler ticks. `plausible_depth_cm` sets the acceptable implied scan-depth range.

## 4. Manual calibration

**Source:** Docstring of `manual_scale_calibration()`.

Allows the user to click two ruler ticks separated by a known distance. The pixel distance between them is used to calculate mm-per-pixel, avoiding reliance on OCR. Press a key to confirm or Esc to cancel.

## 5. Seed selection and level-set helpers

**Source:** Docstring of `_mask_level_set()`.

Builds a signed-distance level set from a binary mask. It is used when the initial ellipse has been clipped by the detected boundary, so the starting level set is no longer a pure analytic ellipse.

## 6. Chan–Vese segmentation

**Source:** Long `THE 3-STAGE PLAN` comment above `segment_bladder_full()`.

- **Stage 1:** Run the existing threshold sweep and region-growing method (`segment_bladder`) to detect a rough bladder-wall boundary.
- **Stage 2:** Fit an ellipse to that boundary and intersect the ellipse with the boundary mask so it cannot expand beyond the detected region. Keep `ellipse_scale` near 1.0 for modest padding.
- **Stage 3:** Run Chan–Vese refinement inside the bounded ellipse.

**Source:** Long comment above the CLAHE setup in `segment_bladder()`.

CLAHE is applied only to the threshold-sweep image. Background exclusion deliberately uses the original crop because CLAHE can brighten dark outside-sector corners and interfere with the near-black background assumption. The comment records that CLAHE reduced leakage on blurry images during validation.

## 7. Measurement distances

**Source:** Docstring of `max_diagonal_distance()`.

The diagonal measurement is constrained to a top-left-to-bottom-right pair because that matches the clinical caliper direction. It is not the unrestricted longest distance, which could select the opposite diagonal or a near-horizontal pair. If no valid oriented pair is found, the function falls back to the unrestricted longest pair.

**Source:** Docstring of `panel_axes()`.

Retained for compatibility and debug visuals. Returns the major Feret diameter, perpendicular caliper width, and the selected endpoint pair.

## 8. Image input modes

**Source:** Usage comments in the `__main__` section.

The script supports either a pre-cropped ROI image or a full image plus four crop coordinates. An optional panel label can override which panel supplies the diagonal/depth measurement.
