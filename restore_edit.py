import numpy as np
from scipy import ndimage

DIFFERENCE_THRESHOLD = 0.05
DIFFERENCE_SOFTNESS = 0.02
DETECTOR_FLOOR = 0.15
MASK_GROW = 12
MASK_FEATHER = 10
CORE_THRESHOLD = 0.5
OPEN_RADIUS = 6
MIN_AREA_SHARE = 0.002
NEAR_PX = 60
DIFFERENCE_WINDOW = 5
TONE_RATIO_MIN = 0.80
TONE_RATIO_MAX = 1.25
MIN_DEVIATION = 1e-6
TONE_REFERENCE_GROW = MASK_GROW * 3


def _dilate(mask, amount):
    if amount <= 0:
        return mask
    return ndimage.maximum_filter(mask, size=amount * 2 + 1, mode="nearest")


def _blur(mask, radius):
    if radius <= 0:
        return mask
    sigma = max(radius / 3.0, 0.5)
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    out = ndimage.convolve1d(mask, kernel, axis=1, mode="nearest")
    return ndimage.convolve1d(out, kernel, axis=0, mode="nearest")


def _change_mask(generated, original):
    mean_g, mean_o = generated.mean((0, 1)), original.mean((0, 1))
    std_g = np.maximum(generated.std((0, 1)), MIN_DEVIATION)
    std_o = np.maximum(original.std((0, 1)), MIN_DEVIATION)
    detector = np.clip((generated - mean_g) * np.clip(std_o / std_g, TONE_RATIO_MIN, TONE_RATIO_MAX) + mean_o, 0.0, 1.0)
    diff = ndimage.uniform_filter(np.abs(detector - original).mean(-1), size=DIFFERENCE_WINDOW, mode="nearest")
    auto = 1.0 / (1.0 + np.exp(-(diff - DIFFERENCE_THRESHOLD) / DIFFERENCE_SOFTNESS))
    return np.clip((auto - DETECTOR_FLOOR) / (1.0 - DETECTOR_FLOOR), 0.0, 1.0)


def _solid_changes(auto):
    hard = ndimage.binary_opening(auto > CORE_THRESHOLD, structure=np.ones((OPEN_RADIUS * 2 + 1,) * 2))
    labels, count = ndimage.label(hard)
    if count == 0:
        return hard.astype(np.float32)
    sizes = ndimage.sum(hard, labels, range(1, count + 1))
    kept = np.isin(labels, np.where(sizes >= MIN_AREA_SHARE * hard.size)[0] + 1)
    return ndimage.binary_fill_holes(kept).astype(np.float32)


def _tone_match(generated, original, untouched):
    weight = untouched[..., None]
    count = max(float(weight.sum()), 1.0)
    mean_g = (generated * weight).sum((0, 1)) / count
    mean_o = (original * weight).sum((0, 1)) / count
    std_g = np.sqrt(np.maximum((((generated - mean_g) ** 2) * weight).sum((0, 1)) / count, MIN_DEVIATION))
    std_o = np.sqrt(np.maximum((((original - mean_o) ** 2) * weight).sum((0, 1)) / count, MIN_DEVIATION))
    return np.clip((generated - mean_g) * np.clip(std_o / std_g, TONE_RATIO_MIN, TONE_RATIO_MAX) + mean_o, 0.0, 1.0)


def restore_edit(generated, original, user_mask):
    if user_mask.any():
        tone_reference = _dilate(user_mask, TONE_REFERENCE_GROW) < 0.5
    else:
        tone_reference = np.ones(generated.shape[:2], dtype=bool)

    generated = _tone_match(generated, original, tone_reference.astype(np.float32))
    auto = _change_mask(generated, original)
    core = np.maximum(user_mask, _solid_changes(auto))
    near = ndimage.binary_dilation(core > 0.5, structure=np.ones((3, 3)), iterations=NEAR_PX).astype(np.float32)
    region = np.maximum(core, auto * near)
    matched = _tone_match(generated, original, 1.0 - _dilate(region, TONE_REFERENCE_GROW))
    alpha = np.clip(_blur(_dilate(region, MASK_GROW), MASK_FEATHER), 0.0, 1.0)[..., None]
    return matched * alpha + original * (1.0 - alpha)
