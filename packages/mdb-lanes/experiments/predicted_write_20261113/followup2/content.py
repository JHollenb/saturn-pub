"""Content measures for rain and snow on decoded 256 px images (numpy + PIL only; no model).

Definitions (fixed before PREREG-ADDENDUM-2 was sealed; validated on native / deleted / sham first,
see ``validate`` and ``results/content-validation.json``):

* ``rain_streak(img)``: vertical-streak energy. Y = Rec.601 luminance in [0, 1]. Horizontal
  high-pass HP = Y - (Y box-averaged over 7 px along x). Vertical coherence V = HP box-averaged
  over 15 px along y (rain streaks are thin, near-vertical and long, so they survive; isolated
  texture averages out). Score = root mean square of V over the whole image.
* ``snow_ground(img)``: fraction of pixels in the bottom 35% of the image (rows >= 0.65 H) that are
  bright and not strongly coloured: HSV value max(R,G,B) >= 0.60 and HSV saturation <= 0.50. Snow
  under the prompt's blue dawn light is light blue-white; the deleted image's ground is dark grey;
  the black cat is dark and the red ball is saturated, so neither counts.

``removed(c_arm, c_ref, c_del) = (c_ref - c_arm) / (c_ref - c_del)``: 0 = as the arm's reference
(iso for head cuts, native for value swaps), 1 = as the deleted image.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

STREAK_X, STREAK_Y = 7, 15
GROUND_FROM, SNOW_V, SNOW_S = 0.65, 0.60, 0.50


def rgb(path_or_image):
    im = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    return np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0


def _box(x, n, axis):
    k = np.ones(n) / n
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), axis, x)


def rain_streak(img):
    a = rgb(img)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    hp = y - _box(y, STREAK_X, axis=1)
    v = _box(hp, STREAK_Y, axis=0)
    m = STREAK_Y                                   # drop the zero-padded border rows/cols
    return float(np.sqrt(np.mean(v[m:-m, m:-m] ** 2)))


def snow_ground(img):
    a = rgb(img)
    h = a.shape[0]
    g = a[int(round(GROUND_FROM * h)):]
    mx, mn = g.max(-1), g.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-12), 0.0)
    return float(np.mean((mx >= SNOW_V) & (sat <= SNOW_S)))


MEASURES = {"rain": rain_streak, "snow": snow_ground}


def removed(c_arm, c_ref, c_del):
    span = c_ref - c_del
    return float((c_ref - c_arm) / span) if span != 0 else float("nan")


def rain_coherence(img):
    """Secondary rain measure, contrast-normalised: RMS of V over RMS of HP (same V, HP as rain_streak).

    The share of the horizontal high-pass energy that is vertically coherent. Unlike rain_streak it
    does not fall when the whole frame loses contrast (the audit found the rain value swap crushes
    contrast); it is noisier (see results/content-validation.json).
    """
    a = rgb(img)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    hp = y - _box(y, STREAK_X, axis=1)
    v = _box(hp, STREAK_Y, axis=0)
    m = STREAK_Y
    return float(np.sqrt(np.mean(v[m:-m, m:-m] ** 2)) / np.sqrt(np.mean(hp[m:-m, m:-m] ** 2)))
