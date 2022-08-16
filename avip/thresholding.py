from math import floor, ceil

import numpy as np
from PIL import Image
from tqdm import trange


def balanced_thresholding(grayscale: Image.Image):
    np_img = np.asarray(grayscale, dtype=np.uint8)
    min_val = np_img.min()
    max_val = np_img.max() + 1
    hist, edges = np.histogram(np_img, density=True, bins=np.arange(min_val, max_val))
    left = 0
    right = len(hist) - 1
    while left < right:
        left_middle = floor((left + right) / 2)
        right_middle = ceil((left + right) / 2)
        left_sum = np.sum(hist[left:left_middle])
        right_sum = np.sum(hist[right_middle:right])
        if left_sum > right_sum:
            left += 1
        else:
            right -= 1
    binary = np.zeros_like(grayscale)
    binary[grayscale > edges[left]] = 255
    return Image.fromarray(binary)


def _otsu_convertation(p0, p1, i0, i1, w0, w1):
    m0 = np.sum(p0 * i0 / w0)
    m1 = np.sum(p1 * i1 / w1)
    d0 = np.sum(p0 * (i0 - m0) ** 2)
    d1 = np.sum(p1 * (i1 - m1) ** 2)
    d_intra = w0 * d0 + w1 * d1
    d_inter = w0 * w1 * (m0 - m1) ** 2
    return d_inter / d_intra


def transform_otsu(grayscale: Image.Image) -> Image.Image:
    np_img = np.asarray(grayscale, dtype=np.uint8)
    min_val = np_img.min()
    max_val = np_img.max() + 1
    hist, edges = np.histogram(np_img, density=True, bins=np.arange(min_val, max_val))

    w0 = np.cumsum(hist)
    w1 = np.ones_like(w0) - w0
    results = []
    for i in trange(len(w0)):
        if w0[i] < 0.01 or w1[i] < 0.01:
            continue
        result = _otsu_convertation(
            p0=hist[:i], i0=edges[1:i + 1], w0=w0[i],
            p1=hist[i:], i1=edges[i + 1:], w1=w1[i],
        )
        results.append(result)
    binary = np.zeros_like(grayscale)
    binary[grayscale > edges[results.index(max(results)) + 1]] = 255
    return Image.fromarray(binary)
