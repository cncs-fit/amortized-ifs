"""Minimal cv2 stub for Tu et al. public code paths unused in MNIST fitting."""

COLOR_HSV2RGB = 0


def cvtColor(src, code, dst=None):
    if dst is not None:
        dst[...] = src
        return dst
    return src
