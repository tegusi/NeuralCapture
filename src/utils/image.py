import numpy as np
import torch
from piq import ssim as ss


def tonemapping(img, method='gamma', gamma=2.2):
    r"""tonemapping and HDR image

    Args:
        img: the HRD image
        method: `'reinhard'` or `'gamma'`, default `'gamma'`
        gamma: gamma values
    Returns:
        tonemapped image
    """
    if method == 'reinhard':
        import cv2
        tonemapper = cv2.createTonemapReinhard(1, 1, 0, 0)
        img = tonemapper.process(img)
    elif method == 'gamma':
        img = (img / img.max()) ** (1 / gamma)
    else:
        raise ValueError(method)

    # Clip, if necessary, to guard against numerical errors
    minv, maxv = img.min(), img.max()
    if minv < 0:
        img = np.clip(img, 0, np.inf)
    if maxv > 1:
        img = np.clip(img, -np.inf, 1)
    return img


def psnr(x):
    if isinstance(x, float):
        return -10. * np.log(x) / np.log(10.)
    elif isinstance(x, torch.Tensor):
        return -10. * torch.log(torch.mean(x)) / torch.log(torch.Tensor([10.]))
    elif isinstance(x, np.ndarray):
        return -10. * np.log(np.mean(x) / np.log(10.))

def ssim(x, y):
    return ss(x.unsqueeze(0), y.unsqueeze(0), data_range=1.)


def bbox_from_mask(arr):
    arr = arr.astype(np.int32)
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def float2uint8(src):
    return (255*np.clip(src, 0, 1)).astype(np.uint8)


def rgb2yuv(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.stack([y, u, v], dim=-1)


def yuv2rgb(x):
    y, u, v = x[..., 0], x[..., 1], x[..., 2]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.3011 * u
    out = torch.stack([r, g, b], dim=-1)
    return torch.clamp(out, 0.0, 1.0)


def whc2cwh(x):
    if x.dim() < 3:
        x = x.unsqueeze(-1)
    return x.permute(2, 0, 1)

def cwh2whc(x):
    if x.dim() < 3:
        x = x.unsqueeze(-1)
    return x.permute(1, 2, 0)
