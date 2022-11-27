# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch.nn.functional import pad
from torchvision.transforms import GaussianBlur
from thre3d_atom.training.augmentation import (
    grid_sample_gradfix,
    upfirdn2d,
)
from thre3d_atom.training.augmentation.utils import constant


# fmt: off
# ----------------------------------------------------------------------------
# Helpers for constructing transformation matrices.


def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else constant(x, shape=ref[0].shape, device=ref[0].device) for x in
             elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))


def translate2d(tx, ty, **kwargs):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
        **kwargs)


def translate3d(tx, ty, tz, **kwargs):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
        **kwargs)


def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1],
        **kwargs)


def scale3d(sx, sy, sz, **kwargs):
    return matrix(
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1],
        **kwargs)


def rotate2d(theta, **kwargs):
    return matrix(
        [torch.cos(theta), torch.sin(-theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1],
        **kwargs)


def rotate3d(v, theta, **kwargs):
    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]
    s = torch.sin(theta)
    c = torch.cos(theta)
    cc = 1 - c
    return matrix(
        [vx * vx * cc + c, vx * vy * cc - vz * s, vx * vz * cc + vy * s, 0],
        [vy * vx * cc + vz * s, vy * vy * cc + c, vy * vz * cc - vx * s, 0],
        [vz * vx * cc - vy * s, vz * vy * cc + vx * s, vz * vz * cc + c, 0],
        [0, 0, 0, 1],
        **kwargs)


def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)


def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)


def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)


# ----------------------------------------------------------------------------
# Versatile image augmentation pipeline from the paper
# "Training Generative Adversarial Networks with Limited Data".
#
# All augmentations are disabled by default; individual augmentations can
# be enabled by setting their probability multipliers to 1.

class AugmentPipe(torch.nn.Module):
    def __init__(self,
                 xflip=0, rotate90=0, xint=0, xint_max=0.125,
                 scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=1, aniso_std=0.2, xfrac_std=0.125,
                 blur=0, blur_size=3, blur_sigma=1.0,
                 brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0, brightness_std=0.2, contrast_std=0.5,
                 hue_max=1, saturation_std=1,
                 noise=0, cutout=0, noise_std=0.1, cutout_size=0.5,
                 ):
        super().__init__()
        self.register_buffer('p', torch.ones([]))  # Overall multiplier for augmentation probability.

        # Pixel blitting.
        self.xflip = float(xflip)  # Probability multiplier for x-flip.
        self.rotate90 = float(rotate90)  # Probability multiplier for 90 degree rotations.
        self.xint = float(xint)  # Probability multiplier for integer translation.
        self.xint_max = float(xint_max)  # Range of integer translation, relative to image dimensions.

        # General geometric transformations.
        self.scale = float(scale)  # Probability multiplier for isotropic scaling.
        self.rotate = float(rotate)  # Probability multiplier for arbitrary rotation.
        self.aniso = float(aniso)  # Probability multiplier for anisotropic scaling.
        self.xfrac = float(xfrac)  # Probability multiplier for fractional translation.
        self.scale_std = float(scale_std)  # Log2 standard deviation of isotropic scaling.
        self.rotate_max = float(rotate_max)  # Range of arbitrary rotation, 1 = full circle.
        self.aniso_std = float(aniso_std)  # Log2 standard deviation of anisotropic scaling.
        self.xfrac_std = float(xfrac_std)  # Standard deviation of frational translation, relative to image dimensions.

        # Image blur parameters
        self.blur = float(blur)
        self.blur_size = int(blur_size)
        self.blur_sigma = blur_sigma if isinstance(blur_sigma, tuple) else float(blur_sigma)

        # Color transformations.
        self.brightness = float(brightness)  # Probability multiplier for brightness.
        self.contrast = float(contrast)  # Probability multiplier for contrast.
        self.lumaflip = float(lumaflip)  # Probability multiplier for luma flip.
        self.hue = float(hue)  # Probability multiplier for hue rotation.
        self.saturation = float(saturation)  # Probability multiplier for saturation.
        self.brightness_std = float(brightness_std)  # Standard deviation of brightness.
        self.contrast_std = float(contrast_std)  # Log2 standard deviation of contrast.
        self.hue_max = float(hue_max)  # Range of hue rotation, 1 = full circle.
        self.saturation_std = float(saturation_std)  # Log2 standard deviation of saturation.

        # Image-space corruptions.
        self.noise = float(noise)  # Probability multiplier for additive RGB noise.
        self.cutout = float(cutout)  # Probability multiplier for cutout.
        self.noise_std = float(noise_std)  # Standard deviation of additive RGB noise.
        self.cutout_size = float(cutout_size)  # Size of the cutout rectangle, relative to image dimensions.

        # Setup orthogonal lowpass filter for geometric augmentations.
        self.register_buffer('Hz_geom', upfirdn2d.setup_filter(
            [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466,
             0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578,
             0.0017677118642428036, -0.007800708325034148]))

        # create a blurring operation object
        self.blurer = GaussianBlur(kernel_size=self.blur_size, sigma=self.blur_sigma)

    def forward(self, images, debug_percentile=None):
        assert isinstance(images, torch.Tensor) and images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        device = images.device
        if debug_percentile is not None:
            debug_percentile = torch.as_tensor(debug_percentile, dtype=torch.float32, device=device)

        # -------------------------------------
        # Select parameters for pixel blitting.
        # -------------------------------------

        # Initialize inverse homogeneous 2D transform: G_inv @ pixel_out ==> pixel_in
        I_3 = torch.eye(3, device=device)
        G_inv = I_3

        # Apply x-flip with probability (xflip * strength).
        if self.xflip > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 2)
            i = torch.where(torch.rand([batch_size], device=device) < self.xflip * self.p, i, torch.zeros_like(i))
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 2))
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)

        # Apply 90 degree rotations with probability (rotate90 * strength).
        if self.rotate90 > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 4)
            i = torch.where(torch.rand([batch_size], device=device) < self.rotate90 * self.p, i, torch.zeros_like(i))
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 4))
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)

        # Apply integer translation with probability (xint * strength).
        if self.xint > 0:
            t = (torch.rand([batch_size, 2], device=device) * 2 - 1) * self.xint_max
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xint * self.p, t, torch.zeros_like(t))
            if debug_percentile is not None:
                t = torch.full_like(t, (debug_percentile * 2 - 1) * self.xint_max)
            G_inv = G_inv @ translate2d_inv(torch.round(t[:, 0] * width), torch.round(t[:, 1] * height))

        # --------------------------------------------------------
        # Select parameters for general geometric transformations.
        # --------------------------------------------------------

        # Apply isotropic scaling with probability (scale * strength).
        if self.scale > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.scale_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.scale * self.p, s, torch.ones_like(s))
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.scale_std))
            G_inv = G_inv @ scale2d_inv(s, s)

        # Apply pre-rotation with probability p_rot.
        p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1))  # P(pre OR post) = p
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            if debug_percentile is not None:
                theta = torch.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.rotate_max)
            G_inv = G_inv @ rotate2d_inv(-theta)  # Before anisotropic scaling.

        # Apply anisotropic scaling with probability (aniso * strength).
        if self.aniso > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.aniso_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.aniso * self.p, s, torch.ones_like(s))
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.aniso_std))
            G_inv = G_inv @ scale2d_inv(s, 1 / s)

        # Apply post-rotation with probability p_rot.
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            if debug_percentile is not None:
                theta = torch.zeros_like(theta)
            G_inv = G_inv @ rotate2d_inv(-theta)  # After anisotropic scaling.

        # Apply fractional translation with probability (xfrac * strength).
        if self.xfrac > 0:
            t = torch.randn([batch_size, 2], device=device) * self.xfrac_std
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xfrac * self.p, t, torch.zeros_like(t))
            if debug_percentile is not None:
                t = torch.full_like(t, torch.erfinv(debug_percentile * 2 - 1) * self.xfrac_std)
            G_inv = G_inv @ translate2d_inv(t[:, 0] * width, t[:, 1] * height)

        # ----------------------------------
        # Execute geometric transformations.
        # ----------------------------------

        # Execute if the transform is not identity.
        if G_inv is not I_3:
            # Calculate padding.
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=device)  # [idx, xyz]
            cp = G_inv @ cp.t()  # [batch, xyz, idx]
            Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1)  # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values  # [x0, y0, x1, y1]
            margin = margin + constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
            margin = margin.max(constant([0, 0] * 2, device=device))
            margin = margin.min(constant([width - 1, height - 1] * 2, device=device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = pad(input=images, pad=[mx0, mx1, my0, my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            images = upfirdn2d.upsample2d(x=images, f=self.Hz_geom, up=2)
            G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
            G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)

            # Execute transformation.
            shape = [batch_size, num_channels, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale2d_inv(2 / shape[3],
                                                                                                           2 / shape[2],
                                                                                                           device=device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:, :2, :], size=shape, align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid)

            # Downsample and crop.
            images = upfirdn2d.downsample2d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad * 2, flip_filter=True)

        # --------------------------------------------
        # Select parameters for color transformations.
        # --------------------------------------------

        # Initialize homogeneous 3D transformation matrix: C @ color_in ==> color_out
        I_4 = torch.eye(4, device=device)
        C = I_4

        # Apply brightness with probability (brightness * strength).
        if self.brightness > 0:
            b = torch.randn([batch_size], device=device) * self.brightness_std
            b = torch.where(torch.rand([batch_size], device=device) < self.brightness * self.p, b, torch.zeros_like(b))
            if debug_percentile is not None:
                b = torch.full_like(b, torch.erfinv(debug_percentile * 2 - 1) * self.brightness_std)
            C = translate3d(b, b, b) @ C

        # Apply contrast with probability (contrast * strength).
        if self.contrast > 0:
            c = torch.exp2(torch.randn([batch_size], device=device) * self.contrast_std)
            c = torch.where(torch.rand([batch_size], device=device) < self.contrast * self.p, c, torch.ones_like(c))
            if debug_percentile is not None:
                c = torch.full_like(c, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.contrast_std))
            C = scale3d(c, c, c) @ C

        # Apply luma flip with probability (lumaflip * strength).
        v = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=device)  # Luma axis.
        if self.lumaflip > 0:
            i = torch.floor(torch.rand([batch_size, 1, 1], device=device) * 2)
            i = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.lumaflip * self.p, i,
                            torch.zeros_like(i))
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 2))
            C = (I_4 - 2 * v.ger(v) * i) @ C  # Householder reflection.

        # Apply hue rotation with probability (hue * strength).
        if self.hue > 0 and num_channels > 1:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.hue_max
            theta = torch.where(torch.rand([batch_size], device=device) < self.hue * self.p, theta,
                                torch.zeros_like(theta))
            if debug_percentile is not None:
                theta = torch.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.hue_max)
            C = rotate3d(v, theta) @ C  # Rotate around v.

        # Apply saturation with probability (saturation * strength).
        if self.saturation > 0 and num_channels > 1:
            s = torch.exp2(torch.randn([batch_size, 1, 1], device=device) * self.saturation_std)
            s = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.saturation * self.p, s,
                            torch.ones_like(s))
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.saturation_std))
            C = (v.ger(v) + (I_4 - v.ger(v)) * s) @ C

        # ------------------------------
        # Execute color transformations.
        # ------------------------------

        # Execute if the transform is not identity.
        if C is not I_4:
            images = images.reshape([batch_size, num_channels, height * width])
            if num_channels == 3:
                images = C[:, :3, :3] @ images + C[:, :3, 3:]
            elif num_channels == 1:
                C = C[:, :3, :].mean(dim=1, keepdims=True)
                images = images * C[:, :, :3].sum(dim=2, keepdims=True) + C[:, :, 3:]
            else:
                raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
            images = images.reshape([batch_size, num_channels, height, width])

        # ------------------------------
        # Execute blur transformation.
        # ------------------------------
        if self.blur > 0:
            images = torch.where(torch.rand([batch_size, 1, 1, 1], device=device) < self.blur * self.p,
                                 self.blurer(images), images)

        # ------------------------
        # Image-space corruptions.
        # ------------------------

        # Apply additive RGB noise with probability (noise * strength).
        if self.noise > 0:
            sigma = torch.randn([batch_size, 1, 1, 1], device=device).abs() * self.noise_std
            sigma = torch.where(torch.rand([batch_size, 1, 1, 1], device=device) < self.noise * self.p, sigma,
                                torch.zeros_like(sigma))
            if debug_percentile is not None:
                sigma = torch.full_like(sigma, torch.erfinv(debug_percentile) * self.noise_std)
            images = images + torch.randn([batch_size, num_channels, height, width], device=device) * sigma

        # Apply cutout with probability (cutout * strength).
        if self.cutout > 0:
            size = torch.full([batch_size, 2, 1, 1, 1], self.cutout_size, device=device)
            size = torch.where(torch.rand([batch_size, 1, 1, 1, 1], device=device) < self.cutout * self.p, size,
                               torch.zeros_like(size))
            center = torch.rand([batch_size, 2, 1, 1, 1], device=device)
            if debug_percentile is not None:
                size = torch.full_like(size, self.cutout_size)
                center = torch.full_like(center, debug_percentile)
            coord_x = torch.arange(width, device=device).reshape([1, 1, 1, -1])
            coord_y = torch.arange(height, device=device).reshape([1, 1, -1, 1])
            mask_x = (((coord_x + 0.5) / width - center[:, 0]).abs() >= size[:, 0] / 2)
            mask_y = (((coord_y + 0.5) / height - center[:, 1]).abs() >= size[:, 1] / 2)
            mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
            images = images * mask

        return images

# ----------------------------------------------------------------------------
# fmt: on
