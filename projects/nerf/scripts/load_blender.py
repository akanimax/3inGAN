import os
import numpy as np
import imageio
import json
import torch
from torch.nn.functional import avg_pool2d

trans_t = lambda t: torch.from_numpy(
    np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
)

rot_phi = lambda phi: torch.from_numpy(
    np.array(
        [
            [1, 0, 0, 0],
            [0, torch.cos(phi), -torch.sin(phi), 0],
            [0, torch.sin(phi), torch.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
)

rot_theta = lambda th: torch.from_numpy(
    np.array(
        [
            [torch.cos(th), 0, -torch.sin(th), 0],
            [0, 1, 0, 0],
            [torch.sin(th), 0, torch.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
        imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    if half_res:
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        imgs = avg_pool2d(imgs, (2, 2), stride=2).permute(0, 2, 3, 1).numpy()
        H = H // 2
        W = W // 2
        focal = focal / 2.0

    return imgs, poses, render_poses, [H, W, focal], i_split
