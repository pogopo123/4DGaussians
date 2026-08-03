"""Rotate a whole multipleview dataset so the world up axis is a coordinate axis.

COLMAP fixes the world frame from the first registered image, so an otherwise
correct reconstruction sits tilted: for backpack_frame0_v2 the real "up" was
(0.937, -0.289, 0.195), i.e. 21 deg off +x. This applies ONE rigid rotation R
consistently to every file that carries a pose, so the reconstruction is
unchanged up to a change of basis:

    points3D_multipleview.ply     X'      = R X          (normals rotated too)
    sparse_/images.bin            R_w2c'  = R_w2c R^T,  t unchanged
    poses_bounds_multipleview.npy regenerated from the rotated extrinsics
    sparse_/cameras.bin           copied verbatim (intrinsics are pose-free)
    camNN/                        symlinked to the source frames

Writing all of them together is the point: rotating only the point cloud would
silently break the reconstruction. Output goes to a NEW directory; the source
dataset is never modified.

Usage:
    python customer_scripts/upright_dataset.py <src_scene> [dst_scene] [--up z]
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np
from plyfile import PlyData

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                 qvec2rotmat, rotmat2qvec)


def image_sort_key(name):
    """Order image1.jpg .. image10.jpg numerically, not lexicographically.

    poses_bounds rows must line up with the camera index the loader derives from
    the file name, and plain string sort puts image10 before image2.
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def write_images_binary(images, path):
    """COLMAP images.bin, keeping the 2D observations of each image."""
    import struct

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images.values():
            f.write(struct.pack("<i", img.id))
            f.write(struct.pack("<4d", *img.qvec))
            f.write(struct.pack("<3d", *img.tvec))
            f.write(struct.pack("<i", img.camera_id))
            f.write(img.name.encode("utf-8") + b"\x00")
            xys = np.atleast_2d(np.asarray(img.xys, dtype=np.float64).reshape(-1, 2))
            ids = np.asarray(img.point3D_ids, dtype=np.int64).ravel()
            n = 0 if ids.size == 0 else len(ids)
            f.write(struct.pack("<Q", n))
            for k in range(n):
                f.write(struct.pack("<ddq", xys[k, 0], xys[k, 1], ids[k]))


def rotation_to_axis(up, target):
    """Minimal rotation taking unit vector `up` onto unit vector `target`."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    t = np.asarray(target, float) / np.linalg.norm(target)
    v, c = np.cross(up, t), float(up @ t)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def poses_bounds_rows(extrinsics, xyz, hwf):
    """LLFF rows: 3x5 pose ([down, right, back, t, hwf]) + near/far.

    Same convention as customer_scripts/convert_to_multipleview.py, which is the
    inverse of the [P1, -P0, P2, P3] remap done by scene/multipleview_dataset.py.
    """
    rows = []
    for key in sorted(extrinsics, key=lambda k: image_sort_key(extrinsics[k].name)):
        extr = extrinsics[key]
        R_w2c = qvec2rotmat(extr.qvec)
        t = np.asarray(extr.tvec, float)
        R_c2w = R_w2c.T
        center = -R_c2w @ t
        G = np.stack([R_c2w[:, 0], -R_c2w[:, 1], -R_c2w[:, 2]], axis=1)
        P = np.stack([-G[:, 1], G[:, 0], G[:, 2], center], axis=1)
        z = (R_w2c @ xyz.T).T[:, 2] + t[2]
        z = z[z > 0]
        near, far = np.percentile(z, 0.5), np.percentile(z, 99.5)
        rows.append(np.concatenate([np.concatenate([P, hwf], axis=1).ravel(), [near, far]]))
    return np.stack(rows)


def rotate_ply(src, dst, R):
    ply = PlyData.read(src)
    v = ply["vertex"]
    data = v.data.copy()
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64) @ R.T
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if all(k in data.dtype.names for k in ("nx", "ny", "nz")):
        n = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float64) @ R.T
        data["nx"], data["ny"], data["nz"] = n[:, 0], n[:, 1], n[:, 2]
    v.data = data
    ply.write(dst)
    return len(data)


def main(src, dst, up_axis):
    sparse_src = os.path.join(src, "sparse_")
    extr = read_extrinsics_binary(os.path.join(sparse_src, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(sparse_src, "cameras.bin"))

    ups = np.stack([-qvec2rotmat(extr[k].qvec).T[:, 1] for k in extr])
    up = ups.mean(0)
    up /= np.linalg.norm(up)
    spread = np.degrees(np.arccos(np.clip(ups @ up, -1, 1))).max()
    target = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[up_axis]
    R = rotation_to_axis(up, target)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"current world up : {np.round(up, 4)}  (cameras agree within {spread:.1f} deg)")
    print(f"rotating by {angle:.2f} deg so up becomes +{up_axis}")
    if spread > 15:
        print(f"WARNING: cameras disagree on 'up' by {spread:.1f} deg -- the rig may not be "
              f"upright, so this rotation is only an average.")

    os.makedirs(dst, exist_ok=True)

    # 1. extrinsics: R_w2c' = R_w2c R^T, translation untouched
    rotated = {}
    for key, e in extr.items():
        rotated[key] = e._replace(qvec=rotmat2qvec(qvec2rotmat(e.qvec) @ R.T))
    sparse_dst = os.path.join(dst, "sparse_")
    os.makedirs(sparse_dst, exist_ok=True)
    write_images_binary(rotated, os.path.join(sparse_dst, "images.bin"))
    shutil.copyfile(os.path.join(sparse_src, "cameras.bin"),
                    os.path.join(sparse_dst, "cameras.bin"))
    print(f"wrote {sparse_dst}/images.bin ({len(rotated)} images) + cameras.bin")

    # 2. point cloud
    n = rotate_ply(os.path.join(src, "points3D_multipleview.ply"),
                   os.path.join(dst, "points3D_multipleview.ply"), R)
    print(f"wrote points3D_multipleview.ply ({n} points)")

    # 3. poses_bounds, regenerated from the rotated poses
    xyz = PlyData.read(os.path.join(dst, "points3D_multipleview.ply"))["vertex"]
    xyz = np.stack([xyz["x"], xyz["y"], xyz["z"]], axis=1).astype(np.float64)
    cam0 = intr[sorted(intr)[0]]
    hwf = np.array([cam0.height, cam0.width, cam0.params[0]], dtype=float).reshape(3, 1)
    rows = poses_bounds_rows(rotated, xyz, hwf)
    np.save(os.path.join(dst, "poses_bounds_multipleview.npy"), rows)
    print(f"wrote poses_bounds_multipleview.npy {rows.shape}")

    # 4. frames: symlink the camNN folders straight at the source frames
    for name in sorted(os.listdir(src)):
        cam_dir = os.path.join(src, name)
        if name.startswith("cam") and os.path.isdir(cam_dir):
            link = os.path.join(dst, name)
            if os.path.islink(link):
                os.remove(link)
            elif os.path.exists(link):
                shutil.rmtree(link)
            os.symlink(os.path.abspath(cam_dir), link)
    print(f"symlinked camNN folders into {dst}")

    # 5. report the result
    print("\nupright check:")
    for key in sorted(rotated, key=lambda k: image_sort_key(rotated[k].name)):
        e = rotated[key]
        R_c2w = qvec2rotmat(e.qvec).T
        c = -R_c2w @ np.asarray(e.tvec)
        print(f"  {e.name:<14} C={np.array2string(c, precision=3)}"
              f"  up={np.array2string(-R_c2w[:, 1], precision=3)}"
              f"  fwd={np.array2string(R_c2w[:, 2], precision=3)}")
    np.save(os.path.join(dst, "upright_rotation.npy"), R)
    print(f"\nrotation matrix saved to {dst}/upright_rotation.npy "
          f"(X_new = R @ X_old)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default=None)
    ap.add_argument("--up", choices=["x", "y", "z"], default="z")
    a = ap.parse_args()
    dst = a.dst or a.src.rstrip("/") + f"_up{a.up}"
    main(a.src.rstrip("/"), dst, a.up)
