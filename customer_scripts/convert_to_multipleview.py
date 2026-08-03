"""Convert a flat colmap capture (images/camXX_YYYY.png + sparse/0 text model)
into the layout expected by scene/multipleview_dataset.py:

    <dst>/cam01/frame_00001.jpg ...      (symlinks to the source frames)
    <dst>/sparse_/{cameras,images}.bin   (one entry per camera, named imageN.jpg)
    <dst>/points3D_multipleview.ply
    <dst>/poses_bounds_multipleview.npy  (LLFF, used for the spiral video path)

By default the world frame is also made upright (--up z): colmap fixes the world
axes from the first registered image, so a correct reconstruction still comes out
tilted -- for backpack_frame0_v2 the real up was 21 deg off +x. The rotation is
applied to the point cloud, the extrinsics and poses_bounds together, so the
reconstruction is unchanged up to a change of basis. Pass --up none to keep
colmap's original frame.

Usage: python customer_scripts/convert_to_multipleview.py <src_dir> <dst_dir> [--up z]
"""
import argparse
import os
import re
import struct
import sys
import shutil

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scene.colmap_loader import (read_extrinsics_text, read_intrinsics_text,
                                 qvec2rotmat, rotmat2qvec)
from upright_dataset import rotation_to_axis, rotate_ply, image_sort_key


CAMERA_MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3, "OPENCV": 4}


def write_cameras_binary(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras.values():
            model_id = CAMERA_MODEL_IDS[cam.model]
            f.write(struct.pack("<iiQQ", cam.id, model_id, cam.width, cam.height))
            f.write(struct.pack("<" + "d" * len(cam.params), *cam.params))


def write_images_binary(images, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images.values():
            f.write(struct.pack("<i", img.id))
            f.write(struct.pack("<4d", *img.qvec))
            f.write(struct.pack("<3d", *img.tvec))
            f.write(struct.pack("<i", img.camera_id))
            f.write(img.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D points needed by the loader


def camera_centers_and_bounds(extrinsics, xyz):
    """LLFF poses_bounds rows: 3x5 pose ([down, right, back, t, hwf]) + near/far."""
    rows = []
    for key in sorted(extrinsics, key=lambda k: image_sort_key(extrinsics[k].name)):
        extr = extrinsics[key]
        R_w2c = qvec2rotmat(extr.qvec)
        t = np.array(extr.tvec)
        R_c2w = R_w2c.T
        center = -R_c2w @ t
        # OpenCV c2w [right, down, forward] -> OpenGL [right, up, back]
        G = np.stack([R_c2w[:, 0], -R_c2w[:, 1], -R_c2w[:, 2]], axis=1)
        # inverse of the loader's [P1, -P0, P2, P3] remap
        P = np.stack([-G[:, 1], G[:, 0], G[:, 2], center], axis=1)
        # depth of the reconstructed points in this camera
        z = (R_w2c @ xyz.T).T[:, 2] + t[2]
        z = z[z > 0]
        near, far = np.percentile(z, 0.5), np.percentile(z, 99.5)
        rows.append((P, near, far))
    return rows


def main(src, dst, up="z"):
    sparse = os.path.join(src, "sparse", "0")
    extrinsics = read_extrinsics_text(os.path.join(sparse, "images.txt"))
    intrinsics = read_intrinsics_text(os.path.join(sparse, "cameras.txt"))

    image_dir = os.path.join(src, "images")
    frames = {}
    for name in os.listdir(image_dir):
        m = re.match(r"(cam\d+)_(\d+)\.\w+$", name)
        if m:
            frames.setdefault(m.group(1), []).append((int(m.group(2)), name))
    for cam in frames:
        frames[cam].sort()

    os.makedirs(dst, exist_ok=True)

    # 1. cam01..camNN folders of symlinked frames, indexed from 1
    ordered = sorted(extrinsics, key=lambda k: image_sort_key(extrinsics[k].name))
    src_cams = [re.match(r"(cam\d+)_", extrinsics[k].name).group(1) for k in ordered]
    for idx, src_cam in enumerate(src_cams, start=1):
        out_dir = os.path.join(dst, "cam%02d" % idx)
        os.makedirs(out_dir, exist_ok=True)
        for i, (_, name) in enumerate(frames[src_cam], start=1):
            link = os.path.join(out_dir, "frame_%05d.jpg" % i)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.abspath(os.path.join(image_dir, name)), link)
        print("%s -> %s (%d frames)" % (src_cam, out_dir, len(frames[src_cam])))

    # 2. sparse_ binary model, images renamed to imageN.jpg so the loader's
    #    name[5:-4] slice yields the camera index
    sparse_dir = os.path.join(dst, "sparse_")
    os.makedirs(sparse_dir, exist_ok=True)
    renamed = {}
    for idx, key in enumerate(ordered, start=1):
        renamed[idx] = extrinsics[key]._replace(id=idx, name="image%d.jpg" % idx)

    # 2b. make the world upright -- one rotation shared by every pose-carrying file
    R = np.eye(3)
    if up != "none":
        ups = np.stack([-qvec2rotmat(e.qvec).T[:, 1] for e in renamed.values()])
        mean_up = ups.mean(0)
        mean_up /= np.linalg.norm(mean_up)
        spread = np.degrees(np.arccos(np.clip(ups @ mean_up, -1, 1))).max()
        R = rotation_to_axis(mean_up, {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[up])
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        print("world up %s -> +%s (rotate %.2f deg, cameras agree within %.1f deg)"
              % (np.round(mean_up, 4), up, angle, spread))
        if spread > 15:
            print("WARNING: cameras disagree on 'up' by %.1f deg -- the rig may not be "
                  "upright, so this rotation is only an average." % spread)
        # R_w2c' = R_w2c R^T, translation untouched
        renamed = {k: e._replace(qvec=rotmat2qvec(qvec2rotmat(e.qvec) @ R.T))
                   for k, e in renamed.items()}
        np.save(os.path.join(dst, "upright_rotation.npy"), R)

    write_images_binary(renamed, os.path.join(sparse_dir, "images.bin"))
    write_cameras_binary(intrinsics, os.path.join(sparse_dir, "cameras.bin"))

    # 3. point cloud, rotated by the same R
    ply_dst = os.path.join(dst, "points3D_multipleview.ply")
    if up == "none":
        shutil.copyfile(os.path.join(sparse, "points3D.ply"), ply_dst)
    else:
        n = rotate_ply(os.path.join(sparse, "points3D.ply"), ply_dst, R)
        print("wrote points3D_multipleview.ply (%d points, rotated)" % n)

    # 4. poses_bounds for the spiral video path, from the same (rotated) poses
    from plyfile import PlyData
    v = PlyData.read(ply_dst)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    cam0 = intrinsics[list(intrinsics)[0]]
    hwf = np.array([cam0.height, cam0.width, cam0.params[0]]).reshape(3, 1)
    rows = []
    for P, near, far in camera_centers_and_bounds(renamed, xyz):
        rows.append(np.concatenate([np.concatenate([P, hwf], axis=1).ravel(), [near, far]]))
    np.save(os.path.join(dst, "poses_bounds_multipleview.npy"), np.stack(rows))
    print("wrote poses_bounds_multipleview.npy", np.stack(rows).shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir")
    ap.add_argument("dst_dir")
    ap.add_argument("--up", choices=["none", "x", "y", "z"], default="z",
                    help="axis the world up direction is rotated onto (default: z)")
    a = ap.parse_args()
    main(a.src_dir, a.dst_dir, a.up)
