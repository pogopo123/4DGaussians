"""View COLMAP cameras (sparse_/{cameras,images}.bin) together with the
points3D_multipleview.ply point cloud.

Writes an interactive standalone HTML (plotly, JS inlined) plus a few static
PNG views so the layout can be checked without a browser.

Usage: python customer_scripts/view_cameras_points.py <multipleview_scene_dir> [out_dir]
"""
import os
import sys

import numpy as np
from plyfile import PlyData

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from upright_dataset import image_sort_key


def load_points(path, max_points=200000):
    ply = PlyData.read(path)["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=1)
    try:
        rgb = np.stack([ply["red"], ply["green"], ply["blue"]], axis=1) / 255.0
    except ValueError:
        rgb = np.full_like(xyz, 0.5)
    if max_points and len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return xyz, rgb


def load_cameras(sparse_dir):
    extr = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    cams = []
    for key in sorted(extr, key=lambda k: image_sort_key(extr[k].name)):
        e = extr[key]
        cam = intr[e.camera_id]
        R_w2c = qvec2rotmat(e.qvec)
        t = np.asarray(e.tvec)
        R_c2w = R_w2c.T
        center = -R_c2w @ t
        fx = cam.params[0]
        cams.append(dict(name=e.name, R_c2w=R_c2w, center=center,
                         w=cam.width, h=cam.height, fx=fx, model=cam.model))
    return cams


def frustum_lines(cam, scale):
    """Return the 3D segments of a camera frustum (pyramid) in world space."""
    w, h, fx = cam["w"], cam["h"], cam["fx"]
    z = scale
    x = 0.5 * w / fx * z
    y = 0.5 * h / fx * z
    # camera coords are [right, down, forward] -> corners 0,1 are the TOP edge
    corners_cam = np.array([[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
                            [0.0, -1.6 * y, z]])  # apex of the "this way up" marker
    pts = corners_cam @ cam["R_c2w"].T + cam["center"]
    corners, apex = pts[:4], pts[4]
    c = cam["center"]
    segs = []
    for i in range(4):
        segs.append((c, corners[i]))
        segs.append((corners[i], corners[(i + 1) % 4]))
    segs.append((corners[0], apex))
    segs.append((corners[1], apex))
    return segs, corners


PALETTE_RGB = [(230, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 49), (145, 30, 180),
               (70, 240, 240), (240, 50, 230), (188, 246, 12), (0, 128, 128), (154, 99, 36)]


def write_ply_points(path, xyz, rgb):
    """Plain binary point-cloud ply (x y z red green blue) — every viewer reads this."""
    from plyfile import PlyElement

    arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def write_ply_mesh(path, verts, rgb, faces):
    """Triangle-mesh ply — solid frustum pyramids for viewers that render faces."""
    from plyfile import PlyElement

    v = np.empty(len(verts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    f = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    f["vertex_indices"] = np.asarray(faces, dtype=np.int32)
    PlyData([PlyElement.describe(v, "vertex"), PlyElement.describe(f, "face")]).write(path)


def sample_frustum_points(cams, scale, spacing):
    """Draw each camera frustum as densely sampled coloured points."""
    pts, cols = [], []
    for i, cam in enumerate(cams):
        col = np.array(PALETTE_RGB[i % len(PALETTE_RGB)], dtype=float)
        for a, b in frustum_lines(cam, scale)[0]:
            a, b = np.asarray(a, float), np.asarray(b, float)
            n = max(int(np.linalg.norm(b - a) / spacing), 2)
            t = np.linspace(0, 1, n)[:, None]
            pts.append(a + t * (b - a))
            cols.append(np.repeat(col[None], n, axis=0))
        # a solid blob at the optical centre so it stays visible when zoomed out
        blob = cam["center"] + np.random.default_rng(i).normal(0, scale * 0.02, (300, 3))
        pts.append(blob)
        cols.append(np.repeat(col[None], len(blob), axis=0))
    return np.concatenate(pts), np.concatenate(cols)


def frustum_mesh(cams, scale):
    verts, cols, faces = [], [], []
    for i, cam in enumerate(cams):
        col = np.array(PALETTE_RGB[i % len(PALETTE_RGB)], dtype=float)
        corners = frustum_lines(cam, scale)[1]
        base = len(verts)
        verts.extend([cam["center"], *corners])
        cols.extend([col] * 5)
        for k in range(4):                       # four side triangles
            faces.append([base, base + 1 + k, base + 1 + (k + 1) % 4])
        faces.append([base + 1, base + 2, base + 3])   # image plane
        faces.append([base + 1, base + 3, base + 4])
    return np.array(verts), np.array(cols), faces


def export_ply(out_dir, xyz, rgb, cams, scale):
    """One merged ply (scene points + camera wireframes) plus a cameras-only pair."""
    spacing = scale / 250.0
    cam_xyz, cam_rgb = sample_frustum_points(cams, scale, spacing)

    merged = os.path.join(out_dir, "cameras_points_merged.ply")
    write_ply_points(merged, np.vstack([xyz, cam_xyz]),
                     np.vstack([rgb * 255.0, cam_rgb]))
    print(f"wrote {merged}  ({len(xyz)} scene + {len(cam_xyz)} camera points)")

    cam_only = os.path.join(out_dir, "cameras_only.ply")
    write_ply_points(cam_only, cam_xyz, cam_rgb)
    print(f"wrote {cam_only}")

    v, c, f = frustum_mesh(cams, scale)
    mesh = os.path.join(out_dir, "cameras_mesh.ply")
    write_ply_mesh(mesh, v, c, f)
    print(f"wrote {mesh}  ({len(f)} triangles)")


def upright_rotation(cams, target):
    """Rigid rotation taking the cameras' shared up direction onto `target`.

    COLMAP's world frame is arbitrary (it is whatever the first registered image
    implied), so an otherwise perfect reconstruction still looks tilted in a
    z-up or y-up viewer. This only re-expresses the same geometry in a nicer
    basis -- no reconstruction parameter is touched.
    """
    up = np.mean([-c["R_c2w"][:, 1] for c in cams], axis=0)
    up /= np.linalg.norm(up)
    t = np.asarray(target, dtype=float)
    t /= np.linalg.norm(t)
    v = np.cross(up, t)
    s, c = np.linalg.norm(v), float(up @ t)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def main(scene_dir, out_dir, up_axis="none"):
    os.makedirs(out_dir, exist_ok=True)
    xyz, rgb = load_points(os.path.join(scene_dir, "points3D_multipleview.ply"), max_points=None)
    cams = load_cameras(os.path.join(scene_dir, "sparse_"))

    if up_axis != "none":
        target = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[up_axis]
        R = upright_rotation(cams, target)
        xyz = xyz @ R.T
        for cam in cams:
            cam["center"] = R @ cam["center"]
            cam["R_c2w"] = R @ cam["R_c2w"]
        print(f"[upright] rotated world so the cameras' up axis becomes +{up_axis}")

    centers = np.stack([c["center"] for c in cams])
    # frustum length ~ a quarter of the typical camera-to-scene distance
    scale = 0.25 * float(np.mean(np.linalg.norm(centers - np.median(xyz, axis=0), axis=1)))

    print(f"points: {len(xyz)}  bbox_min={xyz.min(0)}  bbox_max={xyz.max(0)}")
    print(f"cameras: {len(cams)}  ({cams[0]['model']} {cams[0]['w']}x{cams[0]['h']})")
    for c in cams:
        fwd = c["R_c2w"][:, 2]
        print(f"  {c['name']:<20} C={np.array2string(c['center'], precision=3)}"
              f"  fwd={np.array2string(fwd, precision=3)}  f={c['fx']:.1f}")
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    print(f"camera baselines: min={d[d>0].min():.3f} max={d.max():.3f}")
    print(f"cam centroid -> point centroid dist: "
          f"{np.linalg.norm(centers.mean(0) - xyz.mean(0)):.3f}")

    # ---------- ply export (open in MeshLab / CloudCompare / Blender) ----------
    export_ply(out_dir, xyz, rgb, cams, scale)

    # ---------- interactive html ----------
    import plotly.graph_objects as go

    sel = np.random.default_rng(0).choice(len(xyz), min(200000, len(xyz)), replace=False)
    vx, vc = xyz[sel], rgb[sel]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=vx[:, 0], y=vx[:, 1], z=vx[:, 2], mode="markers", name="points3D",
        marker=dict(size=1.2, color=[f"rgb({r},{g},{b})" for r, g, b in (vc * 255).astype(int)])))

    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
               "#46f0f0", "#f032e6", "#bcf60c", "#008080", "#9a6324"]
    for i, cam in enumerate(cams):
        segs, _ = frustum_lines(cam, scale)
        lx, ly, lz = [], [], []
        for a, b in segs:
            lx += [a[0], b[0], None]
            ly += [a[1], b[1], None]
            lz += [a[2], b[2], None]
        col = palette[i % len(palette)]
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", name=cam["name"],
                                   line=dict(color=col, width=4)))
        fig.add_trace(go.Scatter3d(x=[cam["center"][0]], y=[cam["center"][1]],
                                   z=[cam["center"][2]], mode="text",
                                   text=[cam["name"]], textposition="top center",
                                   showlegend=False, textfont=dict(color=col, size=11)))

    fig.update_layout(scene=dict(aspectmode="data"), title=os.path.basename(scene_dir.rstrip("/")),
                      margin=dict(l=0, r=0, t=30, b=0))
    html = os.path.join(out_dir, "cameras_points.html")
    fig.write_html(html, include_plotlyjs=True)
    print(f"\nwrote {html}")

    # ---------- static previews ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = xyz[np.random.default_rng(1).choice(len(xyz), min(30000, len(xyz)), replace=False)]
    views = [("top_xy", (90, -90)), ("front_xz", (0, -90)), ("side_yz", (0, 0)), ("iso", (25, -60))]
    fig2 = plt.figure(figsize=(16, 12))
    for k, (title, (elev, azim)) in enumerate(views, 1):
        ax = fig2.add_subplot(2, 2, k, projection="3d")
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.3, c="0.6", alpha=0.4, linewidths=0)
        for i, cam in enumerate(cams):
            segs, corners = frustum_lines(cam, scale)
            col = palette[i % len(palette)]
            for a, b in segs:
                ax.plot(*zip(a, b), color=col, lw=1.0)
            ax.text(*cam["center"], cam["name"], color=col, fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title} (elev={elev}, azim={azim})")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_box_aspect(np.ptp(np.vstack([sub, centers]), axis=0))
    png = os.path.join(out_dir, "cameras_points.png")
    fig2.tight_layout(); fig2.savefig(png, dpi=110)
    print(f"wrote {png}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", nargs="?", default="data/multipleview/backpack_frame0_v2")
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument("--up", choices=["none", "x", "y", "z"], default="none",
                    help="rotate the exported world so the cameras' up axis lands on this "
                         "axis (viewing convenience only; does not touch the dataset)")
    a = ap.parse_args()
    out = a.out_dir or os.path.join(a.scene_dir, "viz" if a.up == "none" else f"viz_up{a.up}")
    main(a.scene_dir, out, a.up)
