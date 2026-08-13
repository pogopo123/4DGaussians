#
# Visualization of the Motion Mask Head (Motion_Mask_4DGS_Technical_Report.md)
#
# Produces, for a trained motion-mask model:
#   1. Heatmap renders: gaussians coloured blue (static, m~0) -> red (dynamic, m~1)
#   2. Histogram of m_i values across timestamps (bimodality check)
#   3. Dynamic-only renders: only gaussians with m_i > epsilon, on black background
#   4. Static-only renders + KNN spatial-consistency (isolated-noise) metric
#
# Usage:
#   python visualize_motion_mask.py --model_path output/multipleview/backpack_mask \
#       --configs arguments/multipleview/backpack_mask.py [--iteration 15000]
#
import math
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from utils.general_utils import safe_state
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

to8b = lambda x: (255 * np.clip(x.detach().cpu().numpy(), 0, 1)).astype(np.uint8)


@torch.no_grad()
def deform_at_time(gaussians, t):
    """Run the deformation network at normalized time t, return final params + motion scores."""
    means3D = gaussians.get_xyz
    time = torch.tensor(t, dtype=torch.float32).to(means3D.device).repeat(means3D.shape[0], 1)
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = gaussians._deformation(
        means3D, gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features, time)
    motion_out = gaussians._deformation.deformation_net.motion_out
    if motion_out is None:
        # flow_gate=False / flow_split / flow_merge: khong co motion mask.
        # Tra ve m=None; cac panel heat/dynamic/static se bi bo qua.
        return means3D_final, scales_final, rotations_final, opacity_final, shs_final, None
    m = motion_out["score"].squeeze(-1)  # [N]
    return means3D_final, scales_final, rotations_final, opacity_final, shs_final, m


@torch.no_grad()
def rasterize(cam, gaussians, bg_color, means3D, scales_raw, rotations_raw, opacity_raw, shs,
              colors_precomp=None, point_mask=None):
    """Minimal rasterization pass. point_mask (bool [N]) suppresses opacity of masked-out points."""
    scales = gaussians.scaling_activation(scales_raw)
    rotations = gaussians.rotation_activation(rotations_raw)
    opacity = gaussians.opacity_activation(opacity_raw)
    if point_mask is not None:
        opacity = opacity * point_mask.float().unsqueeze(-1)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height),
        image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5),
        tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam.world_view_transform.cuda(),
        projmatrix=cam.full_proj_transform.cuda(),
        sh_degree=gaussians.active_sh_degree,
        campos=cam.camera_center.cuda(),
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    screenspace_points = torch.zeros_like(means3D)
    image, _, _ = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=shs if colors_precomp is None else None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)
    return image


def heat_colors(m, denom=1.0):
    """m -> blue (static) to red (dynamic).

    `denom` rescales before colouring. With a well-separated mask the raw m
    already spans [0,1] and denom=1 is right. When L_sparse has crushed the
    mask (max(m) well under 1) the raw picture is uniformly blue and shows
    nothing, so passing a high quantile as denom restores the *relative*
    structure -- at the cost that colour no longer reads as absolute m.
    """
    mn = (m / denom).clamp(0.0, 1.0)
    r = mn
    g = 0.15 * torch.ones_like(mn)
    b = 1.0 - mn
    return torch.stack([r, g, b], dim=-1)


@torch.no_grad()
def spatial_consistency(xyz, m, k=8, n_sample=16384):
    """Fraction of points whose m strongly disagrees with their KNN mean ('isolated noise')."""
    n = xyz.shape[0]
    idx = torch.randperm(n, device=xyz.device)[:min(n_sample, n)]
    pts, ms = xyz[idx], m[idx]
    knn = torch.cdist(pts, pts).topk(k + 1, largest=False).indices[:, 1:]
    knn_mean = ms[knn].mean(dim=-1)
    disagree = (ms - knn_mean).abs()
    return (disagree > 0.3).float().mean().item(), disagree.mean().item()


@torch.no_grad()
def run(dataset, hyperparam, iteration, epsilon, n_video_frames, cams_mode="video", cam_name=None,
        heat_scale="raw", video_scale=0.5):
    gaussians = GaussianModel(dataset.sh_degree, hyperparam)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    black = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    white = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    bg = white if dataset.white_background else black

    out_dir = os.path.join(dataset.model_path, "motion_viz")
    os.makedirs(out_dir, exist_ok=True)

    if cams_mode == "video":
        cams = scene.getVideoCameras()
        get_cam = lambda i: cams[i]
        n_total = len(cams)
        has_gt = False
        tag = "video"
    else:
        ds = scene.getTrainCameras() if cams_mode == "train" else scene.getTestCameras()
        paths = getattr(getattr(ds, "dataset", None), "image_paths", None)
        if paths is not None and cam_name:
            indices = [i for i, p in enumerate(paths) if os.sep + cam_name + os.sep in p]
            assert indices, f"no frames found for camera '{cam_name}'"
        else:
            indices = list(range(len(ds)))
        get_cam = lambda i: ds[indices[i]]
        n_total = len(indices)
        has_gt = True
        tag = cams_mode + (f"_{cam_name}" if cam_name else "")
    print(f"Loaded iteration {scene.loaded_iter}, {gaussians.get_xyz.shape[0]} gaussians, "
          f"{n_total} {cams_mode} cameras")

    # pick the heatmap normaliser once, from the mask at mid-sequence
    _, _, _, _, _, m_probe = deform_at_time(gaussians, get_cam(n_total // 2).time)
    has_mask = m_probe is not None
    if not has_mask:
        print("model khong co motion mask -- chi xuat GT + render")
        denom = 1.0
    elif heat_scale == "norm":
        denom = max(m_probe.quantile(0.999).item(), 1e-6)
        print(f"heatmap normalised by q99.9(m) = {denom:.4f} "
              f"(raw max(m) = {m_probe.max().item():.4f}) -- colours are RELATIVE")
    else:
        denom = 1.0
        print(f"heatmap on the raw m scale (max(m) = {m_probe.max().item():.4f})")

    # ---------- 1+3+4. still images at 5 timestamps ----------
    all_m = {}
    frac_idx = [int(f * (n_total - 1)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for i in frac_idx:
        cam = get_cam(i)
        t = cam.time
        mu, s, r, o, shs, m = deform_at_time(gaussians, t)
        if m is not None: all_m[t] = m.cpu()

        rgb = rasterize(cam, gaussians, bg, mu, s, r, o, shs)
        panels = [rgb]
        name = f"{tag}_t{t:.3f}_rgb.png"
        if has_mask:
            panels += [rasterize(cam, gaussians, black, mu, s, r, o, shs, colors_precomp=heat_colors(m, denom)),
                       rasterize(cam, gaussians, black, mu, s, r, o, shs, point_mask=(m > epsilon)),
                       rasterize(cam, gaussians, black, mu, s, r, o, shs, point_mask=(m <= epsilon))]
            name = f"{tag}_t{t:.3f}_rgb_heat_dyn_static.png"
        if has_gt:
            panels.insert(0, cam.original_image[:3].cuda())
            name = "gt_" + name
        row = np.concatenate([to8b(x).transpose(1, 2, 0) for x in panels], axis=1)
        imageio.imwrite(os.path.join(out_dir, name), row)

        if has_mask:
            noise_ratio, mean_disagree = spatial_consistency(mu, m)
            print(f"t={t:.3f}: dynamic ratio={(m > epsilon).float().mean().item():.3f} "
                  f"| m mean={m.mean():.3f} | isolated-noise={noise_ratio:.4f}")
        else:
            print(f"t={t:.3f}: (khong co mask)")

    # ---------- 2. histogram ----------
    if not all_m:
        print("bo qua histogram (khong co mask)")
    else:
      fig, ax = plt.subplots(figsize=(7, 4.5))
      for t, m in all_m.items():
        ax.hist(m.numpy(), bins=100, range=(0, 1), alpha=0.5, label=f"t={t:.2f}", log=True)
      ax.axvline(epsilon, color="k", ls="--", lw=1, label=f"epsilon={epsilon}")
      ax.set_xlabel("motion score $m_i$")
      ax.set_ylabel("count (log)")
      ax.set_title("Motion mask distribution (bimodal = good separation)")
      ax.legend()
      fig.tight_layout()
      fig.savefig(os.path.join(out_dir, "motion_histogram.png"), dpi=150)
      plt.close(fig)

      m_cat = torch.cat(list(all_m.values()))
      lo, hi = (m_cat < 0.1).float().mean().item(), (m_cat > 0.9).float().mean().item()
      print(f"histogram: {lo*100:.1f}% m<0.1, {hi*100:.1f}% m>0.9, {(1-lo-hi)*100:.1f}% giua")

    # ---------- videos: rgb | heatmap | dynamic-only over time ----------
    step = max(1, n_total // n_video_frames)
    frames = []
    for i in tqdm(range(0, n_total, step), desc="heatmap video"):
        cam = get_cam(i)
        mu, s, r, o, shs, m = deform_at_time(gaussians, cam.time)
        rgb = rasterize(cam, gaussians, bg, mu, s, r, o, shs)
        panels = [rgb]
        if has_mask:
            panels += [rasterize(cam, gaussians, black, mu, s, r, o, shs, colors_precomp=heat_colors(m, denom)),
                       rasterize(cam, gaussians, black, mu, s, r, o, shs, point_mask=(m > epsilon))]
        if has_gt:
            panels.insert(0, cam.original_image[:3].cuda())
        row = np.concatenate([to8b(x).transpose(1, 2, 0) for x in panels], axis=1)
        if video_scale != 1.0:
            # four 2048px panels side by side exceed what h264 will encode
            import PIL.Image
            h, w = row.shape[:2]
            row = np.asarray(PIL.Image.fromarray(row).resize(
                (int(w * video_scale) // 2 * 2, int(h * video_scale) // 2 * 2),
                PIL.Image.BILINEAR))
        frames.append(row)
    vid = os.path.join(out_dir, f"motion_mask_video_{tag}.mp4")
    imageio.mimwrite(vid, frames, fps=15, quality=8)
    print(f"video: {vid}  ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")

    # ---------- post-hoc TensorBoard events (motion_histogram over time) ----------
    if not all_m:
        print(f"\nAll outputs saved in {out_dir}")
        return
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(dataset.model_path, "motion_viz_tb"))
        for step, f in enumerate(np.linspace(0, 1, 21)):
            _, _, _, _, _, m = deform_at_time(gaussians, float(f))
            # add_histogram_raw avoids a numpy>=1.24 incompatibility in torch's make_histogram
            v = m.cpu().numpy()
            counts, edges = np.histogram(v, bins=100, range=(0.0, 1.0))
            tb.add_histogram_raw("motion_viz/motion_mask_histogram",
                                 min=float(v.min()), max=float(v.max()), num=int(v.size),
                                 sum=float(v.sum()), sum_squares=float((v ** 2).sum()),
                                 bucket_limits=edges[1:].tolist(), bucket_counts=counts.tolist(),
                                 global_step=step)
            tb.add_scalar("motion_viz/dynamic_ratio", (m > epsilon).float().mean().item(), step)
            tb.add_scalar("motion_viz/m_mean", m.mean().item(), step)
        tb.close()
        print(f"TensorBoard events written to {os.path.join(dataset.model_path, 'motion_viz_tb')}")
    except ImportError:
        print("tensorboard not installed - skipping post-hoc event export")

    print(f"\nAll outputs saved in {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Motion mask visualization")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--epsilon", default=None, type=float, help="dynamic threshold (default: from config)")
    parser.add_argument("--n_video_frames", default=150, type=int)
    parser.add_argument("--cams", default="video", choices=["video", "train", "test"],
                        help="camera set to render from (train/test include a GT panel)")
    parser.add_argument("--cam_name", default=None, type=str,
                        help="restrict train/test rendering to one physical camera, e.g. cam01")
    parser.add_argument("--heat_scale", default="raw", choices=["raw", "norm"],
                        help="'norm' rescales the heatmap by q99.9(m); use it when the mask "
                             "has collapsed and the raw picture is uniformly blue")
    parser.add_argument("--video_scale", default=0.5, type=float)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    safe_state(args.quiet)
    epsilon = getattr(args, "epsilon", None)
    if epsilon is None:
        epsilon = getattr(args, "motion_mask_epsilon", 0.05)
    run(model.extract(args), hyperparam.extract(args), args.iteration, epsilon, args.n_video_frames,
        cams_mode=getattr(args, "cams", "video"), cam_name=getattr(args, "cam_name", None),
        heat_scale=getattr(args, "heat_scale", "raw"),
        video_scale=getattr(args, "video_scale", 0.5))
