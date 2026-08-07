"""Split a rendered frame into deformation time and rasterization time.

Early-exit only ever shrinks the deformation half. At 2K resolution with a few
hundred thousand Gaussians the rasterizer dominates, so an end-to-end FPS number
badly understates what the deformation network gained -- this script measures
the two halves separately so the speedup can be attributed honestly.

    python scripts/benchmark_deform.py --model_path output/multipleview/<exp> \
        --configs arguments/multipleview/backpack_f3gs.py --iteration 14000

Add --sweep to also report the deformation cost against an imposed static
fraction, which shows how the early-exit saving scales with scene sparsity
independently of how well the mask happens to be trained.
"""

import os
import sys
from argparse import ArgumentParser

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, Scene


def timeit(fn, n_warm=5, n_iter=30):
    """-> milliseconds per call."""
    with torch.no_grad():
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(n_iter):
            fn()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def build_raster_settings(cam, pc, pipe, bg):
    import math
    return GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform.cuda(),
        projmatrix=cam.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree, campos=cam.camera_center.cuda(),
        prefiltered=False, debug=pipe.debug)


def measure(pc, cam, pipe, bg, early_exit):
    """-> (deform_ms, raster_ms, dynamic_fraction)."""
    net = pc._deformation.deformation_net
    was = net.early_exit
    net.early_exit = early_exit

    xyz, scales, rots = pc.get_xyz, pc._scaling, pc._rotation
    opacity, shs = pc._opacity, pc.get_features
    t = torch.tensor(cam.time).to(xyz.device).repeat(xyz.shape[0], 1)

    deform = lambda: pc._deformation(xyz, scales, rots, opacity, shs, t)
    deform_ms = timeit(deform)

    with torch.no_grad():
        m3, sc, rt, op, sh = deform()
        mo = net.motion_out
        frac = 1.0
        if mo is not None and mo.get("dynamic_idx") is not None:
            frac = mo["dynamic_idx"].numel() / xyz.shape[0]
        elif mo is not None:
            frac = (mo["score"] > net.motion_mask_epsilon).float().mean().item()
        sc = pc.scaling_activation(sc)
        rt = pc.rotation_activation(rt)
        op = pc.opacity_activation(op)

    rasterizer = GaussianRasterizer(raster_settings=build_raster_settings(cam, pc, pipe, bg))
    means2D = torch.zeros_like(xyz)
    raster_ms = timeit(lambda: rasterizer(
        means3D=m3, means2D=means2D, shs=sh, colors_precomp=None,
        opacities=op, scales=sc, rotations=rt, cov3D_precomp=None))

    net.early_exit = was
    return deform_ms, raster_ms, frac


def row(tag, deform, raster, frac=None):
    total = deform + raster
    extra = f"{100*frac:5.1f}%" if frac is not None else "    -"
    print(f"  {tag:34s} {deform:8.2f} {raster:9.2f} {total:8.2f} {1000/total:7.1f}  {extra}")


def main():
    parser = ArgumentParser("Deformation vs rasterization latency breakdown")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyper = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--view", default=0, type=int, help="index into the test cameras")
    parser.add_argument("--sweep", action="store_true",
                        help="also sweep an imposed static fraction")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mmcv.Config.fromfile(args.configs))

    dataset, pipe, hp = model.extract(args), pipeline.extract(args), hyper.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hp)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                          dtype=torch.float32, device="cuda")
        cams = scene.getTestCameras()
        cam = cams[min(args.view, len(cams) - 1)]
        net = gaussians._deformation.deformation_net
        net.mask_warmup = False

        n = gaussians.get_xyz.shape[0]
        print(f"\n{n:,} Gaussians   {int(cam.image_width)}x{int(cam.image_height)}   "
              f"flow_attn={net.use_flow_attn} mask_from_fused={net.mask_from_fused} "
              f"flow_first={net.can_flow_first()}")
        print(f"\n  {'':34s} {'deform':>8s} {'raster':>9s} {'total':>8s} {'FPS':>7s}  dynamic")
        print("  " + "-" * 76)

        d0, r0, f0 = measure(gaussians, cam, pipe, bg, early_exit=False)
        row("dense (every point deformed)", d0, r0, f0)
        d1, r1, f1 = measure(gaussians, cam, pipe, bg, early_exit=True)
        row("early-exit", d1, r1, f1)

        print("  " + "-" * 76)
        print(f"  deformation      {d0:.2f} -> {d1:.2f} ms  ({100*(d1-d0)/d0:+.0f}%)")
        print(f"  whole frame      {d0+r0:.2f} -> {d1+r1:.2f} ms  ({100*((d1+r1)-(d0+r0))/(d0+r0):+.0f}%)")
        share = 100 * r0 / (d0 + r0)
        print(f"  rasterizer is {share:.0f}% of the dense frame, deformation {100-share:.0f}%"
              + ("  -- deformation dominates, so early-exit attacks the right half"
                 if share < 40 else
                 "  -- the rasterizer caps what early-exit can do end to end"))
        print("  (render.py's reported FPS is much lower than 1000/total: its loop also loads each"
              "\n   image from disk and runs to8b() on the CPU. It is not a rendering benchmark.)")

        if args.sweep:
            if not net.use_flow_attn:
                print("\n  --sweep needs the flow branch; skipping")
                return
            print(f"\n  deformation cost vs imposed static fraction "
                  f"(mask head bias overridden, N={n:,}):")
            print(f"\n  {'static':>8s} {'dense':>10s} {'early-exit':>12s} {'saving':>9s}")
            print("  " + "-" * 43)
            head = (net.flow_field.mask_head if not net.mask_from_fused
                    else net.gated_decoder.mask_head)
            bias0 = head.bias.detach().clone()
            xyz = gaussians.get_xyz
            t = torch.tensor(cam.time).to(xyz.device).repeat(xyz.shape[0], 1)
            for target in (0.0, 0.3, 0.5, 0.7, 0.9):
                with torch.no_grad():
                    head.bias.copy_(bias0)
                    gaussians._deformation(xyz, gaussians._scaling, gaussians._rotation,
                                           gaussians._opacity, gaussians.get_features, t)
                    logits = torch.logit(net.motion_out["score"].clamp(1e-6, 1 - 1e-6))
                    import math
                    shift = (math.log(net.motion_mask_epsilon / (1 - net.motion_mask_epsilon))
                             - logits.squeeze(-1).quantile(target).item())
                    head.bias.copy_(bias0 + shift if target > 0 else bias0 + 50.0)
                da, _, _ = measure(gaussians, cam, pipe, bg, early_exit=False)
                db, _, fr = measure(gaussians, cam, pipe, bg, early_exit=True)
                print(f"  {100*(1-fr):7.0f}% {da:9.2f}ms {db:11.2f}ms {100*(db-da)/da:8.0f}%")
            head.bias.copy_(bias0)


if __name__ == "__main__":
    main()
