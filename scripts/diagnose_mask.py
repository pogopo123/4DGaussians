"""Did the motion mask keep the motion, or did it freeze the subject?

L_sparse only has one fixed point, mean(m) -> 0, so a run can end with a very
sparse mask and still look healthy on the two metrics normally logged:

  * PSNR rises partly *because* of the mask -- static background becomes exactly
    canonical, and background is most of the pixels.
  * the plain flow EPE averages over every valid pixel, and on background both
    the rendered and the prior flow are ~0, so a mask that switched the moving
    subject off still scores well there.

This script separates the two by reporting the same EPE restricted to pixels the
optical-flow prior says are actually moving.

    python scripts/diagnose_mask.py --model_path output/multipleview/<exp> \
        --configs arguments/multipleview/backpack_f3gs.py --iteration 15000
"""

import os
import sys
from argparse import ArgumentParser

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.flow_utils import flow_consistency_loss


def main():
    parser = ArgumentParser("Motion-mask diagnostic")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyper = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--n_views", default=24, type=int)
    parser.add_argument("--moving_px", default=1.0, type=float,
                        help="prior flow longer than this counts as moving")
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
        net = gaussians._deformation.deformation_net
        net.mask_warmup = False
        net.early_exit = False

        cams = scene.getTrainCameras()
        eps = net.motion_mask_epsilon
        n_total = gaussians.get_xyz.shape[0]

        epe_all, epe_mv, cov_all, prior_mag, used = [], [], [], [], 0
        m_last = None
        step = max(1, len(cams) // args.n_views)
        for i in range(0, len(cams), step):
            cam = cams[i]
            if getattr(cam, "flow", None) is None:
                continue
            h_p, w_p = cam.flow.shape[-2:]
            pkg = render(cam, gaussians, pipe, bg, stage="fine",
                         cam_type=scene.dataset_type,
                         flow_dt=cam.flow_dt, flow_size=(w_p, h_p))
            if pkg["flow"] is None:
                continue
            _, epe, cov, epem = flow_consistency_loss(
                pkg["flow"], cam.flow.cuda(), cam.flow_valid.cuda(), cam.flow_orig_size,
                alpha_threshold=hp.flow_alpha_threshold,
                normalize_alpha=hp.flow_normalize_alpha,
                moving_threshold_px=args.moving_px)
            epe_all.append(epe.item()); epe_mv.append(epem.item()); cov_all.append(cov.item())
            # the null baseline: a fully frozen subject renders zero flow there,
            # so its epe_moving would equal the mean prior magnitude
            pf = cam.flow.cuda().float()
            mag = pf.pow(2).sum(0).sqrt()
            mv = cam.flow_valid.cuda() & (mag > args.moving_px)
            if mv.any():
                prior_mag.append((mag * mv).sum().item() / mv.sum().item())
            if pkg["motion_out"] is not None:
                m_last = pkg["motion_out"]["score"].detach()
            used += 1
            if used >= args.n_views:
                break

        mean = lambda v: sum(v) / max(len(v), 1)
        print(f"\n{n_total:,} Gaussians   {used} views with a flow prior   "
              f"epsilon={eps}   moving if prior > {args.moving_px}px")

        if m_last is not None:
            m = m_last.squeeze(-1).float()
            q = [m.quantile(x).item() for x in (0.1, 0.5, 0.9, 0.99)]
            dyn = (m > eps).float().mean().item()
            print(f"\nmotion mask   dynamic={100*dyn:.2f}%  "
                  f"p10={q[0]:.2e} p50={q[1]:.2e} p90={q[2]:.2e} p99={q[3]:.3f}  "
                  f"max={m.max().item():.3f}")

        print(f"\nflow end-point error (pixels at {cams[0].flow_orig_size if used else '?'}):")
        print(f"  all valid pixels          {mean(epe_all):7.3f} px      <- what training logged")
        print(f"  moving pixels only        {mean(epe_mv):7.3f} px      <- the honest one")
        print(f"  coverage                  {100*mean(cov_all):7.1f} %")
        if mean(epe_all) > 0:
            print(f"  ratio moving/all         {mean(epe_mv)/mean(epe_all):7.1f} x")

        pm = mean(prior_mag)
        print(f"\nhow much of the real motion was reproduced:")
        print(f"  mean prior motion         {pm:7.3f} px   (what the video actually moves)")
        print(f"  frozen-subject baseline   {pm:7.3f} px   (epe_moving if nothing moved at all)")
        print(f"  our epe_moving            {mean(epe_mv):7.3f} px")
        if pm > 0:
            cap = 100 * (1 - mean(epe_mv) / pm)
            print(f"  motion captured           {cap:7.1f} %   (0% = frozen, 100% = perfect)")


if __name__ == "__main__":
    main()
