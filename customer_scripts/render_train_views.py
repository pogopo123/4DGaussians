"""Render the training views of a multipleview scene, streaming to disk.

render.py's render_set() buffers every rendered frame (as a CUDA tensor) before
writing, which needs hundreds of GB for a 5632-view training set. This script
writes each frame as soon as it is rendered and encodes one mp4 per camera.

Usage:
    python customer_scripts/render_train_views.py --model_path output/... \
        --configs arguments/multipleview/default.py [--stride N] [--save_gt]
"""
import os
import sys
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
import torchvision
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state


def main():
    parser = ArgumentParser(description="Stream-render training views")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--stride", default=1, type=int, help="render every Nth frame per camera")
    parser.add_argument("--save_gt", action="store_true", help="also dump ground-truth pngs")
    parser.add_argument("--fps", default=30, type=int)
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    safe_state(args.quiet)

    dataset = model.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam.extract(args))
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        views = scene.getTrainCameras()
        cam_type = scene.dataset_type
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                          dtype=torch.float32, device="cuda")
        pipe = pipeline.extract(args)

        out_root = os.path.join(dataset.model_path, "train", "ours_%d" % scene.loaded_iter)
        render_dir = os.path.join(out_root, "renders")
        gt_dir = os.path.join(out_root, "gt")
        os.makedirs(render_dir, exist_ok=True)
        if args.save_gt:
            os.makedirs(gt_dir, exist_ok=True)
        print("point nums:", gaussians._xyz.shape[0])

        # training views are laid out camera-major: all frames of cam01, then cam02, ...
        paths = scene.train_camera.dataset.image_paths
        n_cams = len({os.path.dirname(p) for p in paths})
        per_cam = len(views) // n_cams
        print("cameras: %d, frames per camera: %d" % (n_cams, per_cam))

        writers = {}
        for idx in tqdm(range(0, len(views), 1), desc="Rendering"):
            cam_id, frame_id = idx // per_cam, idx % per_cam
            if frame_id % args.stride:
                continue
            view = views[idx]
            image = render(view, gaussians, pipe, bg, cam_type=cam_type)["render"]
            name = "cam%02d_%05d.png" % (cam_id + 1, frame_id)
            torchvision.utils.save_image(image, os.path.join(render_dir, name))
            if args.save_gt:
                torchvision.utils.save_image(view.original_image[0:3], os.path.join(gt_dir, name))

            if cam_id not in writers:
                writers[cam_id] = imageio.get_writer(
                    os.path.join(out_root, "cam%02d.mp4" % (cam_id + 1)), fps=args.fps)
            frame = (255 * np.clip(image.cpu().numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0)
            writers[cam_id].append_data(frame)
            del image, view

        for w in writers.values():
            w.close()
        print("wrote", out_root)


if __name__ == "__main__":
    main()
