"""Precompute the optical-flow prior used by the flow-consistency loss.

Runs a pretrained RAFT (torchvision) over every consecutive frame pair of every
camera folder and writes, per pair, the forward flow plus a forward-backward
consistency mask. The flow is stored in pixels of the *original* image
resolution even though it is computed at a smaller working resolution, so the
training code never has to know how it was produced.

Layout written:
    <datadir>/flow/meta.json
    <datadir>/flow/<camXX>/flow_<frame:05d>.npz   keys: flow [2,h,w] f16, valid [h,w] bool

`frame` is the 0-based index of the first frame of the pair, matching
multipleview_dataset's frame indexing.

Example:
    python scripts/precompute_flow.py \
        --datadir data/multipleview/backpack_frame0_v2_upz \
        --max_side 512 --batch 4 --device cuda:0
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.optical_flow import (Raft_Large_Weights,
                                             Raft_Small_Weights, raft_large,
                                             raft_small)

IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")


def round8(x):
    return max(8, int(round(x / 8.0)) * 8)


class PairDataset(Dataset):
    """Yields (img_t, img_t+gap) already resized to the working resolution."""

    def __init__(self, paths, gap, size):
        self.paths = paths
        self.gap = gap
        self.size = size  # (w, h)

    def __len__(self):
        return len(self.paths) - self.gap

    def _load(self, path):
        img = Image.open(path).convert("RGB").resize(self.size, Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        return arr * 2.0 - 1.0  # RAFT expects [-1, 1]

    def __getitem__(self, i):
        return self._load(self.paths[i]), self._load(self.paths[i + self.gap]), i


def warp(x, flow):
    """Sample x at p + flow. x [B,C,H,W], flow [B,2,H,W] in pixels."""
    b, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device),
                            torch.arange(w, device=x.device), indexing="ij")
    grid = torch.stack([gx, gy]).float()[None] + flow
    grid_x = 2.0 * grid[:, 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid[:, 1] / max(h - 1, 1) - 1.0
    vgrid = torch.stack([grid_x, grid_y], dim=-1)
    sampled = F.grid_sample(x, vgrid, mode="bilinear", padding_mode="zeros",
                            align_corners=True)
    inside = (grid_x.abs() <= 1.0) & (grid_y.abs() <= 1.0)
    return sampled, inside


def consistency_mask(flow_fwd, flow_bwd, alpha=0.01, beta=0.5):
    """Standard forward-backward check: |F_f + F_b(p+F_f)| small enough."""
    warped_bwd, inside = warp(flow_bwd, flow_fwd)
    diff = (flow_fwd + warped_bwd).pow(2).sum(1)
    thresh = alpha * (flow_fwd.pow(2).sum(1) + warped_bwd.pow(2).sum(1)) + beta
    return (diff <= thresh) & inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True,
                    help="scene folder containing camXX/ image folders")
    ap.add_argument("--out", default=None, help="default <datadir>/flow")
    ap.add_argument("--cam_glob", default="cam*")
    ap.add_argument("--gap", type=int, default=1,
                    help="frame gap of each pair; >1 gives a stronger signal on slow motion")
    ap.add_argument("--max_side", type=int, default=512,
                    help="working resolution (longest side); also the stored resolution")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="raft_large", choices=["raft_large", "raft_small"])
    ap.add_argument("--no_fwd_bwd", action="store_true",
                    help="skip the backward pass (2x faster, no occlusion mask)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs per camera")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = args.out or os.path.join(args.datadir, "flow")
    cam_dirs = sorted(d for d in glob.glob(os.path.join(args.datadir, args.cam_glob))
                      if os.path.isdir(d))
    if not cam_dirs:
        raise SystemExit(f"no camera folders matching {args.cam_glob} under {args.datadir}")

    device = torch.device(args.device)
    if args.model == "raft_large":
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True)
    else:
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True)
    model = model.to(device).eval()

    # working resolution, derived from the first image and rounded for RAFT
    probe = sorted(f for f in os.listdir(cam_dirs[0]) if f.endswith(IMG_EXT))[0]
    w_orig, h_orig = Image.open(os.path.join(cam_dirs[0], probe)).size
    scale = args.max_side / float(max(w_orig, h_orig))
    w_work, h_work = round8(w_orig * scale), round8(h_orig * scale)
    sx, sy = w_orig / float(w_work), h_orig / float(h_work)
    print(f"original {w_orig}x{h_orig} -> working {w_work}x{h_work} "
          f"(flow rescaled by {sx:.3f}, {sy:.3f} back to original pixels)")

    os.makedirs(out_root, exist_ok=True)
    total = 0
    for cam_dir in cam_dirs:
        cam_name = os.path.basename(cam_dir)
        paths = sorted(os.path.join(cam_dir, f) for f in os.listdir(cam_dir)
                       if f.endswith(IMG_EXT))
        if len(paths) <= args.gap:
            print(f"{cam_name}: not enough frames, skipped")
            continue
        cam_out = os.path.join(out_root, cam_name)
        os.makedirs(cam_out, exist_ok=True)

        ds = PairDataset(paths, args.gap, (w_work, h_work))
        n_pairs = len(ds) if args.limit <= 0 else min(len(ds), args.limit)
        loader = DataLoader(torch.utils.data.Subset(ds, range(n_pairs)),
                            batch_size=args.batch, num_workers=args.workers,
                            shuffle=False, pin_memory=True)

        written = 0
        for img1, img2, idx in loader:
            keep = [k for k in range(len(idx))
                    if args.overwrite or not os.path.isfile(
                        os.path.join(cam_out, f"flow_{int(idx[k]):05d}.npz"))]
            if not keep:
                continue
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            with torch.no_grad():
                flow_fwd = model(img1, img2)[-1]
                if args.no_fwd_bwd:
                    valid = torch.ones_like(flow_fwd[:, 0], dtype=torch.bool)
                else:
                    flow_bwd = model(img2, img1)[-1]
                    valid = consistency_mask(flow_fwd, flow_bwd)
            # working-resolution pixels -> original-resolution pixels
            flow_fwd = flow_fwd * torch.tensor([sx, sy], device=device).view(1, 2, 1, 1)
            flow_np = flow_fwd.cpu().numpy().astype(np.float16)
            valid_np = valid.cpu().numpy()
            for k in keep:
                np.savez(os.path.join(cam_out, f"flow_{int(idx[k]):05d}.npz"),
                         flow=flow_np[k], valid=valid_np[k])
                written += 1
            if written % (50 * args.batch) < args.batch:
                print(f"  {cam_name}: {written}/{n_pairs}", flush=True)
        total += written
        print(f"{cam_name}: wrote {written} flow maps")

    meta = {"orig_size": [w_orig, h_orig], "work_size": [w_work, h_work],
            "gap": args.gap, "model": args.model,
            "fwd_bwd": not args.no_fwd_bwd, "num_files": total}
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done: {total} flow maps under {out_root}")


if __name__ == "__main__":
    main()
