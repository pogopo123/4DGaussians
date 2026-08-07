"""Helpers for the flow-supervised motion mask.

Three pieces live here:

* projection of 3D points to NDC with the same convention as the CUDA rasterizer,
  so a rendered 2D flow can be built from two deformed point sets;
* the L1 flow-consistency loss against a precomputed optical-flow prior;
* a FlowPriorStore that reads the prior written by scripts/precompute_flow.py.

Flow is carried in *normalized* units throughout: one unit = the full image
width (x) or height (y). That makes the loss independent of the render
resolution and of the resolution the prior was computed at.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Projection
# --------------------------------------------------------------------------- #
def project_to_ndc(xyz, proj_matrix):
    """[N,3] world points -> [N,2] NDC xy, matching the rasterizer's convention.

    proj_matrix is the camera's full_proj_transform (row-vector convention, i.e.
    p_clip = [x,y,z,1] @ proj_matrix), which is exactly what is handed to the
    CUDA rasterizer as `projmatrix`.
    """
    ones = torch.ones_like(xyz[:, :1])
    p_hom = torch.cat([xyz, ones], dim=-1) @ proj_matrix
    p_w = 1.0 / (p_hom[:, 3:4] + 1e-7)
    return p_hom[:, :2] * p_w


def screen_flow(xyz_t, xyz_t2, proj_matrix):
    """Per-Gaussian 2D motion from time t to t+dt, in normalized image units.

    The rasterizer maps NDC to pixels with px = ((ndc+1)*S - 1)*0.5, so a delta
    of `d` in NDC is `d*S/2` pixels, i.e. `d/2` in units of the full image side.
    """
    ndc_t = project_to_ndc(xyz_t, proj_matrix)
    ndc_t2 = project_to_ndc(xyz_t2, proj_matrix)
    return 0.5 * (ndc_t2 - ndc_t)


# --------------------------------------------------------------------------- #
#  Loss
# --------------------------------------------------------------------------- #
def flow_consistency_loss(rendered_flow, prior_flow, prior_valid, prior_size,
                          alpha_threshold=0.5, normalize_alpha=True,
                          moving_threshold_px=1.0):
    """L1 between the rendered flow and the optical-flow prior.

    Args:
        rendered_flow: [3,H,W] from the flow rasterization pass. Channels 0/1 are
            the alpha-blended normalized flow, channel 2 is the accumulated alpha
            (the flow pass renders a constant 1 there against a black background).
        prior_flow:  [2,h,w] flow in pixels of the *original* image resolution.
        prior_valid: [h,w] bool, False where the forward-backward check failed.
        prior_size:  (W_orig, H_orig) the pixel units of prior_flow refer to.
        alpha_threshold: pixels with less accumulated alpha are ignored (nothing
            was splatted there, so the rendered flow is meaningless).
        moving_threshold_px: pixels whose prior flow is shorter than this are
            treated as background for `epe_moving`.

    Returns:
        (loss, epe_pixels, coverage, epe_moving) -- epe is reported in pixels of
        the original resolution so it is directly interpretable.

        `epe_moving` restricts the same error to pixels the prior says are
        actually moving. The plain `epe` cannot distinguish a well-reconstructed
        motion from a frozen object: static background dominates the average and
        there both flows are ~0, so a mask that switched the subject off still
        scores well. `epe_moving` is the number that catches that.
    """
    w_orig, h_orig = prior_size
    h, w = prior_flow.shape[-2:]

    # rendered flow is at render resolution -> resample onto the prior's grid.
    if rendered_flow.shape[-2:] != (h, w):
        rendered_flow = F.interpolate(rendered_flow[None], size=(h, w),
                                      mode="bilinear", align_corners=False)[0]

    alpha = rendered_flow[2:3]
    flow_r = rendered_flow[:2]
    if normalize_alpha:
        flow_r = flow_r / alpha.detach().clamp(min=1e-3)

    # prior: pixels of the original image -> normalized image units.
    scale = torch.tensor([1.0 / w_orig, 1.0 / h_orig],
                         device=flow_r.device, dtype=flow_r.dtype).view(2, 1, 1)
    flow_p = prior_flow.to(flow_r.dtype) * scale

    valid = prior_valid.to(flow_r.device) & (alpha[0] > alpha_threshold)
    n_valid = valid.sum()
    if n_valid < 1:
        zero = rendered_flow.sum() * 0.0
        return zero, zero.detach(), zero.detach(), zero.detach()

    err = (flow_r - flow_p).abs().sum(0)
    loss = (err * valid).sum() / n_valid

    with torch.no_grad():
        # end-point error back in original pixels, for logging
        diff_px = (flow_r - flow_p) * torch.tensor(
            [w_orig, h_orig], device=flow_r.device, dtype=flow_r.dtype).view(2, 1, 1)
        epe = ((diff_px.pow(2).sum(0) + 1e-12).sqrt() * valid).sum() / n_valid
        coverage = n_valid.float() / valid.numel()

        # same error, restricted to where the prior reports real motion
        prior_px = flow_p * torch.tensor(
            [w_orig, h_orig], device=flow_r.device, dtype=flow_r.dtype).view(2, 1, 1)
        moving = valid & (prior_px.pow(2).sum(0).sqrt() > moving_threshold_px)
        n_moving = moving.sum()
        epe_moving = (((diff_px.pow(2).sum(0) + 1e-12).sqrt() * moving).sum() / n_moving
                      if n_moving > 0 else epe * 0.0)

    return loss, epe, coverage, epe_moving


# --------------------------------------------------------------------------- #
#  Visualization
# --------------------------------------------------------------------------- #
def flow_to_image(flow, max_mag=None):
    """[2,H,W] flow -> [3,H,W] float RGB in [0,1] using the standard colour wheel."""
    import matplotlib.colors as mcolors

    f = flow.detach().float().cpu().numpy()
    u, v = f[0], f[1]
    mag = np.sqrt(u ** 2 + v ** 2)
    if max_mag is None:
        max_mag = max(np.percentile(mag, 99), 1e-6)
    ang = (np.arctan2(-v, -u) / np.pi + 1.0) / 2.0
    hsv = np.stack([ang, np.clip(mag / max_mag, 0, 1), np.ones_like(ang)], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


# --------------------------------------------------------------------------- #
#  Prior store
# --------------------------------------------------------------------------- #
class FlowPriorStore:
    """Reads the .npz flow priors written by scripts/precompute_flow.py.

    Layout:
        <root>/meta.json
        <root>/<cam_name>/flow_<frame:05d>.npz   (keys: flow [2,h,w] f16, valid [h,w] bool)

    `frame` is the 0-based index of the *first* frame of the pair, so the file
    describes the motion from frame -> frame+gap.
    """

    def __init__(self, root):
        self.root = root
        self.enabled = False
        self.meta = {}
        meta_path = os.path.join(root, "meta.json")
        if not os.path.isfile(meta_path):
            return
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.enabled = True
        self.orig_size = tuple(self.meta.get("orig_size", [0, 0]))
        self.gap = int(self.meta.get("gap", 1))
        print(f"[flow] prior found at {root}: "
              f"work_size={self.meta.get('work_size')} gap={self.gap} "
              f"orig_size={self.orig_size}")

    def get(self, cam_name, frame_idx):
        """-> (flow [2,h,w] float32 tensor, valid [h,w] bool tensor) or None."""
        if not self.enabled:
            return None
        path = os.path.join(self.root, cam_name, f"flow_{frame_idx:05d}.npz")
        if not os.path.isfile(path):
            return None
        with np.load(path) as data:
            flow = torch.from_numpy(data["flow"].astype(np.float32))
            valid = torch.from_numpy(data["valid"].astype(np.bool_))
        return flow, valid
