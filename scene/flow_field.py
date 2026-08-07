"""Flow-HexPlane: the velocity half of the disentangled dual grid.

The appearance HexPlane in scene/deformation.py answers "what does this point
look like at time t".  This one answers "how is this point moving at time t".
Both are queried *directly* at (x, y, z, t) -- there is deliberately no Fourier
encoding of t: the Flow-HexPlane is supervised straight from a real optical-flow
prior, which is already a strong motion signal, and keeping the feature vector
small keeps the cross-attention cheap and debuggable.

What it exposes:

    tokens(xyz, t) -> [N, S, W]   one token per multi-resolution scale, the
                                  Key/Value sequence of the cross-attention
    forward(xyz, t) -> (m, v)     pure-flow read-out, used for the
                                  `mask_from_fused=False` ablation and for the
                                  explicit 3D velocity head

Only the flow-consistency loss in train.py gives these features meaning, so the
planes end up storing a latent velocity field rather than an appearance field.
"""

import torch
import torch.nn as nn
import torch.nn.init as init

from scene.hexplane import HexPlaneField


class FlowField(nn.Module):
    def __init__(self, args, embed_dim):
        super().__init__()
        self.args = args

        plane_cfg = getattr(args, "flow_kplanes_config", None) or args.kplanes_config
        multires = getattr(args, "flow_multires", None) or args.multires
        # named *_grid so gaussian_model's parameter split routes it to the grid
        # learning-rate group together with the appearance planes.
        self.flow_grid = HexPlaneField(args.bounds, plane_cfg, multires)

        # HexPlaneField concatenates one block per multi-resolution scale.
        self.n_scales = len(multires)
        self.scale_dim = self.flow_grid.feat_dim // self.n_scales
        self.token_mode = str(getattr(args, "flow_attn_tokens", "multires"))

        # Project the flow feature into the attention's embedding space. In
        # "multires" mode the same projection is shared across scales, so a
        # scale token keeps its identity only through its content.
        if self.token_mode == "multires":
            self.token_proj = nn.Linear(self.scale_dim, embed_dim)
        else:
            self.token_proj = nn.Linear(self.flow_grid.feat_dim, embed_dim)

        # Small trunk on the raw flow feature, feeding the velocity read-out and
        # the pure-flow mask head. Kept separate from the fused backbone so the
        # velocity field stays a function of f_flow alone.
        self.W = int(getattr(args, "flow_net_width", 64))
        depth = max(1, int(getattr(args, "flow_net_depth", 2)))
        layers = [nn.Linear(self.flow_grid.feat_dim, self.W), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(self.W, self.W), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)

        self.mask_head = nn.Linear(self.W, 1)
        self.velocity_head = (nn.Linear(self.W, 3)
                              if getattr(args, "flow_velocity_head", False) else None)

        self.reset_heads()

    def reset_heads(self):
        # Start with m ~= sigmoid(bias): near 1, so switching the mask on after
        # warm-up (where m is pinned to 1) does not shock the deformation field.
        # The weights are tiny but *not* zero -- with an exactly-zero weight
        # dm/dh vanishes and neither the trunk nor the Flow-HexPlane would see
        # any gradient on the first steps.
        init.normal_(self.mask_head.weight, std=1e-3)
        init.constant_(self.mask_head.bias,
                       float(getattr(self.args, "flow_mask_init_bias", 2.0)))
        if self.velocity_head is not None:
            init.normal_(self.velocity_head.weight, std=1e-3)
            init.zeros_(self.velocity_head.bias)

    def set_aabb(self, xyz_max, xyz_min):
        self.flow_grid.set_aabb(xyz_max, xyz_min)

    def features(self, xyz, t):
        """Raw Flow-HexPlane feature, [N, feat_dim]."""
        feat = self.flow_grid(xyz, t)
        if feat.dim() == 1:                       # grid_sample_wrapper squeezes
            feat = feat.view(xyz.shape[0], -1)
        return feat

    def tokens(self, xyz, t, feat=None):
        """Key/Value sequence for the cross-attention, [N, S, embed_dim].

        S = number of multi-resolution scales in "multires" mode, 1 in "single"
        mode (the collapsed form of the reference implementation).
        """
        if feat is None:
            feat = self.features(xyz, t)
        if self.token_mode == "multires":
            feat = feat.view(feat.shape[0], self.n_scales, self.scale_dim)
        else:
            feat = feat.unsqueeze(1)
        return self.token_proj(feat)

    def forward(self, xyz, t, feat=None):
        """Pure-flow read-out -> (m [N,1] in [0,1], v [N,3] or None)."""
        if feat is None:
            feat = self.features(xyz, t)
        h = self.trunk(feat)
        m = torch.sigmoid(self.mask_head(h))
        v = self.velocity_head(h) if self.velocity_head is not None else None
        return m, v

    def velocity(self, xyz, t, feat=None):
        """Explicit 3D velocity only, [N,3] or None."""
        if self.velocity_head is None:
            return None
        if feat is None:
            feat = self.features(xyz, t)
        return self.velocity_head(self.trunk(feat))
