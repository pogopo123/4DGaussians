"""Cross-Attention fusion between the appearance and the flow branch.

The appearance HexPlane answers "what does this point look like / where is it in
the canonical frame"; the Flow-HexPlane answers "how is this point moving".
This module lets the first one *query* the second:

    f_fused = LayerNorm( f_deform + MHA(Q=f_deform, K=V=f_flow_tokens) )

Tokenisation
------------
`HexPlaneField` returns the concatenation of one feature block per
multi-resolution scale, so a flow feature of dim `S * d` is really S separate
readings of the velocity field at S spatial scales.  Handing the attention that
sequence (`tokens="multires"`) is what makes the softmax do any work: with a
single token the softmax is identically 1 and the whole block collapses to
`LayerNorm(f_deform + W_o W_v f_flow)`, a fixed linear projection with no
content-dependent weighting.  `tokens="single"` keeps that collapsed form
available as an ablation.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Q from the geometry branch, K/V from the velocity branch."""

    def __init__(self, embed_dim=128, num_heads=4, tokens="multires"):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"flow_attn_heads={num_heads} must divide net_width={embed_dim}")
        self.tokens = tokens
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                         batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.reset_out_proj()

    def reset_out_proj(self):
        """Zero the output projection so the block starts as an exact identity
        (`f_fused == LayerNorm(f_deform)`).  Switching the flow branch on after
        warm-up then does not shock the deformation heads.  `out_proj` still
        receives gradient on the very first step, so it does not stay zero.

        Must be re-applied after any blanket `module.apply(xavier_init)` pass:
        `out_proj` is an `nn.Linear` subclass and would be overwritten.
        """
        nn.init.zeros_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, f_deform, f_flow_tokens, residual=None):
        """f_deform [N,C], f_flow_tokens [N,S,C] -> [N,C].

        `residual` overrides what the attention output is added to, so the
        caller can detach the query without also detaching the skip path (see
        `flow_attn_detach` in scene/deformation.py).
        """
        if residual is None:
            residual = f_deform
        attn_out, _ = self.mha(f_deform.unsqueeze(1), f_flow_tokens, f_flow_tokens,
                               need_weights=False)
        return self.norm(residual + attn_out.squeeze(1))


class GatedDeformationDecoder(nn.Module):
    """Shared backbone on f_fused feeding the mask head.

    The five deformation heads of `scene/deformation.py` (position, scale,
    rotation, opacity, SH) read the same backbone output, so this only owns the
    trunk and the mask head; the gated multiplication `Delta = m_i * delta`
    happens where the deltas are produced.
    """

    def __init__(self, width=128, depth=2, mask_init_bias=2.0):
        super().__init__()
        layers = []
        for _ in range(max(1, depth)):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.mask_head = nn.Linear(width, 1)
        self.mask_init_bias = mask_init_bias
        self.reset_mask_head()

    def reset_mask_head(self):
        # m starts near sigmoid(bias) ~ 0.88.  Weights are tiny but *not* zero:
        # with an exactly-zero weight dm/dh vanishes and no gradient would reach
        # the backbone or either HexPlane on the first steps.
        nn.init.normal_(self.mask_head.weight, std=1e-3)
        nn.init.constant_(self.mask_head.bias, float(self.mask_init_bias))

    def forward(self, f_fused, with_mask=True):
        """-> (h [N,W] for the deformation heads, m [N,1] in [0,1] or None).

        `with_mask=False` is the `mask_from_fused=False` configuration, where
        m_i comes from the flow branch alone and this head would be dead weight.
        """
        h = self.backbone(f_fused)
        return h, (torch.sigmoid(self.mask_head(h)) if with_mask else None)
