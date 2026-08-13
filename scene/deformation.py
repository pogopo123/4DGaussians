import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
from scene.flow_field import FlowField
from scene.cross_attention import CrossAttentionFusion, GatedDeformationDecoder
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))

        self.ratio=0
        # motion-mask state: mask_warmup=True freezes m_i=1 (warm-up stage);
        # train.py flips it per-iteration. Default False so eval/render uses the learned mask.
        self.mask_warmup = False
        self.motion_out = None
        # flow-driven mask: m_i comes from a dedicated Flow-HexPlane instead of
        # the appearance MLP's hidden feature.
        self.use_flow_mask = bool(getattr(self.args, "flow_mask", False))
        self.use_motion_mask = self.use_flow_mask or bool(getattr(self.args, "motion_mask", False))
        # Cross-Attention fusion of the two branches (the F3GS architecture).
        # Without it the flow branch only produces the gate and the deformation
        # heads never see the velocity feature at all.
        self.use_flow_attn = self.use_flow_mask and bool(getattr(self.args, "flow_attn", False))
        # 'query': only the attention Query is detached, so L_rgb still trains
        #          the appearance HexPlane through the residual (default).
        # 'full' : the reference implementation -- requires freezing the
        #          appearance HexPlane, otherwise it receives no gradient at all.
        # 'none' : gradients flow freely both ways.
        self.flow_attn_detach = str(getattr(self.args, "flow_attn_detach", "query"))
        self.mask_from_fused = bool(getattr(self.args, "mask_from_fused", True))
        # flow_gate=False strips the mask head: the flow branch becomes a pure
        # Key/Value source for the attention and nothing gates the deltas.
        self.use_gate = self.use_flow_mask and bool(getattr(self.args, "flow_gate", True))
        # F3GS-split: head chia theo ngữ nghĩa thuộc tính thay vì gộp chung một backbone.
        #   Flow-HexPlane -> dx, dq  (SE(3))      App-HexPlane -> ds, dopacity, dSH
        # Không có backbone/residual dùng chung, nên L_flow không còn đường về App-HexPlane.
        self.use_flow_merge = self.use_flow_mask and bool(getattr(self.args, "flow_merge", False))
        if self.use_flow_merge:
            self.use_flow_attn = False
        self.use_flow_split = self.use_flow_mask and bool(getattr(self.args, "flow_split", False))
        if self.use_flow_split:
            self.use_flow_attn = False          # attention nối hai nhánh bằng gradient
        self.split_ctx = bool(getattr(self.args, "flow_split_ctx", True))
        self.split_detach_rgb = bool(getattr(self.args, "flow_split_detach_rgb", False))
        self.split_flow_heads = str(getattr(self.args, "flow_split_heads", "xq"))
        if self.use_flow_split and self.split_detach_rgb and self.split_flow_heads != "x":
            raise ValueError(
                "flow_split_detach_rgb=True cần flow_split_heads='x': L_flow mù với phép "
                "xoay (bản đồ flow dựng từ tâm Gaussian), nên nếu dq nằm ở nhánh flow mà "
                "L_rgb bị detach thì dq không nhận gradient từ đâu cả.")
        self.split_out = None
        # Early-exit render pass: skip the deformation heads for points the mask
        # calls static. Inference only -- it would break the gradient flow of
        # L_sparse, which needs m_i for every point.
        self.early_exit = bool(getattr(self.args, "early_exit", False))
        self.motion_mask_epsilon = float(getattr(self.args, "motion_mask_epsilon", 0.05))
        if self.use_flow_mask:
            self.flow_field = FlowField(self.args, embed_dim=self.W)
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
        if self.use_flow_mask:
            self.flow_field.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))
        if self.use_motion_mask and not self.use_flow_mask:
            self.motion_head = nn.Linear(self.W, 1)
        if self.use_flow_merge:
            in_dim = self.W + sum(self.branch_dims())
            depth = max(1, int(getattr(self.args, "flow_merge_depth", 2)))
            layers = [nn.Linear(in_dim, self.W), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(self.W, self.W), nn.ReLU()]
            self.merge_mlp = nn.Sequential(*layers)
            print(f"[F3GS] flow_merge=True -- concat({in_dim}) -> MLP -> 5 head chung; "
                  f"khong tach gradient")
        if self.use_flow_split:
            fd = self.flow_field.flow_grid.feat_dim
            Wf = int(getattr(self.args, "flow_net_width", 64))
            depth = max(1, int(getattr(self.args, "flow_net_depth", 2)))
            in_dim = fd + (self.W if self.split_ctx else 0)
            layers = [nn.Linear(in_dim, Wf), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(Wf, Wf), nn.ReLU()]
            self.flow_trunk_split = nn.Sequential(*layers)
            self.flow_dx = nn.Linear(Wf, 3)     # tịnh tiến
            self.flow_dq = nn.Linear(Wf, 4)     # xoay
            print("[F3GS] flow_split=True -- Flow-HexPlane -> (dx, dq),  "
                  "App-HexPlane -> (ds, dopacity, dSH); khong co cross-attention")
        if self.use_flow_attn:
            self.fusion = CrossAttentionFusion(
                embed_dim=self.W,
                num_heads=int(getattr(self.args, "flow_attn_heads", 4)),
                tokens=str(getattr(self.args, "flow_attn_tokens", "multires")))
            self.gated_decoder = GatedDeformationDecoder(
                width=self.W,
                depth=int(getattr(self.args, "flow_attn_backbone_depth", 2)),
                mask_init_bias=float(getattr(self.args, "flow_mask_init_bias", 2.0)))
            if not self.use_gate:
                print("[F3GS] flow_gate=False -- no motion mask; the Flow-HexPlane only "
                      "supplies Key/Value to the cross-attention, deltas are ungated")

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        
        
        hidden = self.feature_out(hidden)   
 

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def motion_gate(self, hidden, rays_pts_emb, time_emb):
        """-> (h, mask, motion_score, velocity).

        `h` is the feature the deformation heads read: `hidden` on the legacy
        path, the fused feature once Cross-Attention is on.
        `mask` is the legacy multiplicative mask applied to the canonical value;
        `motion_score` is the m_i in [0,1] that gates the *deltas* -- they are
        never both active. Shared by forward_dynamic and forward_position so the
        t and t+dt passes cannot drift apart.
        """
        ones = hidden.new_ones(hidden.shape[0], 1)
        velocity = None
        if self.use_motion_mask:
            # soft gate m_i in [0,1]; multiplies the deltas (canonical is kept when m_i -> 0)
            if self.use_flow_attn:
                h, motion_score, velocity = self.fuse(hidden, rays_pts_emb, time_emb)
                if motion_score is None:            # flow_gate=False: nothing to gate with
                    return h, ones, None, velocity
                if self.mask_warmup:
                    motion_score = ones
                return h, ones, motion_score, velocity
            if self.mask_warmup:
                motion_score = ones
            elif self.use_flow_mask:
                motion_score, velocity = self.flow_field(rays_pts_emb[:, :3], time_emb[:, :1])
            else:
                motion_score = torch.sigmoid(self.motion_head(hidden))
            return hidden, ones, motion_score, velocity
        if self.args.static_mlp:
            return hidden, self.static_mlp(hidden), None, None
        if self.args.empty_voxel:
            return hidden, self.empty_voxel(rays_pts_emb[:, :3]), None, None
        return hidden, ones, None, None

    def fuse(self, hidden, rays_pts_emb, time_emb):
        """Cross-Attention fusion of the geometry and velocity branches.

            f_fused = LayerNorm( f_deform + MHA(Q=f_deform, K=V=f_flow_tokens) )
            h, m_i  = GatedDeformationDecoder(f_fused)

        -> (h, m_i, velocity). Both branches are queried directly at (xyz, t).
        """
        xyz, t = rays_pts_emb[:, :3], time_emb[:, :1]
        feat = self.flow_field.features(xyz, t)
        tokens = self.flow_field.tokens(xyz, t, feat)

        # Detaching the Query stops L_flow from steering the appearance planes
        # through the attention weights; leaving the residual attached keeps
        # L_rgb training them as before. With 'full' both are cut, which only
        # makes sense once the appearance HexPlane is frozen.
        q = hidden.detach() if self.flow_attn_detach in ("query", "full") else hidden
        res = hidden.detach() if self.flow_attn_detach == "full" else hidden
        fused = self.fusion(q, tokens, residual=res)

        if not self.use_gate:
            h, _ = self.gated_decoder(fused, with_mask=False)
            return h, None, self.flow_field.velocity(xyz, t, feat)

        h, m_fused = self.gated_decoder(fused, with_mask=self.mask_from_fused)
        if self.mask_from_fused:
            motion_score = m_fused
        else:
            # m_i stays a function of the velocity feature alone, which is what
            # lets forward_dynamic_flow_first threshold before ever touching the
            # appearance grid (see can_flow_first)
            motion_score, _ = self.flow_field(xyz, t, feat)
        velocity = self.flow_field.velocity(xyz, t, feat)
        return h, motion_score, velocity

    def forward_position(self, rays_pts_emb, time_emb):
        """Position-only deformation, used for the t+dt pass of the flow render.

        Skips the scale/rotation/opacity/SH heads, whose outputs the flow pass
        would throw away anyway.
        """
        if self.use_flow_merge:
            h = self.merge_feature(rays_pts_emb, time_emb)
            return rays_pts_emb[:, :3] if self.args.no_dx else rays_pts_emb[:, :3] + self.pos_deform(h)
        if self.use_flow_split:
            # lượt t+dt của bản đồ flow: chỉ nhánh động học nhận gradient
            _, dx, _ = self.split_branches(rays_pts_emb, time_emb)
            return rays_pts_emb[:, :3] if self.args.no_dx else rays_pts_emb[:, :3] + dx
        hidden = self.query_time(rays_pts_emb, None, None, None, time_emb)
        h, mask, motion_score, _ = self.motion_gate(hidden, rays_pts_emb, time_emb)
        if self.args.no_dx:
            return rays_pts_emb[:, :3]
        dx = self.pos_deform(h)
        if motion_score is not None:
            return rays_pts_emb[:, :3] + motion_score * dx
        return rays_pts_emb[:, :3] * mask + dx

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        if self.use_flow_merge:
            return self.forward_dynamic_merge(rays_pts_emb, scales_emb, rotations_emb,
                                              opacity_emb, shs_emb, time_emb)
        if self.use_flow_split:
            return self.forward_dynamic_split(rays_pts_emb, scales_emb, rotations_emb,
                                              opacity_emb, shs_emb, time_emb)
        # gated on grad-enabled rather than self.training: nothing in this repo
        # calls .eval(), whereas every render path runs under torch.no_grad()
        if self.early_exit and not torch.is_grad_enabled() and self.can_flow_first():
            return self.forward_dynamic_flow_first(
                rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_emb)
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        h, mask, motion_score, velocity = self.motion_gate(hidden, rays_pts_emb, time_emb)
        # breakpoint()
        if self.early_exit and not torch.is_grad_enabled() and motion_score is not None:
            return self.forward_dynamic_masked(
                h, motion_score, velocity,
                rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb)
        dx = None
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(h)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            if motion_score is not None:
                pts = rays_pts_emb[:,:3] + motion_score*dx
            else:
                pts = rays_pts_emb[:,:3]*mask + dx
        if self.args.no_ds :

            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(h)

            scales = torch.zeros_like(scales_emb[:,:3])
            if motion_score is not None:
                scales = scales_emb[:,:3] + motion_score*ds
            else:
                scales = scales_emb[:,:3]*mask + ds

        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(h)

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            elif motion_score is not None:
                rotations = rotations_emb[:,:4] + motion_score*dr
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1]
        else:
            do = self.opacity_deform(h)

            opacity = torch.zeros_like(opacity_emb[:,:1])
            if motion_score is not None:
                opacity = opacity_emb[:,:1] + motion_score*do
            else:
                opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(h).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            # breakpoint()
            if motion_score is not None:
                shs = shs_emb + motion_score.unsqueeze(-1)*dshs
            else:
                shs = shs_emb*mask.unsqueeze(-1) + dshs

        if motion_score is not None:
            self.motion_out = {"score": motion_score, "dx": dx, "velocity": velocity}

        return pts, scales, rotations, opacity, shs

    def branch_dims(self):
        """Số chiều của từng nhánh phụ nối vào MLP hợp nhất.

        Thêm một grid mới (ví dụ Depth-HexPlane) chỉ cần nối thêm feat_dim của nó
        ở đây và thêm đặc trưng tương ứng trong `branch_features`.
        """
        return [self.flow_field.flow_grid.feat_dim]

    def branch_features(self, xyz, t):
        return [self.flow_field.features(xyz, t)]

    def merge_feature(self, rays_pts_emb, time_emb):
        """h = MLP( concat(f_app, f_flow, ...) ) -- đặc trưng chung cho cả 5 head."""
        xyz, t = rays_pts_emb[:, :3], time_emb[:, :1]
        hidden = self.query_time(rays_pts_emb, None, None, None, time_emb)
        return self.merge_mlp(torch.cat([hidden] + self.branch_features(xyz, t), -1))

    def forward_dynamic_merge(self, rays_pts_emb, scales_emb, rotations_emb,
                              opacity_emb, shs_emb, time_emb):
        h = self.merge_feature(rays_pts_emb, time_emb)
        xyz = rays_pts_emb[:, :3]
        pts = xyz if self.args.no_dx else xyz + self.pos_deform(h)
        scales = (scales_emb[:, :3] if self.args.no_ds
                  else scales_emb[:, :3] + self.scales_deform(h))
        rotations = (rotations_emb[:, :4] if self.args.no_dr
                     else rotations_emb[:, :4] + self.rotations_deform(h))
        opacity = (opacity_emb[:, :1] if self.args.no_do
                   else opacity_emb[:, :1] + self.opacity_deform(h))
        shs = (shs_emb if self.args.no_dshs
               else shs_emb + self.shs_deform(h).reshape([shs_emb.shape[0], 16, 3]))
        self.motion_out = None
        self.split_out = None
        return pts, scales, rotations, opacity, shs

    def split_branches(self, rays_pts_emb, time_emb):
        """-> (hidden, dx, dq) cho nhánh split.

        `hidden` là f_app (dùng cho các head diện mạo); `dx`, `dq` sinh hoàn toàn
        từ Flow-HexPlane, chỉ đọc thêm sg(f_app) làm ngữ cảnh hình học nên không
        có gradient nào chảy ngược về App-HexPlane qua đường đó.
        """
        xyz, t = rays_pts_emb[:, :3], time_emb[:, :1]
        hidden = self.query_time(rays_pts_emb, None, None, None, time_emb)
        feat = self.flow_field.features(xyz, t)
        inp = torch.cat([feat, hidden.detach()], -1) if self.split_ctx else feat
        h = self.flow_trunk_split(inp)
        return hidden, self.flow_dx(h), self.flow_dq(h)

    def forward_dynamic_split(self, rays_pts_emb, scales_emb, rotations_emb,
                              opacity_emb, shs_emb, time_emb):
        hidden, dx, dq = self.split_branches(rays_pts_emb, time_emb)
        xyz = rays_pts_emb[:, :3]

        # lượt RGB: các head diện mạo mở gradient, phần động học tuỳ cờ
        dx_r = dx.detach() if self.split_detach_rgb else dx

        pts = xyz if self.args.no_dx else xyz + dx_r
        if self.args.no_dr:
            rotations = rotations_emb[:, :4]
        elif self.split_flow_heads == "x":
            # dq về nhánh app: chỉ L_rgb quan sát được phép xoay
            rotations = rotations_emb[:, :4] + self.rotations_deform(hidden)
        else:
            rotations = rotations_emb[:, :4] + (dq.detach() if self.split_detach_rgb else dq)
        scales = (scales_emb[:, :3] if self.args.no_ds
                  else scales_emb[:, :3] + self.scales_deform(hidden))
        opacity = (opacity_emb[:, :1] if self.args.no_do
                   else opacity_emb[:, :1] + self.opacity_deform(hidden))
        shs = (shs_emb if self.args.no_dshs
               else shs_emb + self.shs_deform(hidden).reshape([shs_emb.shape[0], 16, 3]))

        # lượt FLOW dùng vị trí này: động học mở gradient, phần còn lại không liên quan
        self.split_out = {"dx": dx, "dq": dq, "pos_flow": xyz + dx}
        self.motion_out = None
        return pts, scales, rotations, opacity, shs

    def can_flow_first(self):
        """Whether the mask can be evaluated without touching the appearance grid.

        Only when m_i is a function of f_flow alone (`mask_from_fused=False`).
        With the mask on the fused feature the appearance HexPlane has to be
        queried for every point just to decide who is static, which is where
        almost all of the deformation cost sits -- the head MLPs the plain
        early-exit skips are only a couple of percent of it.
        """
        return (self.use_flow_attn and self.use_gate and not self.mask_from_fused
                and not self.mask_warmup and not self.args.no_dx)

    def forward_dynamic_flow_first(self, rays_pts_emb, scales_emb, rotations_emb,
                                   opacity_emb, shs_emb, time_emb):
        """Early-exit that also skips the appearance grid.

        Order matters: query only the (small) Flow-HexPlane, threshold m_i, then
        run the appearance HexPlane + cross-attention + heads on the dynamic
        subset alone. Static points never touch the expensive grid at all and
        come out bit-for-bit canonical.
        """
        xyz, t = rays_pts_emb[:, :3], time_emb[:, :1]
        feat = self.flow_field.features(xyz, t)
        motion_score, velocity = self.flow_field(xyz, t, feat)
        idx = (motion_score.squeeze(-1) > self.motion_mask_epsilon).nonzero(as_tuple=True)[0]

        pts = xyz.clone()
        scales = scales_emb[:, :3].clone()
        rotations = rotations_emb[:, :4].clone()
        opacity = opacity_emb[:, :1].clone()
        shs = shs_emb.clone()

        if idx.numel() > 0:
            sub_pts_emb, sub_t = rays_pts_emb[idx], time_emb[idx]
            hidden = self.query_time(sub_pts_emb, None, None, None, sub_t)
            tokens = self.flow_field.tokens(xyz[idx], t[idx], feat[idx])
            h, _ = self.gated_decoder(self.fusion(hidden, tokens), with_mask=False)
            ms = motion_score[idx]

            pts[idx] = pts[idx] + ms * self.pos_deform(h)
            if not self.args.no_ds:
                scales[idx] = scales[idx] + ms * self.scales_deform(h)
            if not self.args.no_dr:
                dr = self.rotations_deform(h)
                if self.args.apply_rotation:
                    rotations[idx] = batch_quaternion_multiply(rotations_emb[idx], dr)
                else:
                    rotations[idx] = rotations[idx] + ms * dr
            if not self.args.no_do:
                opacity[idx] = opacity[idx] + ms * self.opacity_deform(h)
            if not self.args.no_dshs:
                shs[idx] = shs[idx] + ms.unsqueeze(-1) * \
                    self.shs_deform(h).reshape([idx.shape[0], 16, 3])

        self.motion_out = {"score": motion_score, "dx": None, "velocity": velocity,
                           "dynamic_idx": idx}
        return pts, scales, rotations, opacity, shs

    def forward_dynamic_masked(self, h, motion_score, velocity, rays_pts_emb,
                               scales_emb, rotations_emb, opacity_emb, shs_emb):
        """Early-exit pass: run the deformation heads on dynamic points only.

        The mask itself still has to be evaluated everywhere (it decides who is
        dynamic), but the five per-attribute head MLPs -- which dominate the
        cost -- only see the m_i > epsilon subset. Points below the threshold
        keep their canonical values bit-for-bit, which is also what removes the
        residual jitter of the static background.

        Inference only: L_sparse needs m_i to reach every point, so training
        always takes the dense path.
        """
        idx = (motion_score.squeeze(-1) > self.motion_mask_epsilon).nonzero(as_tuple=True)[0]

        pts = rays_pts_emb[:, :3].clone()
        scales = scales_emb[:, :3].clone()
        rotations = rotations_emb[:, :4].clone()
        opacity = opacity_emb[:, :1].clone()
        shs = shs_emb.clone()

        if idx.numel() > 0:
            hs = h[idx]
            ms = motion_score[idx]
            if not self.args.no_dx:
                pts[idx] = pts[idx] + ms * self.pos_deform(hs)
            if not self.args.no_ds:
                scales[idx] = scales[idx] + ms * self.scales_deform(hs)
            if not self.args.no_dr:
                dr = self.rotations_deform(hs)
                if self.args.apply_rotation:
                    rotations[idx] = batch_quaternion_multiply(rotations_emb[idx], dr)
                else:
                    rotations[idx] = rotations[idx] + ms * dr
            if not self.args.no_do:
                opacity[idx] = opacity[idx] + ms * self.opacity_deform(hs)
            if not self.args.no_dshs:
                dshs = self.shs_deform(hs).reshape([idx.shape[0], 16, 3])
                shs[idx] = shs[idx] + ms.unsqueeze(-1) * dshs

        self.motion_out = {"score": motion_score, "dx": None, "velocity": velocity,
                           "dynamic_idx": idx}
        return pts, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # re-init the mask heads *after* the generic xavier pass, which would
        # otherwise overwrite them: start the mask near 1 (sigmoid(2.0)~0.88) so
        # switching it on right after warm-up does not shock the deformation field
        if self.deformation_net.use_flow_mask:
            self.deformation_net.flow_field.reset_heads()
            if self.deformation_net.use_flow_attn:
                # both must be redone: out_proj is an nn.Linear subclass and the
                # fused mask head was just xavier'd away from its sigmoid(2.0) bias
                self.deformation_net.fusion.reset_out_proj()
                self.deformation_net.gated_decoder.reset_mask_head()
        elif self.deformation_net.use_motion_mask:
            init.zeros_(self.deformation_net.motion_head.weight)
            init.constant_(self.deformation_net.motion_head.bias, 2.0)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs
    def forward_position(self, point, times_sel):
        """Deformed positions only -- the cheap second pass of the flow render."""
        point_emb = poc_fre(point, self.pos_poc)
        return self.deformation_net.forward_position(point_emb, times_sel)
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb