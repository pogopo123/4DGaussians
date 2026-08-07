#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.render_process=False
        self.add_points=False
        self.extension=".png"
        self.llffhold=8
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")
class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64 # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4 # useless
        self.defor_depth = 1 # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10 # useless
        self.scale_rotation_pe = 2 # useless
        self.opacity_pe = 2 # useless
        self.timenet_width = 64 # useless
        self.timenet_output = 32 # useless
        self.bounds = 1.6 
        self.plane_tv_weight = 0.0001 # TV loss of spatial grid
        self.time_smoothness_weight = 0.01 # TV loss of temporal grid
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }
        self.multires = [1, 2, 4, 8] # multi resolution of voxel grid
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_ds=False # cancel the deformation of Gaussians' scaling
        self.no_dr=False # cancel the deformation of Gaussians' rotations
        self.no_do=True # cancel the deformation of Gaussians' opacity
        self.no_dshs=True # cancel the deformation of SH colors.
        self.empty_voxel=False # useless
        self.grid_pe=0 # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp=False # useless
        self.apply_rotation=False # useless

        # ---- Motion Mask Head (dynamic/static separation) ----
        self.motion_mask = False # enable the motion mask head (m_i = sigmoid(W_m h_i + b_m)) gating the deformation deltas
        self.mask_warmup_iters = 3000 # fine-stage iterations with m_i frozen to 1 (warm-up stage of the 3-stage schedule)
        self.lambda_motion_sparse = 0.01 # weight of L_sparse = mean(m_i)
        self.lambda_motion_bind = 0.1 # weight of L_bind = mean|m_i - tanh(gamma*||dx_i||)|
        self.lambda_motion_smooth = 0.01 # weight of KNN spatial-consistency loss on m_i
        self.motion_bind_gamma = 10.0 # gamma inside tanh of L_bind
        self.motion_mask_epsilon = 0.05 # threshold for a point to count as dynamic (logging / inference skip)
        self.motion_smooth_knn = 8 # neighbours per point for L_smooth
        self.motion_smooth_sample = 4096 # points sampled per iteration for L_smooth

        # ---- Flow-aware auto-decoder (Flow-HexPlane driving the motion mask) ----
        self.flow_mask = False # m_i = sigmoid(MLP_flow(f_flow)) from a dedicated Flow-HexPlane (implies motion_mask)
        self.flow_kplanes_config = {} # plane config of the Flow-HexPlane; empty -> reuse kplanes_config
        self.flow_multires = [] # multi-resolution of the Flow-HexPlane; empty -> reuse multires
        self.flow_net_width = 64 # width of the flow decoder MLP (velocity head / pure-flow mask ablation)
        self.flow_net_depth = 2 # depth of that MLP
        self.flow_velocity_head = True # extra head decoding an explicit 3D velocity from f_flow
        self.flow_mask_init_bias = 2.0 # bias of the mask head at init (sigmoid(2.0)~0.88)
        # Both HexPlanes are queried directly at (x,y,z,t); there is no Fourier
        # encoding of t. The Flow-HexPlane is supervised straight from the
        # optical-flow prior, so the extra frequencies are redundant and only
        # widen the vector the cross-attention has to digest.

        # ---- Cross-Attention fusion (f_fused = LN(f_deform + MHA(Q=f_deform, K=V=f_flow))) ----
        self.flow_attn = False # merge the geometry and velocity branches with cross-attention
        self.flow_attn_heads = 4 # attention heads; must divide net_width
        self.flow_attn_tokens = "multires" # "multires": one Key/Value token per HexPlane scale, so the
                                           # softmax actually selects a motion scale. "single": the
                                           # collapsed reference form, where a length-1 softmax makes
                                           # the block a fixed linear projection of f_flow.
        self.flow_attn_backbone_depth = 2 # depth of the shared MLP backbone on f_fused
        self.flow_attn_detach = "query" # "query": detach only the attention Query, so L_rgb keeps
                                        # training the appearance HexPlane through the residual.
                                        # "full": detach the whole f_deform (the reference code) --
                                        # only valid together with freeze_grid_from_iter > 0.
                                        # "none": no stop-gradient at all.
        self.mask_from_fused = False # True = m_i reads the fused backbone (the paper diagram).
                                     # False = m_i stays a function of f_flow alone, which is what
                                     # makes the early-exit pass worth anything: the mask can then be
                                     # thresholded before the appearance HexPlane is ever queried.
                                     # Measured at N=200k, 70% static: -62% deformation time with
                                     # False, only -15% with True. The deformation heads read the
                                     # fused feature either way, so the fusion still does its job.
        self.early_exit = False # inference: skip the deformation of points with m_i <= motion_mask_epsilon

        # ---- Flow-consistency supervision ----
        self.lambda_flow = 10.0 # weight of L_flow = ||Flow_render - Flow_prior||_1 (replaces L_bind).
                                # L_flow is in normalized image units (1.0 = full image side), so at
                                # 2048px width this is ~0.005 of loss per pixel of end-point error.
        self.lambda_flow_velocity = 0.01 # weight tying the velocity head to the realised displacement
        self.flow_from_iter = 3000 # fine-stage iteration at which L_flow switches on
        self.flow_interval = 1 # apply the flow pass every N iterations (>1 trades supervision for speed)
        self.flow_alpha_threshold = 0.5 # ignore pixels where less than this much alpha accumulated
        self.flow_normalize_alpha = True # divide the blended flow by the accumulated alpha
        self.freeze_grid_from_iter = 0 # fine-stage iteration from which the appearance HexPlane is frozen (0 = never)


        super().__init__(parser, "ModelHiddenParams")
        
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.dataloader=False
        self.zerostamp_init=False
        self.custom_sampler=None
        self.iterations = 30_000
        self.coarse_iterations = 3000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0
        self.lambda_lpips = 0
        self.weight_constraint_init= 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        self.opacity_reset_interval = 3000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.batch_size=1
        self.add_point=False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
