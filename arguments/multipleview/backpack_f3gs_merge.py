"""F3GS-merge -- nối đặc trưng hai nhánh rồi để một MLP tự học phân công.

    h = MLP( concat(f_app [128], f_flow [24]) )   ->  dx, ds, dq, dopacity, dSH

Biểu đạt mạnh hơn split: split là một trường hợp riêng mà MLP có thể học ra nếu
đó thật sự là cách chia tốt nhất. Không áp đặt giả định nào về việc grid nào chi
phối thuộc tính nào.

Cái đánh đổi: cả L_rgb lẫn L_flow đều chạm cả hai grid, nên mất hoàn toàn khả
năng tách gradient mà bản split đạt được (L_flow -> App-HexPlane = 0%).

Điểm cắm cho grid thứ ba (ví dụ Depth-HexPlane): thêm feat_dim vào
Deformation.branch_dims() và đặc trưng tương ứng vào branch_features().
"""

ModelHiddenParams = dict(
    # ---- appearance HexPlane (unchanged) ----
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4,8,16,32],
    defor_depth = 0,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False,

    # ---- Flow-HexPlane (latent velocity field) ----
    # Asymmetric resolution on purpose: a velocity field is far lower-frequency
    # in space than radiance, so fewer scales and a narrower feature keep the
    # extra VRAM at roughly a tenth of the appearance grid -- and the reduced
    # spatial capacity doubles as a natural smoothness prior on motion.
    flow_mask = True,
    flow_kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 8,
     'resolution': [64, 64, 64, 150]
    },
    flow_multires = [1,2,4],   # -> feat_dim 24 = 3 scales x 8, i.e. 3 attention tokens
    flow_net_width = 64,
    flow_net_depth = 2,
    flow_velocity_head = False,
    flow_split = False,
    flow_merge = True,
    flow_merge_depth = 2,
    flow_mask_init_bias = 2.0,

    # ---- Cross-Attention fusion ----
    # f_fused = LayerNorm( f_deform + MHA(Q=f_deform, K=V=f_flow_tokens) )
    flow_attn = False,            # split thay thế hoàn toàn
    flow_attn_heads = 4,           # must divide net_width (128 / 4 = 32)
    flow_attn_tokens = 'multires', # one token per Flow-HexPlane scale, so the softmax has
                                   # a real choice to make. 'single' reproduces the reference
                                   # code, where a length-1 softmax degenerates the block into
                                   # a fixed linear projection of f_flow.
    flow_attn_backbone_depth = 2,
    flow_attn_detach = 'query',    # detach only the Query: L_flow cannot steer the appearance
                                   # planes through the attention weights, but L_rgb still trains
                                   # them through the residual. Set to 'full' (the reference code)
                                   # ONLY together with freeze_grid_from_iter > 0 -- otherwise the
                                   # appearance HexPlane receives no gradient at all.
    mask_from_fused = False,       # Where m_i is read from, and the one place this config knowingly
                                   # departs from the paper diagram.
                                   #   True  = m_i from the fused backbone, as drawn. The mask sees
                                   #           geometry as well as motion, but deciding who is static
                                   #           then requires the App-HexPlane AND the attention for
                                   #           every point -- together ~2/3 of the deformation cost --
                                   #           so early-exit only saves the head MLPs (measured -15%).
                                   #   False = m_i from f_flow alone. The Flow-HexPlane costs 1.9 ms
                                   #           against the appearance grid's 12.0 ms, so the mask can
                                   #           be thresholded first and the appearance grid +
                                   #           attention + heads run on the dynamic subset only
                                   #           (measured -62% at 70% static, N=200k -- which lands
                                   #           28% BELOW plain 4DGS while still carrying the whole
                                   #           flow branch).
                                   # The deformation heads read f_fused in both cases, so the
                                   # cross-attention contribution is unaffected either way.
    early_exit = False,            # inference-time flag; pass --early_exit to render.py

    # ---- losses ----
    # L_flow replaces L_bind: the mask is bound to real image-space motion
    # instead of to the raw amplitude of dx.
    # L_flow is in normalized image units (1.0 = full image side). The scene is
    # mostly static, so the mean over pixels is small -- watch fine/flow/epe_px
    # rather than the loss value when tuning this.
    lambda_flow = 10.0,
    lambda_flow_velocity = 0.0,
    flow_from_iter = 3000,     # == mask_warmup_iters: flow switches on with the mask
    flow_interval = 1,         # raise to 2-4 if the extra pass is too slow
    flow_alpha_threshold = 0.5,

    lambda_motion_sparse = 0.0,
    lambda_motion_bind = 0.0,  # superseded by L_flow -- set >0 only for the ablation
    lambda_motion_smooth = 0.0,
    motion_bind_gamma = 10.0,
    motion_mask_epsilon = 0.05,
    motion_smooth_knn = 8,
    motion_smooth_sample = 4096,

    # ---- two-stage protocol ----
    # Stage 1 (fine iters 0-3000): m_i pinned to 1, learn the canonical motion field.
    # Stage 2 (3000+):             mask + cross-attention + L_flow + L_sparse active.
    mask_warmup_iters = 3000,
    freeze_grid_from_iter = 0, # set to 3000 for the paper's literal Stage 2 (freeze the
                               # appearance HexPlane); required if flow_attn_detach='full'.
)
OptimizationParams = dict(
    dataloader=True,
    iterations = 15_000,
    batch_size=1,
    coarse_iterations = 3_000,
    densify_until_iter = 10_000,
    opacity_reset_interval = 100_000,
    opacity_threshold_coarse = 0.001,
    opacity_threshold_fine_init = 0.001,
    opacity_threshold_fine_after = 0.001,
    opacity_lr = 0.02,
    # pruning_interval = 2000
)
