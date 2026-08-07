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

    # ---- Flow-aware auto-decoder ----
    # m_i = sigmoid(MLP_flow(f_flow)) where f_flow is queried from a dedicated
    # Flow-HexPlane. Deliberately lower capacity than the appearance planes:
    # a velocity field is much lower-frequency in space than radiance, and this
    # keeps the extra VRAM at roughly a tenth of the appearance grid.
    flow_mask = True,
    flow_kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 8,
     'resolution': [64, 64, 64, 150]
    },
    flow_multires = [1,2,4],
    flow_net_width = 64,
    flow_net_depth = 2,
    flow_velocity_head = True,
    flow_mask_init_bias = 2.0,

    # ---- losses ----
    # L_flow replaces L_bind: the mask is now bound to real image-space motion
    # instead of to the raw amplitude of dx.
    # L_flow is in normalized image units (1.0 = full image side). The scene is
    # mostly static, so the mean over pixels is small -- watch fine/flow/epe_px
    # rather than the loss value when tuning this.
    lambda_flow = 10.0,
    lambda_flow_velocity = 0.01,
    flow_from_iter = 3000,     # == mask_warmup_iters: flow switches on with the mask
    flow_interval = 1,         # raise to 2-4 if the extra pass is too slow
    flow_alpha_threshold = 0.5,

    lambda_motion_sparse = 0.01,
    lambda_motion_bind = 0.0,  # superseded by L_flow -- set >0 only for the ablation
    lambda_motion_smooth = 0.01,
    motion_bind_gamma = 10.0,
    motion_mask_epsilon = 0.05,
    motion_smooth_knn = 8,
    motion_smooth_sample = 4096,

    mask_warmup_iters = 3000,  # fine iters 0-3000: m_i frozen to 1, learn the motion field
    freeze_grid_from_iter = 0, # set to e.g. 9000 to freeze the appearance HexPlane
                               # and let only the flow branch + MLP decoders learn
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
