ModelHiddenParams = dict(
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

    # Motion Mask Head
    motion_mask = True,
    mask_warmup_iters = 3000,      # fine iters 0-3000: m_i frozen to 1
    lambda_motion_sparse = 0.01,
    lambda_motion_bind = 0.1,
    lambda_motion_smooth = 0.01,
    motion_bind_gamma = 10.0,
    motion_mask_epsilon = 0.05,
    motion_smooth_knn = 8,
    motion_smooth_sample = 4096
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
