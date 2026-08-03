# Thi nghiem 2: tang do phan giai HexPlane (spatial 64->128, temporal 150->300),
# giu nguyen MLP. Moi tham so khac giong het bn_baseline.py.
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [128, 128, 128, 300]   # <-- bien duy nhat
    },
    multires = [1,2],
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
    static_mlp=False
)
OptimizationParams = dict(
    dataloader=True,
    iterations = 15000,
    batch_size=1,
    coarse_iterations = 3000,
    densify_until_iter = 10_000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
)
