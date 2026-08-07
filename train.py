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
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from utils.flow_utils import flow_consistency_loss, flow_to_image
from time import time
import copy

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def safe_add_histogram(tb_writer, tag, values, iteration, **kwargs):
    """add_histogram that cannot take training down.

    torch 1.13's bundled tensorboard calls np.greater(counts, 0, dtype=np.int32),
    which numpy >= 1.24 rejects. Fall back to a few summary scalars so the
    distribution is still observable.
    """
    try:
        tb_writer.add_histogram(tag, values, iteration, **kwargs)
    except (TypeError, ValueError):
        v = values.detach().float().flatten()
        tb_writer.add_scalar(f"{tag}/mean", v.mean().item(), iteration)
        for q in (0.1, 0.5, 0.9):
            tb_writer.add_scalar(f"{tag}/p{int(q*100)}", v.quantile(q).item(), iteration)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()


    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
                            # 
    count = 0
    grid_frozen = False
    for iteration in range(first_iter, final_iter+1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(video_cams)
                    if (count //(len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # 3-stage schedule: coarse (static 3DGS) -> warm-up (deform on, m_i frozen to 1)
        # -> fine-tune (mask head + sparsity/flow losses active)
        flow_mask_on = getattr(hyper, "flow_mask", False)
        motion_mask_on = getattr(hyper, "motion_mask", False) or flow_mask_on
        if motion_mask_on and stage == "fine":
            gaussians._deformation.deformation_net.mask_warmup = iteration <= hyper.mask_warmup_iters
        mask_warmup = motion_mask_on and getattr(
            gaussians._deformation.deformation_net, "mask_warmup", False)

        # VRAM/stability strategy: once the canonical motion field is in place,
        # freeze the appearance HexPlane and let only the Flow-HexPlane + MLP
        # decoders keep learning.
        freeze_from = getattr(hyper, "freeze_grid_from_iter", 0)
        if stage == "fine" and freeze_from > 0 and iteration >= freeze_from and not grid_frozen:
            for p in gaussians._deformation.deformation_net.grid.parameters():
                p.requires_grad_(False)
            grid_frozen = True
            print(f"\n[ITER {iteration}] appearance HexPlane frozen "
                  f"(flow branch and MLP decoders keep training)")

        # flow-consistency supervision is active after the mask warm-up
        flow_loss_on = (stage == "fine" and getattr(hyper, "lambda_flow", 0.0) > 0
                        and not mask_warmup
                        and iteration > getattr(hyper, "flow_from_iter", 0)
                        and iteration % max(1, getattr(hyper, "flow_interval", 1)) == 0)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        motion_out_list = []
        flow_pkg_list = []
        for viewpoint_cam in viewpoint_cams:
            # the flow pass costs one extra position-only deformation query plus
            # one extra rasterization, so only run it when it will be used
            cam_flow_dt, cam_flow_size = None, None
            if flow_loss_on and getattr(viewpoint_cam, "flow", None) is not None:
                cam_flow_dt = viewpoint_cam.flow_dt
                h_p, w_p = viewpoint_cam.flow.shape[-2:]
                cam_flow_size = (w_p, h_p)   # render the flow pass at the prior's resolution
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type,
                                flow_dt=cam_flow_dt, flow_size=cam_flow_size)
            if cam_flow_dt is not None and render_pkg["flow"] is not None:
                flow_pkg_list.append((render_pkg["flow"], render_pkg["disp3d"],
                                      render_pkg["motion_out"], viewpoint_cam))
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            if render_pkg.get("motion_out") is not None:
                motion_out_list.append(render_pkg["motion_out"])
        

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        

        loss = Ll1
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)

        # Motion-mask losses (fine-tune stage only, after warm-up)
        motion_stats = None
        if stage == "fine" and motion_mask_on and len(motion_out_list) > 0 and not mask_warmup:
            sparse_loss = 0.0
            bind_loss = 0.0
            for motion_out in motion_out_list:
                m = motion_out["score"]
                # L_sparse: push most points towards m=0
                sparse_loss = sparse_loss + m.mean()
                # L_bind: tie m to the actual displacement magnitude (stop-grad on dx).
                # Superseded by L_flow when the flow branch is on -- keep the weight
                # at 0 there, it binds m to raw amplitude rather than real motion.
                if hyper.lambda_motion_bind > 0 and motion_out["dx"] is not None:
                    dx_norm = motion_out["dx"].detach().norm(dim=-1, keepdim=True)
                    bind_loss = bind_loss + (m - torch.tanh(hyper.motion_bind_gamma * dx_norm)).abs().mean()
            sparse_loss = sparse_loss / len(motion_out_list)
            bind_loss = bind_loss / len(motion_out_list)
            # L_smooth: KNN spatial consistency on a random subsample
            m_last = motion_out_list[-1]["score"].squeeze(-1)
            xyz = gaussians.get_xyz.detach()
            n_pts = xyz.shape[0]
            n_sample = min(hyper.motion_smooth_sample, n_pts)
            sample_idx = torch.randperm(n_pts, device=xyz.device)[:n_sample]
            sample_xyz = xyz[sample_idx]
            sample_m = m_last[sample_idx]
            knn_idx = torch.cdist(sample_xyz, sample_xyz).topk(hyper.motion_smooth_knn + 1, largest=False).indices[:, 1:]
            smooth_loss = ((sample_m.unsqueeze(-1) - sample_m[knn_idx]) ** 2).mean()
            loss = loss + hyper.lambda_motion_sparse * sparse_loss \
                        + hyper.lambda_motion_bind * bind_loss \
                        + hyper.lambda_motion_smooth * smooth_loss
            motion_stats = {"sparse": sparse_loss, "bind": bind_loss, "smooth": smooth_loss,
                            "m": motion_out_list[-1]["score"].detach()}

        # Flow-consistency loss: the rendered 2D motion of the Gaussians between
        # t and t+dt must match the optical-flow prior extracted from the video.
        # This is what gives the Flow-HexPlane features their meaning -- a point
        # moving the wrong way pays here, forcing m_i or dx_i to correct.
        flow_stats = None
        if len(flow_pkg_list) > 0:
            flow_loss = 0.0
            vel_loss = 0.0
            epe_sum = 0.0
            cov_sum = 0.0
            epe_mv_sum = 0.0
            for flow_map, disp3d, motion_out, cam in flow_pkg_list:
                fl, epe, cov, epe_mv = flow_consistency_loss(
                    flow_map, cam.flow.cuda(), cam.flow_valid.cuda(), cam.flow_orig_size,
                    alpha_threshold=hyper.flow_alpha_threshold,
                    normalize_alpha=hyper.flow_normalize_alpha)
                flow_loss = flow_loss + fl
                epe_sum += epe.item()
                cov_sum += cov.item()
                epe_mv_sum += epe_mv.item()
                # velocity read-out: make the Flow-HexPlane store an explicit
                # velocity field, supervised by the displacement it produced
                if hyper.lambda_flow_velocity > 0 and motion_out is not None \
                        and motion_out.get("velocity") is not None and disp3d is not None:
                    vel_target = disp3d.detach() / max(cam.flow_dt, 1e-8)
                    vel_loss = vel_loss + (motion_out["velocity"] - vel_target).abs().mean()
            n_flow = len(flow_pkg_list)
            flow_loss = flow_loss / n_flow
            vel_loss = vel_loss / n_flow if torch.is_tensor(vel_loss) else 0.0
            loss = loss + hyper.lambda_flow * flow_loss
            if torch.is_tensor(vel_loss):
                loss = loss + hyper.lambda_flow_velocity * vel_loss
            flow_stats = {"flow": flow_loss, "velocity": vel_loss,
                          "epe_px": epe_sum / n_flow, "coverage": cov_sum / n_flow,
                          "epe_moving_px": epe_mv_sum / n_flow,
                          "map": flow_pkg_list[-1][0].detach(), "cam": flow_pkg_list[-1][3]}
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if tb_writer and motion_stats is not None and iteration % 100 == 0:
                m_vals = motion_stats["m"]
                # bind_loss stays a plain 0.0 float when lambda_motion_bind == 0
                as_float = lambda v: v.item() if torch.is_tensor(v) else float(v)
                tb_writer.add_scalar(f'{stage}/motion/sparse_loss', as_float(motion_stats["sparse"]), iteration)
                tb_writer.add_scalar(f'{stage}/motion/bind_loss', as_float(motion_stats["bind"]), iteration)
                tb_writer.add_scalar(f'{stage}/motion/smooth_loss', as_float(motion_stats["smooth"]), iteration)
                tb_writer.add_scalar(f'{stage}/motion/dynamic_ratio',
                                     (m_vals > hyper.motion_mask_epsilon).float().mean().item(), iteration)
                safe_add_histogram(tb_writer, f"{stage}/scene/motion_mask_histogram", m_vals, iteration, max_bins=500)
            if tb_writer and flow_stats is not None and iteration % 100 == 0:
                tb_writer.add_scalar(f'{stage}/flow/flow_loss', flow_stats["flow"].item(), iteration)
                tb_writer.add_scalar(f'{stage}/flow/lambda_x_flow_loss',
                                     hyper.lambda_flow * flow_stats["flow"].item(), iteration)
                tb_writer.add_scalar(f'{stage}/flow/epe_px', flow_stats["epe_px"], iteration)
                # the one that separates "motion reconstructed" from "object frozen":
                # plain epe_px is dominated by static background where both flows are ~0
                tb_writer.add_scalar(f'{stage}/flow/epe_moving_px', flow_stats["epe_moving_px"], iteration)
                tb_writer.add_scalar(f'{stage}/flow/coverage', flow_stats["coverage"], iteration)
                if torch.is_tensor(flow_stats["velocity"]):
                    tb_writer.add_scalar(f'{stage}/flow/velocity_loss', flow_stats["velocity"].item(), iteration)
                if iteration % 1000 == 0:
                    cam = flow_stats["cam"]
                    w_o, h_o = cam.flow_orig_size
                    rend = flow_stats["map"]
                    alpha = rend[2:3].clamp(min=1e-3)
                    rend_px = (rend[:2] / alpha) * torch.tensor(
                        [w_o, h_o], device=rend.device).view(2, 1, 1)
                    # share the colour-wheel scale so the two are comparable
                    scale = cam.flow.norm(dim=0).flatten().float().quantile(0.99).item()
                    tb_writer.add_images(f"{stage}/flow/rendered",
                                         flow_to_image(rend_px, max_mag=scale)[None], global_step=iteration)
                    tb_writer.add_images(f"{stage}/flow/prior",
                                         flow_to_image(cam.flow, max_mag=scale)[None], global_step=iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<2000000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<2000000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            safe_add_histogram(tb_writer, f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            safe_add_histogram(tb_writer, f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration, max_bins=500)
        
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
