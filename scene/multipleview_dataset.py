import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_spiral
from utils.flow_utils import FlowPriorStore
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split
    ):
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height=cam_intrinsics[1].height
        width=cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.image_paths, self.image_poses, self.image_times= self.load_images_path(cam_folder, cam_extrinsics,cam_intrinsics,split)
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)

        # optical-flow prior (written by scripts/precompute_flow.py); silently
        # disabled when the folder is absent, so unmodified runs are unaffected.
        self.flow_store = FlowPriorStore(os.path.join(cam_folder, "flow"))
        self.flow_dt = (self.flow_store.gap / float(self.image_length)
                        if self.flow_store.enabled else None)

    def load_images_path(self, cam_folder, cam_extrinsics,cam_intrinsics,split):
        image_length = len(os.listdir(os.path.join(cam_folder,"cam01")))
        self.image_length = image_length
        #len_cam=len(cam_extrinsics)
        image_paths=[]
        image_poses=[]
        image_times=[]
        # (camera folder name, 0-based frame index) per entry, so the flow prior
        # can be looked up by dataset index.
        self.image_cams=[]
        self.image_frames=[]
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            number = os.path.basename(extr.name)[5:-4]
            cam_name="cam"+number.zfill(2)
            images_folder=os.path.join(cam_folder,cam_name)

            image_range=range(image_length)
            if split=="test":
                image_range = [image_range[0],image_range[int(image_length/3)],image_range[int(image_length*2/3)]]

            for i in image_range:
                num=i+1
                image_path=os.path.join(images_folder,"frame_"+str(num).zfill(5)+".jpg")
                image_paths.append(image_path)
                image_poses.append((R,T))
                image_times.append(float(i/image_length))
                self.image_cams.append(cam_name)
                self.image_frames.append(i)

        return image_paths, image_poses,image_times

    def get_flow(self, index):
        """-> dict(flow, valid, dt, orig_size) for this frame, or None."""
        if not self.flow_store.enabled:
            return None
        entry = self.flow_store.get(self.image_cams[index], self.image_frames[index])
        if entry is None:
            return None
        flow, valid = entry
        return {"flow": flow, "valid": valid, "dt": self.flow_dt,
                "orig_size": self.flow_store.orig_size}
    
    def get_video_cam_infos(self,datadir):
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        val_poses = get_spiral(poses, near_fars, N_views=N_views)

        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]