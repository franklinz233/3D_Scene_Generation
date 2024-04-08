import imageio
import numpy as np
from PIL import Image
from ops.gs.basic import *
from ops.sky import Sky_Seg_Tool
from ops.visual_check import Check
from pipe.reconstruct import Reconstruct_Tool

from pipe.cfgs import load_cfg
cfg = load_cfg(f'pipe/cfgs/basic.yaml')

class Base_Check_Traj():
    def __init__(self,
                 rgb_fn = f'data/sd_readingroom/color.png',
                 save_dir = './check_traj.') -> None:
        self.render_checkor = Check()
        self.sky_tool = Sky_Seg_Tool(cfg)
        self.depth_tool = Reconstruct_Tool(cfg)
        self.image = np.array(Image.open(rgb_fn))/255.
        self.save_dir = save_dir
        
    def _fake_scene(self):
        H,W = self.image.shape[0:2]
        sky = self.sky_tool(self.image)
        dpt,intrinsic,edge = self.depth_tool._ProDpt_(self.image)
        fake_frame = Frame(H=H,W=W,
                           rgb=self.image,
                           dpt=dpt,sky=sky,
                           inpaint=(~edge)&(~sky),
                           intrinsic=intrinsic,extrinsic=np.eye(4))
        self.fake_scene = Gaussian_Scene(cfg)
        self.fake_scene._add_trainable_frame(fake_frame,require_grad=False)
    
    @torch.no_grad()
    def _render_video(self):
        # render 5times frames
        H,W,intrinsic = self.fake_scene.frames[0].H,self.fake_scene.frames[0].W,deepcopy(self.fake_scene.frames[0].intrinsic)
        if H<W:
            if H>512:
                ratio = 512/H
                W,H = int(W*ratio),int(H*ratio)
                intrinsic[0:2] = intrinsic[0:2]*ratio
        else:
            if W>512:
                ratio = 512/W
                W,H = int(W*ratio),int(H*ratio)
                intrinsic[0:2] = intrinsic[0:2]*ratio
        # render
        rgbs,dpts = [],[]
        for pose in self.trajs:
            frame = Frame(H=H,W=W,
                          intrinsic=intrinsic,
                          extrinsic=np.linalg.inv(pose))
            rgb,dpt,alpha = self.fake_scene._render_RGBD(frame)
            rgb = rgb.detach().float().cpu().numpy()
            dpt = dpt.detach().float().cpu().numpy()
            dpts.append(dpt)
            rgbs.append((rgb * 255).astype(np.uint8))
        rgbs = np.stack(rgbs, axis=0)
        dpts = np.stack(dpts, axis=0)
        valid_dpts = dpts[dpts>0.]
        _min = np.percentile(valid_dpts,1)
        _max = np.percentile(valid_dpts,99)
        dpts = (dpts-_min) / (_max-_min+1e-5)
        dpts = dpts.clip(0,1)
            
        imageio.mimwrite(f'{self.save_dir}video_rgb.mp4',rgbs,fps=20)
        imageio.mimwrite(f'{self.save_dir}video_dpt.mp4',dpts,fps=20)
    
    def _traj(self):
        pass
    
    def __call__(self):
        self._fake_scene()
        self._traj()
        self._render_video()
        