"""Depth estimation and depth-map alignment helpers."""

import torch
from scenedreamer.geometry.array_ops import edge_filter
from scenedreamer.geometry.depth_blending import SmoothDepthBlender
from scenedreamer.integrations.depth_pro_adapter import DepthProEstimator


class DepthReconstructionStage():
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
        self.connector = SmoothDepthBlender()
        
    def _load_model(self):
        self.pro_dpt = DepthProEstimator(ckpt=self.cfg.model.mde.dpt_pro.ckpt,device='cpu')

    def _clear_cuda_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ProDpt_(self, rgb, intrinsic=None):
        # conduct reconstruction
        print(f'Pro_dpt[1/3] Move Pro_dpt.model to {self.device}...')
        self.pro_dpt.to(self.device)
        print('Pro_dpt[2/3] Pro_dpt Estimation...')
        f_px = intrinsic[0,0] if intrinsic is not None else None
        metric_dpt,intrinsic = self.pro_dpt(rgb,f_px)
        print('Pro_dpt[3/3] Move Pro_dpt.model to CPU...')
        self.pro_dpt.to('cpu')
        self._clear_cuda_cache()
        edge_mask = edge_filter(metric_dpt,times=0.05)
        return metric_dpt, intrinsic, edge_mask

    def _Guide_ProDpt_(self, rgb, intrinsic=None, refer_dpt=None, refer_msk=None):
        # conduct reconstruction
        print(f'Pro_dpt[1/3] Move Pro_dpt.model to {self.device}...')
        self.pro_dpt.to(self.device)
        print('Pro_dpt[2/3] Pro_dpt Estimation...')
        f_px = intrinsic[0,0] if intrinsic is not None else None
        metric_dpt,intrinsic = self.pro_dpt(rgb,f_px=f_px)
        metric_dpt_connect = self.connector._affine_dpt_to_GS(refer_dpt,metric_dpt,~refer_msk)
        print('Pro_dpt[3/3] Move Pro_dpt.model to CPU...')
        self.pro_dpt.to('cpu')
        self._clear_cuda_cache()
        edge_mask = edge_filter(metric_dpt_connect,times=0.05)
        return metric_dpt_connect, metric_dpt, intrinsic, edge_mask
    
    # ------------- TODO: Metricv2 + Guide-GeoWizard ------------------ #
