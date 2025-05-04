import sys

from scenedreamer.paths import TOOLS_ROOT

DEPTH_PRO_DIR = TOOLS_ROOT / 'DepthPro'
if str(DEPTH_PRO_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_PRO_DIR))

from command_pro_dpt import apple_pro_depth


class DepthProEstimator(apple_pro_depth):
    def __init__(self, device='cuda', ckpt='/mnt/proj/SOTAs/ml-depth-pro-main/checkpoints/depth_pro.pt'):
        super().__init__(device, ckpt)

    def __call__(self, image, f_px=None):
        return super().__call__(image, f_px)
