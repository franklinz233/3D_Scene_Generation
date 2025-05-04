import sys

from scenedreamer.paths import TOOLS_ROOT

ONEFORMER_DIR = TOOLS_ROOT / 'OneFormer'
if str(ONEFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(ONEFORMER_DIR))

from oneformer_command import OneFormer_Segment


class SkySegmenter():
    def __init__(self,cfg):
        self.ckpt = cfg.model.sky.oneformer.ckpt
        self.yaml = cfg.model.sky.oneformer.yaml
        # ckpt='/mnt/proj/GaussianAnythingV2/Tools/OneFormer/ckpts/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth',
        # yaml='/mnt/proj/GaussianAnythingV2/Tools/OneFormer/configs/ade20k/dinat/coco_pretrain_oneformer_dinat_large_bs16_160k_1280x1280.yaml') -> None:
        self.segor = OneFormer_Segment(self.ckpt,self.yaml)

    def __call__(self, rgb):
        '''
        input rgb should be numpy in range of 0-1 or 0-255
        '''
        return self.segor(rgb)
