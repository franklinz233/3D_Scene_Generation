import argparse  
from pipe.cfgs import load_cfg  
from pipe.c2f_recons import Pipeline  

if __name__ == "__main__":
    cfg = load_cfg('pipe/cfgs/basic.yaml')  
    
    cfg.scene.input.rgb = "test_data/figure/castle.png"
    
    dreamer = Pipeline(cfg)  
    dreamer()