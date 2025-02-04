# SceneDreamer

## Installation


- Basic environment
```
conda create -n scenedreamer python=3.10
conda activate scenedreamer
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
pip install numpy==1.25.2
```

- Install Depth model
```
cd tools/DepthPro
pip install -e .
cd ../..
```

- Install requirements of segmentation
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

cd tools/OneFormer/oneformer/modeling/pixel_decoder/ops
bash make.sh
cd ../../../../../..
```

## Download Pretrained model

```
wget -P tools/DepthPro/checkpoints https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt 
wget -P tools/OneFormer/checkpoints https://shi-labs.com/projects/oneformer/ade20k/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth 

wget -P tools/StableDiffusion/lcm_ckpt https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
```


## Demo
```
python run.py
```
