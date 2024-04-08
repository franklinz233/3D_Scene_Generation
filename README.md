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
wget -P tools/Fooocus/models/checkpoints/ https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors 
wget -P tools/Fooocus/models/loras/ https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors 

wget -P tools/Fooocus/models/inpaint/ https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch?download=true
mv tools/Fooocus/models/inpaint/inpaint_v26.fooocus.patch?download=true tools/Fooocus/models/inpaint/inpaint_v26.fooocus.patch

wget -P tools/Fooocus/models/prompt_expansion/fooocus_expansion/ https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin?download=true
mv tools/Fooocus/models/prompt_expansion/fooocus_expansion/fooocus_expansion.bin?download=true tools/Fooocus/models/prompt_expansion/fooocus_expansion/pytorch_model.bin

wget -P tools/Fooocus/models/unscale_models/ https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin?download=true
mv tools/Fooocus/models/unscale_models/fooocus_upscaler_s409985e5.bin?download=true tools/Fooocus/models/unscale_models/fooocus_upscaler_s409985e5.bin

wget -P tools/DepthPro/checkpoints https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt 
wget -P tools/OneFormer/checkpoints https://shi-labs.com/projects/oneformer/ade20k/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth 

wget -P tools/StableDiffusion/lcm_ckpt https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
```


## Demo
```
python run.py
```
