#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  tools/Fooocus/models/checkpoints \
  tools/Fooocus/models/loras \
  tools/Fooocus/models/inpaint \
  tools/Fooocus/models/prompt_expansion/fooocus_expansion \
  tools/Fooocus/models/upscale_models \
  tools/DepthPro/checkpoints \
  tools/OneFormer/checkpoints \
  tools/StableDiffusion/lcm_ckpt

# Fooocus base model
wget -O tools/Fooocus/models/checkpoints/juggernautXL_v8Rundiffusion.safetensors \
  https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors

# Fooocus LoRA model
wget -O tools/Fooocus/models/loras/sd_xl_offset_example-lora_1.0.safetensors \
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors

# Fooocus inpaint model
wget -O tools/Fooocus/models/inpaint/inpaint_v26.fooocus.patch \
  'https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch?download=true'

# Fooocus prompt expansion
wget -O tools/Fooocus/models/prompt_expansion/fooocus_expansion/pytorch_model.bin \
  'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin?download=true'

# Fooocus upscale model
wget -O tools/Fooocus/models/upscale_models/fooocus_upscaler_s409985e5.bin \
  'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin?download=true'

# Depth Pro
wget -O tools/DepthPro/checkpoints/depth_pro.pt \
  https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt

# OneFormer
wget -O tools/OneFormer/checkpoints/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth \
  https://shi-labs.com/projects/oneformer/ade20k/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth

# Stable Diffusion LCM LoRA
# The SD 1.5 base model is downloaded automatically from Hugging Face on first use.
wget -O tools/StableDiffusion/lcm_ckpt/pytorch_lora_weights.safetensors \
  https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
