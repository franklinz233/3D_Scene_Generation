# 3D-SceneDreamer: Text-Driven 3D-Consistent Scene Generation

SceneDreamer is an single-image 3D scene generation pipeline. It expands an input RGB image, estimates scene text and geometry, completes novel views, optimizes a 3D Gaussian scene, and refines the result with a multi-view diffusion prior.

The repository combines project orchestration code with several external model/tool integrations:

- `scenedreamer/engine/`: the orchestration layer for the end-to-end pipeline.
- `scenedreamer/stages/`: RGB completion, depth reconstruction, and multi-view refinement stages.
- `scenedreamer/integrations/`: wrappers around external model/tool code.
- `scenedreamer/gaussian/`, `scenedreamer/cameras/`, `scenedreamer/geometry/`: 3DGS, camera-path, and geometry utilities.
- `tools/`: external integrations such as Depth Pro, OneFormer, Fooocus, and Stable Diffusion helpers.
- `configs/`: runnable project configs.
- `test_data/figure/`: small example inputs for smoke runs.

## Requirements

- Linux or WSL2 is recommended. Native Windows may need extra setup for `bash make.sh`, CUDA extension builds, and `wget`.
- NVIDIA GPU with CUDA. The full pipeline is currently designed for CUDA execution.
- Conda or Mamba.
- Python 3.10.
- PyTorch 2.1.0 with CUDA 12.1.

## Installation

The recommended path is to create the environment from `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate scenedreamer
```

Manual setup is also supported:

```bash
conda create -n scenedreamer python=3.10
conda activate scenedreamer
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Install Depth Pro:

```bash
cd tools/DepthPro
pip install -e .
cd ../..
```

Install Detectron2 and build the OneFormer CUDA op:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

cd tools/OneFormer/oneformer/modeling/pixel_decoder/ops
bash make.sh
cd ../../../../../..
```

## Model Weights

Download the required checkpoints:

```bash
bash download.sh
```

The script downloads:

- Fooocus base, LoRA, inpaint, prompt expansion, and upscale models.
- Apple Depth Pro checkpoint.
- OneFormer ADE20K checkpoint.
- Stable Diffusion LCM LoRA.

Stable Diffusion 1.5, LLaVA, and other Hugging Face models are downloaded automatically by `transformers` or `diffusers` on first use. If your network is restricted, configure the Hugging Face cache or mirror first, or download the models manually and update `configs/default.yaml`.

## Usage

Show CLI options:

```bash
python -m scenedreamer --help
```

Run the bundled example:

```bash
python -m scenedreamer
```

Run with a custom input image:

```bash
python -m scenedreamer --input path/to/image.png
```

Override the input resize size:

```bash
python -m scenedreamer --input path/to/image.png --resize-long-edge 768
```

`python run.py ...` remains available as a thin compatibility entry point.

## Outputs

Outputs are written to the input image directory:

- `scene.pth`: generated Gaussian scene.
- `video_rgb.mp4`: RGB camera-path render.
- `video_dpt.mp4`: depth camera-path render.
- `temp.coarse.interval.png`: coarse-stage preview.
- `temp.refine.interval.png`: refinement-stage preview.
- `<input>.original.<ext>`: original input backup created on first run.

The pipeline resizes the input image and writes it back to the same path. On later runs it uses the `.original` backup as the resize source, which avoids repeated quality loss.

## Configuration

The default config is `configs/default.yaml`. Common fields:

- `scene.input.rgb`: input image path.
- `scene.input.resize_long_edge`: long-edge resize size.
- `scene.outpaint`: outpaint directions, extension ratio, and seed.
- `scene.traj`: trajectory type, sample count, and forward/backward motion ratios.
- `scene.gaussian.opt_iters_per_frame`: coarse-stage Gaussian optimization steps per frame.
- `scene.mcs`: multi-view refinement steps, view count, and optimization steps.
- `model.*`: checkpoint and model paths.

## Troubleshooting

If Detectron2 or the OneFormer op fails to build, verify that PyTorch, CUDA, and the compiler versions are compatible. Linux or WSL2 is the expected build environment.

If the run fails because CUDA is unavailable, use a machine with an NVIDIA GPU. CPU mode is only practical for imports or small local utility tests.

If Hugging Face downloads fail, log in and configure `HF_HOME` or `HUGGINGFACE_HUB_CACHE`, or download the model files manually and update the config paths.

Generated previews, videos, checkpoints, and resized-input backups are ignored by `.gitignore`.

## Code Notes

The project-owned implementation lives under `scenedreamer/`. External model integrations that are still needed by the pipeline live under `tools/`; check upstream licenses and dependency constraints before modifying those directories.
