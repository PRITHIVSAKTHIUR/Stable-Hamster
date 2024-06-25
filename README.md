---
title: STABLE HAMSTER
emoji: 🐹
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
base_model: stabilityai/sdxl-turbo
model: -
type: base_model, model
pinned: true
header: mini
theme: bethecloud/storj_theme
get_hamster_from: https://prithivhamster.vercel.app/
license: creativeml-openrail-m
short_description: Fast as Hamster | Stable Hamster | Stable Diffusion
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

![alt text](assets/xy.gif)

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Spaces: https://huggingface.co/spaces/prithivMLmods/STABLE-HAMSTER

Take Clone : 
    
    # Make sure you have git-lfs installed (https://git-lfs.com)
    git lfs install
    
    git clone https://huggingface.co/spaces/prithivMLmods/STABLE-HAMSTER
    
    # If you want to clone without large files - just their pointers
    
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/prithivMLmods/STABLE-HAMSTER
    

## Sample Images 

| ![Image 1](assets/1.png) | ![Image 2](assets/2.png) |
|-------------------------|-------------------------|
| ![Image 3](assets/3.png) | ![Image 4](assets/4.png) |

## How it works
![alt text](assets/rr.gif)


## Libraries Overview

| Library        | Library        | Library        |
|----------------|----------------|----------------|
| `diffusers`    | `pipeline`     | `sentencepiece`|
| `torch`        | `transformers` | `spaces`       |
| `gdown`        | `accelerate`   | `peft`         |
| `torchvision`  | `safetensors`  | `gradio`       |
| `pillow`       |                |                |



![alt text](assets/ty.gif)

## Compatibility

ZeroGPU Spaces should mostly be compatible with any PyTorch-based GPU Space.
Compatibility with high level HF libraries like transformers or diffusers is slightly more guaranteed
That said, ZeroGPU Spaces are not as broadly compatible as classical GPU Spaces and you might still encounter unexpected bugs

Also, for now, ZeroGPU Spaces only works with the Gradio SDK

Supported versions:

    Gradio: 4+
    PyTorch: All versions from 2.0.0 to 2.2.0
    Python: 3.10.13

## Usage

In order to make your Space work with ZeroGPU you need to decorate the Python functions that actually require a GPU with @spaces.GPU
During the time when a decorated function is invoked, the Space will be attributed a GPU, and it will release it upon completion of the function.
Here is a practical example :

    +import spaces
    from diffusers import DiffusionPipeline
    
    pipe = DiffusionPipeline.from_pretrained(...)
    pipe.to('cuda')
    
    +@spaces.GPU
    def generate(prompt):
        return pipe(prompt).images
    
    gr.Interface(
        fn=generate,
        inputs=gr.Text(),
        outputs=gr.Gallery(),
    ).launch()

   
We first import spaces (importing it first might prevent some issues but is not mandatory)
Then we decorate the generate function by adding a @spaces.GPU line before its definition

## Duration

If you expect your GPU function to take more than 60s then you need to specify a duration param in the decorator like:
    
    @spaces.GPU(duration=120)
    def generate(prompt):
       return pipe(prompt).images

It will set the maximum duration of your function call to 120s.

You can also specify a duration if you know that your function will take far less than the 60s default.

The lower the duration, the higher priority your Space visitors will have in the queue


.

.

.

