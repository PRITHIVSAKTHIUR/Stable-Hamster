import os
import random
import uuid
import json
import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

#Load the HTML content
#html_file_url = "https://prithivmlmods-hamster-static.static.hf.space/index.html"
#html_content = f'<iframe src="{html_file_url}" style="width:100%; height:180px; border:none;"></iframe>'
#html_file_url = "https://prithivmlmods-static-loading-theme.static.hf.space/index.html"

html_file_url = "https://prithivhamster.vercel.app/"
html_content = f'<iframe src="{html_file_url}" style="width:100%; height:400px; border:none"></iframe>'

DESCRIPTIONx = """## STABLE HAMSTER
"""

css = '''
.gradio-container{max-width: 560px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''

examples = [
    "3d image, cute girl, in the style of Pixar --ar 1:2 --stylize 750, 4K resolution highlights, Sharp focus, octane render, ray tracing, Ultra-High-Definition, 8k, UHD, HDR, (Masterpiece:1.5), (best quality:1.5)",
    "Cold coffee in a cup bokeh --ar 85:128 --v 6.0 --style raw5, 4K"
]

#Set an os.Getenv variable
#set VAR_NAME=”VALUE”
#Fetch an environment variable
#echo %VAR_NAME%

MODEL_ID = os.getenv("MODEL_REPO")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))  # Allow generating multiple images at once

#Load model outside of function
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True,
    add_watermarker=False,
).to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# <compile speedup >
if USE_TORCH_COMPILE:
    pipe.compile()

# Offloading capacity (RAM)
if ENABLE_CPU_OFFLOAD:
    pipe.enable_model_cpu_offload()

MAX_SEED = np.iinfo(np.int32).max

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@spaces.GPU(duration=60, enable_queue=True)
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 1,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    num_inference_steps: int = 25,
    randomize_seed: bool = False,
    use_resolution_binning: bool = True, 
    num_images: int = 1,  # Number of images to generate
    progress=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(device=device).manual_seed(seed)

    #Options 
    options = {
        "prompt": [prompt] * num_images,
        "negative_prompt": [negative_prompt] * num_images if use_negative_prompt else None,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "output_type": "pil",
    }

    #VRAM usage Lesser
    if use_resolution_binning:
        options["use_resolution_binning"] = True

    #Images potential batches
    images = []
    for i in range(0, num_images, BATCH_SIZE):
        batch_options = options.copy()
        batch_options["prompt"] = options["prompt"][i:i+BATCH_SIZE]
        if "negative_prompt" in batch_options:
            batch_options["negative_prompt"] = options["negative_prompt"][i:i+BATCH_SIZE]
        images.extend(pipe(**batch_options).images)

    image_paths = [save_image(img) for img in images]
    return image_paths, seed
#Main gr.Block
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown(DESCRIPTIONx)
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", columns=1, show_label=False) 
    with gr.Accordion("Advanced options", open=False, visible=False):
        num_images = gr.Slider(
            label="Number of Images",
            minimum=1,
            maximum=4,
            step=1,
            value=1,
        )
        with gr.Row():
            use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True)
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=5,
                lines=4,
                placeholder="Enter a negative prompt",
                value="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                visible=True,
            )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=MAX_IMAGE_SIZE,
                step=64,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=MAX_IMAGE_SIZE,
                step=64,
                value=1024,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=6,
                step=0.1,
                value=3.0,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=25,
                step=1,
                value=23,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        cache_examples=False
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            randomize_seed,
            num_images
        ],
        outputs=[result, seed],
        api_name="run",
    )   
    gr.HTML(html_content)
if __name__ == "__main__":
    demo.queue(max_size=40).launch()