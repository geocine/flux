import gradio as gr
import torch
from diffusers import FluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
import os
import sys
from datetime import datetime

# Read the token
if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"]:
    if not os.path.isfile("token"):
        with open("token", "w") as token_file:
            pass
        print("Token file created. Please add your Hugging Face token to the 'token' file and run the script again.")
        sys.exit(1)
    elif os.path.getsize("token") == 0:
        print("Token file is empty. Please add your Hugging Face token to the 'token' file and run the script again.")
        sys.exit(1)
    with open("token", "r") as token_file:
        os.environ["HF_TOKEN"] = token_file.read().strip()
else:
    print("Using HF_TOKEN from environment variable.")


# Define the output directories
output_dir = "/workspace/output"
lora_dir = "/workspace/ai-toolkit/output/axst2"

# Ensure the output directories exist
os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(lora_dir):
    os.makedirs(lora_dir, exist_ok=True)
elif os.path.islink(lora_dir):
    print(f"{lora_dir} is a symlink, skipping creation.")

# Load the base FLUX model
model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Global variable to track the currently loaded LoRA
current_lora = None

def get_lora_files():
    return [f for f in os.listdir(lora_dir) if f.endswith('.safetensors') or (os.path.islink(os.path.join(lora_dir, f)) and os.path.realpath(os.path.join(lora_dir, f)).endswith('.safetensors'))]

def get_default_lora():
    lora_files = get_lora_files()
    return lora_files[0] if lora_files else None

def refresh_lora_list():
    lora_files = get_lora_files()
    return gr.Dropdown(choices=lora_files, value=lora_files[0] if lora_files else None)

def load_lora(lora_name):
    global current_lora
    if current_lora != lora_name:
        # Unload the previous LoRA if it exists
        if current_lora:
            print(f"Unloading LoRA: {current_lora}")
            pipe.unload_lora_weights()
            #pipe.unfuse_lora()
        
        # Load the new LoRA weights
        lora_path = os.path.join(lora_dir, lora_name)
        print(f"Loading LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path)
        #pipe.fuse_lora(lora_scale=1.0)
        current_lora = lora_name

def generate_image(prompt, num_inference_steps, lora_name, num_images, width, height):
    # Load the LoRA weights if necessary
    load_lora(lora_name)

    # Create and set the FlowMatch scheduler
    flow_match_scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = flow_match_scheduler

    # Generate images
    images = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        num_images_per_prompt=num_images,
        height=height,
        width=width
    ).images

    # Save the generated images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_images = []
    for i, image in enumerate(images):
        filename = f"generated_{timestamp}_{i+1}.png"
        save_path = os.path.join(output_dir, filename)
        image.save(save_path)
        saved_images.append(save_path)
        print(f"Image saved to: {save_path}")

    return saved_images

# Create Gradio interface
with gr.Blocks(css="""
    .small-button {
        max-width: 40px !important;
        min-width: 40px !important;
        height: 40px !important;
        padding: 0px !important;
    }
""") as iface:
    gr.Markdown("# FLUX Image Generator")
    gr.Markdown("Generate images using FLUX model with custom LoRA")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=2)
            with gr.Row():
                num_images = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Number of images")
                steps = gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Steps")
            with gr.Row():
                width = gr.Slider(minimum=512, maximum=2048, step=8, value=1024, label="Width")
                height = gr.Slider(minimum=512, maximum=2048, step=8, value=1024, label="Height")
            with gr.Row():
                lora_dropdown = gr.Dropdown(choices=get_lora_files(), value=get_default_lora(), label="LoRA File", scale=15)
                refresh_button = gr.Button("🔄", elem_classes="small-button", scale=1)
            generate_button = gr.Button("Generate")

        with gr.Column(scale=1):
            gallery = gr.Gallery(
                label="Generated Images", 
                show_label=True, 
                elem_id="gallery", 
                columns=[2], 
                rows=[2], 
                object_fit="contain", 
                height="768px"
            )

    # Set up event handlers
    refresh_button.click(refresh_lora_list, outputs=[lora_dropdown])
    generate_button.click(
        generate_image,
        inputs=[prompt, steps, lora_dropdown, num_images, width, height],
        outputs=gallery
    )

# Launch the app
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)