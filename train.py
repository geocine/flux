import os
import sys

# Read the token
with open("token", "r") as token_file:
    os.environ["HF_TOKEN"] = token_file.read().strip()

sys.path.append('/workspace/ai-toolkit')
import argparse
import logging
import torch
import oyaml as yaml
from collections import OrderedDict
from PIL import Image
from toolkit.job import run_job
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Union

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

def caption_image(image_path, prompt="<DETAILED_CAPTION>"):
    logging.info(f"Captioning image: {image_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    # Open the image and convert it to RGB mode
    image = Image.open(image_path).convert('RGB')
    image_size = image.size  # Get the image size (width, height)
    
    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    except ValueError as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

    # Convert inputs to the correct dtype and device
    inputs = {
        "input_ids": inputs["input_ids"].to(device).long(),
        "pixel_values": inputs["pixel_values"].to(device, dtype=torch_dtype)
    }

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1000,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    caption_dict = processor.post_process_generation(generated_text, task=prompt, image_size=image_size)
    
    # Extract the caption from the dictionary
    if isinstance(caption_dict, dict) and prompt in caption_dict:
        caption = caption_dict[prompt]
        if isinstance(caption, list):
            caption = ' '.join(caption)
    else:
        caption = str(caption_dict)  # Fallback to string representation if unexpected format

    return caption

def process_images(input_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(input_folder, filename)
            caption_filename = f"{os.path.splitext(filename)[0]}.txt"
            caption_path = os.path.join(input_folder, caption_filename)

            # Check if caption file already exists
            if os.path.exists(caption_path):
                logging.info(f"Caption already exists for {filename}, skipping...")
                continue

            # Get caption
            caption = caption_image(image_path)

            if caption.startswith("The image shows a"):
                caption = "A" + caption[17:]

            caption = caption.replace("of the image", "")
            
            # Save caption
            if caption:
                with open(caption_path, "w", encoding='utf-8') as caption_file:
                    caption_file.write(caption)
                logging.info(f"Caption saved for {filename}")
            else:
                logging.error(f"Failed to get caption for {filename}")

    logging.info(f"Processed images in {input_folder}")

def count_images(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Flux.1 dev LoRA training")
    parser.add_argument("trigger", nargs='?', default=None, help="Optional trigger word")
    parser.add_argument("model_name", nargs='?', default="lora", help="Name of the model (default: lora)")
    args = parser.parse_args()

    # Static paths
    input_folder = '/workspace/data'
    output_folder = '/workspace/output'
    config_file = '/workspace/config.yaml'

    # Check if input folder exists and contains images
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    num_images = count_images(input_folder)
    if num_images == 0:
        print(f"Error: No images found in the input folder '{input_folder}'.")
        print("Please add some images (PNG, JPG, JPEG, or GIF) to the input folder and try again.")
        sys.exit(1)

    # Load config
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        print("Please create a 'config.yaml' file in the /workspace directory.")
        sys.exit(1)

    config = load_config(config_file)

    # Get trigger from config if not provided as argument
    trigger = args.trigger or config.get('trigger')

    # Get prompts from config
    prompts = config.get('prompts', [])
    if not prompts:
        print("Error: No prompts found in the config file.")
        print("Please add prompts to the 'config.yaml' file.")
        sys.exit(1)

    # Process images
    process_images(input_folder)
    logging.info("Pre-process completed")

    # Calculate steps
    steps = num_images * 100

    # Prepare job configuration
    job_to_run = OrderedDict([
        ('job', 'extension'),
        ('config', OrderedDict([
            ('name', args.model_name),
            ('process', [
                OrderedDict([
                    ('type', 'sd_trainer'),
                    ('training_folder', output_folder),
                    ('device', 'cuda:0'),
                    ('trigger_word', trigger),
                    ('network', OrderedDict([
                        ('type', 'lora'),
                        ('linear', 16),
                        ('linear_alpha', 16)
                    ])),
                    ('save', OrderedDict([
                        ('dtype', 'float16'),
                        ('save_every', 200),
                        ('max_step_saves_to_keep', 4)
                    ])),
                    ('datasets', [
                        OrderedDict([
                            ('folder_path', input_folder),
                            ('caption_ext', 'txt'),
                            ('caption_dropout_rate', 0.05),
                            ('shuffle_tokens', False),
                            ('cache_latents_to_disk', True),
                            ('resolution', [512, 768, 1024])
                        ])
                    ]),
                    ('train', OrderedDict([
                        ('batch_size', 1),
                        ('steps', steps),
                        ('gradient_accumulation_steps', 1),
                        ('train_unet', True),
                        ('train_text_encoder', False),
                        ('content_or_style', 'balanced'),
                        ('gradient_checkpointing', True),
                        ('noise_scheduler', 'flowmatch'),
                        ('optimizer', 'adamw8bit'),
                        ('lr', 4e-4),
                        ('skip_first_sample', True),
                        ('ema_config', OrderedDict([
                            ('use_ema', True),
                            ('ema_decay', 0.99)
                        ])),

                        ('dtype', 'bf16')
                    ])),
                    ('model', OrderedDict([
                        ('name_or_path', 'black-forest-labs/FLUX.1-dev'),
                        ('is_flux', True),
                        ('quantize', True)
                    ])),
                    ('sample', OrderedDict([
                        ('sampler', 'flowmatch'),
                        ('sample_every', 200),
                        ('width', 1024),
                        ('height', 1024),
                        ('prompts', prompts),
                        ('neg', ''),
                        ('seed', 42),
                        ('walk_seed', True),
                        ('guidance_scale', 4),
                        ('sample_steps', 20)
                    ]))
                ])
            ]),
            ('meta', OrderedDict([
                ('name', '[name]'),
                ('version', '1.0')
            ]))
        ]))
    ])

    # Run the job
    run_job(job_to_run)

if __name__ == "__main__":
    main()