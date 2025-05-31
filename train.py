import os
import sys

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

sys.path.append('/workspace/ai-toolkit')
import argparse
import logging
import torch
import oyaml as yaml
import json
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

# Global model and processor variables
model = None
processor = None

def load_model():
    """Load Florence-2 model and processor once globally"""
    global model, processor
    if model is None or processor is None:
        logging.info("Loading Florence-2 model and processor...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(
                "MiaoshouAI/Florence-2-base-PromptGen-v2.0", 
                trust_remote_code=True
            ).to(device)
            processor = AutoProcessor.from_pretrained(
                "MiaoshouAI/Florence-2-base-PromptGen-v2.0", 
                trust_remote_code=True
            )
        logging.info("Model and processor loaded successfully")
    return model, processor

def caption_image(image_path, prompt="<DETAILED_CAPTION>"):
    logging.info(f"Captioning image: {image_path}")
    
    # Load model and processor
    model, processor = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process image and prompt
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
        
        # Decode generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Post-process the generated text
        caption = processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        logging.info(f"Caption generated successfully")
        logging.info(f"Caption type: {type(caption)}")
        
        # Handle the case where caption is a dictionary
        if isinstance(caption, dict):
            # Log the keys to help with debugging
            logging.info(f"Caption keys: {caption.keys()}")
            
            # Check for the <DETAILED_CAPTION> key specifically
            if prompt in caption:
                caption_result = caption[prompt]
            # Try other common keys
            elif 'caption' in caption:
                caption_result = caption['caption']
            elif 'text' in caption:
                caption_result = caption['text']
            elif 'description' in caption:
                caption_result = caption['description']
            else:
                # If we can't find a specific key, convert the whole dict to a string
                caption_result = json.dumps(caption)
        else:
            caption_result = caption
        
        # Ensure we return a string
        if not isinstance(caption_result, str):
            caption_result = str(caption_result)
            
        return caption_result
    
    except Exception as e:
        logging.error(f"Error generating caption for {image_path}: {str(e)}")
        return None

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
    config_file = '/workspace/config.yaml'

    # Load config
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        print("Please create a 'config.yaml' file in the /workspace directory.")
        sys.exit(1)

    config = load_config(config_file)

    # Get input folder from config, with alias support
    input_folder = config.get('data_folder', config.get('input_folder', '/workspace/data'))
    output_folder = '/workspace/output'

    # Check if input folder exists and contains images
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    num_images = count_images(input_folder)
    if num_images == 0:
        print(f"Error: No images found in the input folder '{input_folder}'.")
        print("Please add some images (PNG, JPG, JPEG, or GIF) to the input folder and try again.")
        sys.exit(1)

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

    # Calculate steps (use config value if present, otherwise default)
    steps = config.get('steps', num_images * 100)

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
                    ('network', config.get('network', OrderedDict([
                        ('type', 'lora'),
                        ('linear', 16),
                        ('linear_alpha', 16)
                    ]))),
                    ('save', OrderedDict([
                        ('dtype', 'float16'),
                        ('save_every', config.get('save_every', 200)),
                        ('max_step_saves_to_keep', config.get('max_step_saves_to_keep', 10)),
                        ('push_to_hub', False)
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
                        ('batch_size', config.get('batch_size', 1)),
                        ('steps', steps),
                        ('gradient_accumulation_steps', 1),
                        ('train_unet', True),
                        ('train_text_encoder', False),
                        ('gradient_checkpointing', config.get('gradient_checkpointing', True)),
                        ('noise_scheduler', 'flowmatch'),
                        ('timestep_type', config.get('timestep_type', 'linear')),
                        ('optimizer', config.get('optimizer', 'adamw8bit')),
                        ('optimizer_args', config.get('optimizer_args', {})),
                        ('lr', config.get('lr', 1e-4)),
                        ('skip_first_sample', True),
                        ('ema_config', OrderedDict([
                            ('use_ema', True),
                            ('ema_decay', 0.99)
                        ])),
                        ('dtype', 'bf16')
                    ])),
                    ('model', OrderedDict([
                        ('name_or_path', config.get('base_model', config.get('name_or_path', 'black-forest-labs/FLUX.1-dev'))),
                        ('arch', config.get('arch', 'flux')),
                        ('quantize', True)
                    ])),
                    ('sample', OrderedDict([
                        ('sampler', 'flowmatch'),
                        ('sample_every', config.get('sample_every', 200)),
                        ('width', config.get('sample_width', 1024)),
                        ('height', config.get('sample_height', 1024)),
                        ('prompts', prompts),
                        ('neg', ''),
                        ('seed', 42),
                        ('walk_seed', True),
                        ('guidance_scale', 4),
                        ('sample_steps', config.get('sample_steps', 20))
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