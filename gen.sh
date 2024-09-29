curl -o /workspace/inference.py https://geocine.github.io/flux/inference.py
pip install "transformers[sentencepiece]" transformers accelerate peft diffusers safetensors gradio torch  hf_transfer 
# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
touch /workspace/token