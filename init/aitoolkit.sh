#!/bin/bash
set -e

# Check if zip and unzip are installed, if not install them
if ! command -v zip &> /dev/null || ! command -v unzip &> /dev/null; then
    apt update
    apt install -y zip unzip
fi

# Create workspace data directory if it doesn't exist
mkdir -p /workspace/data

# Clone repository if not already present
if [ ! -d "/workspace/ai-toolkit" ]; then
    git clone https://github.com/ostris/ai-toolkit.git
    cd /workspace/ai-toolkit
    git submodule update --init --recursive
else
    echo "ai-toolkit already cloned, skipping..."
    cd /workspace/ai-toolkit
fi

# Setup virtual environment if not already done
if [ ! -d "venv" ]; then
    python -m venv venv
    . venv/bin/activate
    ipython kernel install --user --name=VENV
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    pip3 install "transformers[sentencepiece]"
else
    echo "Virtual environment already exists, skipping setup..."
    . venv/bin/activate
fi

cd /workspace

# Create config.yaml file if it doesn't exist
if [ ! -f "config.yaml" ]; then
    cat << EOF > config.yaml
trigger: "ohwx woman"
prompts:
  - "portrait of a young [trigger] with long black hair, standing in a modern indoor space with pink and purple neon lighting"
  - "a [trigger] holding a coffee cup, in a beanie, sitting at a cafe"
  - "[trigger] as wonder woman"

# You may tweak these settings to modify how the training works.
# steps: 2000
# lr: 4e-4
# sample_every: 200
# save_every: 200
# max_step_saves_to_keep: 4
# base_model: "black-forest-labs/FLUX.1-dev"
# content_or_style: "balanced"
EOF
    echo "Created config.yaml"
else
    echo "config.yaml already exists, skipping..."
fi

# Create .env file if it doesn't exist, but don't ask for input
if [ ! -f "/workspace/token" ]; then
    touch /workspace/token
    echo "Created empty token file. Please edit it to add your token."
else
    echo "token file already exists, skipping..."
fi

# Download run.py if it doesn't exist
if [ ! -f "/workspace/ai-toolkit/train.py" ]; then
    curl -o /workspace/ai-toolkit/train.py https://geocine.github.io/flux/train.py
    echo "Downloaded train.py"
else
    echo "train.py already exists, skipping download..."
fi

# Create run.sh file if it doesn't exist
if [ ! -f "/workspace/run.sh" ]; then
    cat << EOF > /workspace/run.sh
#!/bin/bash

. /workspace/ai-toolkit/venv/bin/activate
python /workspace/ai-toolkit/train.py
EOF

    chmod +x /workspace/run.sh
    echo "Created run.sh file"
else
    echo "run.sh already exists, skipping..."
fi

# Download inference.py if it doesn't exist
if [ ! -f "/workspace/ai-toolkit/inference.py" ]; then
    curl -o /workspace/ai-toolkit/inference.py https://geocine.github.io/flux/inference.py
    echo "Downloaded inference.py"
else
    echo "inference.py already exists, skipping download..."
fi

# Create gen.sh file if it doesn't exist
if [ ! -f "/workspace/gen.sh" ]; then
    cat << EOF > /workspace/gen.sh
#!/bin/bash

. /workspace/ai-toolkit/venv/bin/activate
python /workspace/ai-toolkit/inference.py
EOF

    chmod +x /workspace/gen.sh
    echo "Created gen.sh file"
else
    echo "gen.sh already exists, skipping..."
fi

echo "Setup complete! Visit https://huggingface.co/black-forest-labs/FLUX.1-dev to accept the license and create your token at https://huggingface.co/settings/tokens/new?"