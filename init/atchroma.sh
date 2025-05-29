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
    git clone https://github.com/geocine/ai-toolkit.git
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
trigger: "lyn"
prompts:
  - "portrait of [trigger] with long black hair, standing in a modern indoor space with pink and purple neon lighting"
  - "[trigger] holding a coffee cup, in a beanie, sitting at a cafe"
  - "[trigger] as wonder woman"

base_model: "lodestones/Chroma"
timestep_type: "sigmoid"
optimizer: "radamschedulefree"
optimizer_args:
    betas: [0.9, 0.99]
    weight_decay: 1e-4
arch: "chroma"
sample_width: 512
sample_height: 512
sample_steps: 25

# You may tweak these settings to modify how the training works.
steps: 5000
lr: 3e-4
# sample_every: 200
# save_every: 200
# max_step_saves_to_keep: 4
# gradient_checkpointing: false # not implemented for chroma yet
# These are the default settings for chroma.
network:
    type: "lora"
    linear: 32 # same thing as rank in this context
    linear_alpha: 32 # Not actuallly used in chroma, defaults to rank
    ramp_double_blocks: true
    ramp_target_lr: 1.5e-6       # final LR you want for the mapped blocks
    ramp_warmup_steps: 1000      # number of optimisation steps for the climb
    ramp_type: linear            # or "cosine"
    network_kwargs:
        lr_if_contains:
            double_blocks\$\$0\$\$: 0.001
            double_blocks\$\$1\$\$: 0.00289
            double_blocks\$\$2\$\$: 0.00456
            double_blocks\$\$3\$\$: 0.006
            double_blocks\$\$4\$\$: 0.00722
            double_blocks\$\$5\$\$: 0.00822
            double_blocks\$\$6\$\$: 0.009
            double_blocks\$\$7\$\$: 0.00956
            double_blocks\$\$8\$\$: 0.00989
            double_blocks\$\$9\$\$: 0.01
            double_blocks\$\$10\$\$: 0.00989
            double_blocks\$\$11\$\$: 0.00956
            double_blocks\$\$12\$\$: 0.009
            double_blocks\$\$13\$\$: 0.00822
            double_blocks\$\$14\$\$: 0.00722
            double_blocks\$\$15\$\$: 0.006
            double_blocks\$\$16\$\$: 0.00456
            double_blocks\$\$17\$\$: 0.00289
            double_blocks\$\$18\$\$: 0.0013
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


echo "Setup complete! Visit https://huggingface.co/settings/tokens/new? to create your token"