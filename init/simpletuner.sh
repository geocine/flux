#!/bin/bash
set -e

# Create workspace data directory if it doesn't exist
mkdir -p /workspace/data

# Clone repository if not already present
if [ ! -d "/workspace/SimpleTuner" ]; then
    git clone --branch=release https://github.com/bghira/SimpleTuner.git
    cd /workspace/SimpleTuner
else
    echo "SimpleTuner already cloned, skipping..."
    cd /workspace/SimpleTuner
fi

# Create symbolic link of /workspace/SimpleTuner/config into /workspace/config, skip if already created
if [ ! -L "/workspace/config" ]; then
    ln -s /workspace/SimpleTuner/config /workspace/config
    echo "Created symbolic link for config"
else
    echo "Symbolic link for config already exists, skipping..."
fi

# Setup virtual environment if not already done
if [ ! -d ".venv" ]; then
    python -m venv .venv
    . .venv/bin/activate
    pip install -U poetry pip
    pip install hf_transfer
    poetry install
else
    echo "Virtual environment already exists, skipping setup..."
    . .venv/bin/activate
fi

# Create .env file if it doesn't exist, but don't ask for input
if [ ! -f "/workspace/token" ]; then
    touch /workspace/token
    echo "Created empty token file. Please edit it to add your token."
else
    echo "token file already exists, skipping..."
fi

cd /workspace

# Create run.sh file if it doesn't exist
if [ ! -f "/workspace/run.sh" ]; then
    cat << EOF > /workspace/run.sh
#!/bin/bash

. /workspace/SimpleTuner/.venv/bin/activate

# Read the Hugging Face token from the token file
HUGGINGFACE_TOKEN=\$(cat /workspace/token)

# Set environment variables
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=\$HUGGINGFACE_TOKEN

# Log in to Hugging Face
huggingface-cli login --token \$HUGGINGFACE_TOKEN

# Run the training script
cd /workspace/SimpleTuner
./train.sh
EOF

    chmod +x /workspace/run.sh
    echo "Created run.sh file"
else
    echo "run.sh already exists, skipping..."
fi

echo "SimpleTuner setup complete!"