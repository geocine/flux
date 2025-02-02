#!/bin/bash
set -e

RELEASE=28102024
DEST_DIR=~/.cache/trainer/FLUX.1-dev
 
if [ -z "$HF_TOKEN" ]; then
    if [ ! -f "token" ]; then
        touch token
        echo "Token file created. Please add your Hugging Face token to the 'token' file and run the script again."
        exit 1
    elif [ ! -s "token" ]; then
        echo "Token file is empty. Please add your Hugging Face token to the 'token' file and run the script again."
        exit 1
    fi
    export HF_TOKEN=$(cat token)
else
    echo "Using HF_TOKEN from environment variable."
fi

pip install -U "huggingface_hub[cli]" hf_transfer
mkdir -p "$DEST_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1

download_if_not_exists() {
    local repo=$1
    local file=$2
    local dest="$DEST_DIR/$file"
    if [ ! -L "$dest" ]; then
        echo "Downloading $file..."
        huggingface-cli download "$repo" "$file"
        
        # Construct the path to the downloaded file
        local org=$(echo $repo | cut -d'/' -f1)
        local repo_name=$(echo $repo | cut -d'/' -f2)
        local cache_dir="/root/.cache/huggingface/hub/models--${org}--${repo_name}/snapshots"
        local latest_snapshot=$(ls -t $cache_dir | head -n1)
        local source_file="$cache_dir/$latest_snapshot/$file"
        
        # Create a symlink to the file
        if [ -f "$source_file" ]; then
            ln -s "$source_file" "$dest"
            echo "Created symlink for $file at $dest"
        else
            echo "Error: Downloaded file not found at $source_file"
        fi
    else
        echo "Symlink for $file already exists. Skipping download."
    fi
}

download_if_not_exists "black-forest-labs/FLUX.1-dev" "flux1-dev.safetensors"
download_if_not_exists "black-forest-labs/FLUX.1-dev" "ae.safetensors"
download_if_not_exists "comfyanonymous/flux_text_encoders" "clip_l.safetensors"
download_if_not_exists "comfyanonymous/flux_text_encoders" "t5xxl_fp16.safetensors"

# Create workspace data directory if it doesn't exist
mkdir -p /workspace/data

# Check if zip and unzip are installed, if not install them
if ! command -v zip &> /dev/null || ! command -v unzip &> /dev/null; then
    apt update
    apt install -y zip unzip
fi
# Clone repository if not already present
if [ ! -d "/workspace/trainer" ]; then
    wget https://pub-4f2510d6d6de4750901ab8f82f214c02.r2.dev/files/trainer-${RELEASE}.zip -O /workspace/trainer.zip
    if [ -z "$ZIP_PASSWORD" ]; then
        echo "Please enter the password for the zip file:"
        read ZIP_PASSWORD < /dev/tty
    fi
    unzip -P "$ZIP_PASSWORD" /workspace/trainer.zip -d /workspace/trainer
    rm /workspace/trainer.zip
    cd /workspace/trainer
else
    echo "trainer already cloned, skipping..."
    cd /workspace/trainer
fi

# Create symbolic link of /workspace/SimpleTuner/config into /workspace/config, skip if already created
if [ ! -L "/workspace/configs" ]; then
    ln -s /workspace/trainer/configs /workspace/configs
    echo "Created symbolic link for config"
else
    echo "Symbolic link for config already exists, skipping..."
fi

# Setup virtual environment if not already done
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    pip install transformers==4.44.2
    pip install hf_transfer
else
    echo "Virtual environment already exists, skipping setup..."
    . .venv/bin/activate
fi

# Write accelerate config file if it doesn't exist
if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    mkdir -p ~/.cache/huggingface/accelerate
    cat << EOF > ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

    echo "Accelerate config file written to ~/.cache/huggingface/accelerate/default_config.yaml"
else
    echo "Accelerate config file already exists, skipping..."
fi

cd /workspace

# Create run.sh file if it doesn't exist
if [ ! -f "/workspace/run.sh" ]; then
    cat << 'EOF' > /workspace/run.sh
#!/bin/bash

echo "Activating virtual environment..."
. /workspace/trainer/.venv/bin/activate

if [ -z "$1" ]; then
    echo "Error: No configuration file provided."
    echo "Usage: $0 <path_to_config_file>"
    exit 1
fi

python /workspace/trainer/train.py "$1"
EOF

    chmod +x /workspace/run.sh
    echo "Created run.sh file"
else
    echo "run.sh already exists, skipping..."
fi

echo "Setup complete!"