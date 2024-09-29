# Generate Images

> Video to follow, for now please follow instructions below

## Generate

> [!WARNING]
> You cannot run this while training is still ongoing. 

### Step 1: Run the gradio interface

> [!NOTE]  
> **Option 1**: If you are using the same runpod for training and generation, follow this instruction

Start the gradio interface using the following command:
```
./gen.sh
```

> [!NOTE]  
> **Option 2**: If your are NOT using the same runpod for training and generation, follow this instruction.

1. Setup the workspace by running following command:

    ```
    curl -s https://geocine.github.io/flux/gen.sh | sh
    ```
2. Paste your [huggingface](https://huggingface.co/settings/tokens) token on the `token` file.
3. Start the gradio interface using the following command:
   ```
   python inference.py
   ```

### Step 2: Ensure you have a LoRA file

Upload your LoRA file to the `/workspace/output/lora` folder. If you need to, else you can just use the existing LoRA files.

### Step 3: Generate images

> [!NOTE]  
> First generation will be slower since it requires to load the model into the VRAM. The subsequent generations will be faster.
1. You will get a URL to access the Gradio interface.
2. Enter a prompt and click generate.
3. Your files will be saved to the `/workspace/output/generations` folder.
