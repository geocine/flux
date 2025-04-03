# Train Flux LoRA

> [!WARNING]
> If you're not a beginner, check out the original repo: https://github.com/ostris/ai-toolkit. This script is just for beginners.

https://github.com/user-attachments/assets/15dce82a-bb3f-4e46-b328-5732aea41e56

This is meant to be run on the following spec server. Register at https://console.quickpod.io

| Specification   | Recommended Value  |
|-----------------|--------------------|
| GPU             | Minimum 24GB VRAM. Use a 3090/4090 |
| Disk Space Size  | 100GB |

## Setup

1. Initialize

    ```
    curl -s https://geocine.github.io/flux/init.sh | sh
    ```
2. Paste your [huggingface](https://huggingface.co/settings/tokens) token on the `token` file.
3. Upload your photos on `data` folder
4. Modify `prompts` and `trigger_word` on `config.yaml`
5. Start training

    ```
    ./run.sh
    ```
6. After training your model, you can download it from the `output` folder. Checkout the [generate images](./docs/GENERATE.md) documentation.
