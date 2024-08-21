# Train Flux LoRA

Please visit the original repository https://github.com/ostris/ai-toolkit, this is just a convenience script for total beginners.

https://github.com/user-attachments/assets/66a9c8cd-fbcf-4a12-b8be-9d7ab3396442

This is meant to be run on the following spec server:

| Specification   | Value  |
|-----------------|--------|
| GPU             | 24GB VRAM (A5000 or 3090 as options)   |
| Container Size  | 40GB   |
| Volume Size     | 10GB   |

## Setup

- Initialize

    ```
    curl -s https://geocine.github.io/flux/init.sh | sh
    ```
- Paste your token on `token`
- Upload your photos on `data`
- Modify `prompts` and `trigger_wor` on `config.yaml`
- Start training

    ```
    ./run.sh
    ```
- After training your model, you can download it from the `output` folder.
