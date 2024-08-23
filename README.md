# Train Flux LoRA

> If you are not a beginner please visit the original repository https://github.com/ostris/ai-toolkit. This is just a convenience script for total beginners.

https://github.com/user-attachments/assets/ad955ff5-bcaf-4299-8685-02fe1225b5af


This is meant to be run on the following spec server:

| Specification   | Value  |
|-----------------|--------|
| GPU             | 24GB and above VRAM (suggestions: A40, A5000, 3090)  |
| Container Size  | 40GB   |
| Volume Size     | 10GB   |

## Setup

- Initialize

    ```
    curl -s https://geocine.github.io/flux/init.sh | sh
    ```
- Paste your token on `token`
- Upload your photos on `data`
- Modify `prompts` and `trigger_word` on `config.yaml`
- Start training

    ```
    ./run.sh
    ```
- After training your model, you can download it from the `output` folder.
