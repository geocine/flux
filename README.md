# Train Flux LoRA

[![Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgeocine%2Fflux&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

> [!WARNING]
> If you're not a beginner, check out the original repo: https://github.com/ostris/ai-toolkit. This script is just for beginners.

https://github.com/user-attachments/assets/ad955ff5-bcaf-4299-8685-02fe1225b5af


This is meant to be run on the following spec server:

| Specification   | Recommended Value  |
|-----------------|--------------------|
| GPU             | Minimum 24GB VRAM (e.g., A40, A5000, 3090). Use 48GB VRAM for image generation. |
| Container Size  | Minimum 40GB. Set to 60GB if you also want to [generate images](./docs/GENERATE.md). |
| Volume Size     | 20GB               |

## Setup

1. Initialize (default trainer is aitoolkit)

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