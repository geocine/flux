# Train Flux LoRA

[![Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgeocine%2Fflux&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

> [!WARNING]
> If you're not a beginner, check out the original repo: https://github.com/ostris/ai-toolkit. This script is just for beginners.



https://github.com/user-attachments/assets/50c53eca-98c7-4c2a-bd3d-84736acbf1a5


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
