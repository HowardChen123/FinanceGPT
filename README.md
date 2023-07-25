#  LLM for Sentimental Analysis

This repository contains code for fine tuning LLaMA 7B using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf) for the use case of sentiment analysis. This project is greatly inspired by [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/tree/main)

### Local Setup

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Install required packages to test the model in Streamlit

    ```bash
    pip install -q streamlit
    npm install localtunnel
    ```

### Training (`finetune.py`)

This file contains the process of fine-tuning the LLaMA 7b model

Example usage:

```bash
python finetune.py
```

### Inference (`generate.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `howardchen123/alpaca-lora-llama-sentiment`

Example usage:

```bash
python 
import generate
generate.generate_response("Text to get sentiment")
```

### Demo App (`streamlit_app.py`)

This file runs a Streamlit App for inference. The app was tested on Google Colab using T4 GPU

Example usage:

```bash
streamlit run streamlit_app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com 8501
```