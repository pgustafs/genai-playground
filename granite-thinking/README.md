# Granite 3.2 Thinking Mode Demo
This repository contains a demonstration application that allows you to test the "thinking mode" capabilities of IBM's Granite 3.2-8B-Instruct language model using a Gradio web interface.

## Overview
The demo showcases the IBM Granite 3.2-8B model's ability to perform step-by-step reasoning through complex problems when prompted with specific instructions. The setup:

- Serves the Granite 3.2-8B-Instruct model using vLLM for optimized inference
- Provides a user-friendly web interface built with Gradio
- Implements a "thinking mode" pipeline using LangGraph
- Allows comparison between standard responses and enhanced reasoning responses

## Requirements

- RHEL 9 or compatible Linux distribution
- NVIDIA GPU with at least 16GB VRAM
- NVIDIA drivers installed
- CUDA toolkit installed
- NVIDIA Container Toolkit (for containerized setup)

## Setup Options
You can run this demo in two ways:

1. Standard setup: Direct installation on your host system
2. Containerized setup: Using Podman containers for improved isolation and portability

## Standard Setup Instructions
```bash
# Install Python 3.12
sudo dnf install python3.12 

# Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip
pip install vllm
pip install bitsandbytes
pip install huggingface_hub
pip install langchain_openai
pip install langgraph
pip install gradio

# Create a directory for models and download Granite 3.2-8B-Instruct
mkdir ~/models/
huggingface-cli download ibm-granite/granite-3.2-8b-instruct --local-dir ~/models/granite-3.2-8b-instruct

# Start the vLLM server with the Granite model
vllm serve ~/models/granite-3.2-8b-instruct --dtype=half --max-model-len=8192 --served-model-name granite-3.2-8b-instruct --quantization bitsandbytes --load-format bitsandbytes

# Open a new terminal window and configure firewall (keep the vLLM server running)
sudo firewall-cmd --permanent --add-port=7860/tcp 
sudo firewall-cmd --reload

# Clone the repository and start the application
git clone https://github.com/pgustafs/genai-playground.git
cd genai-playground/granite-thinking
python app.py 

# Access the web interface
# Browse to: http://<your-server-ip>:7860
```

## Containerized Setup Instructions
```bash
# Clone the repository
git clone https://github.com/pgustafs/genai-playground.git

# Build the Gradio application container
cd genai-playground/granite-thinking
podman build -t gradio-granite-thinking .

# Build the NGINX proxy container
cd ../nginx-gradio-proxy
podman build -t nginx-gradio-proxy .

# Create a pod for the application
podman pod create -p 8080:80 -p 8443:443 granite-thinking

# Run the vLLM server container with GPU access
podman run -d --name vllm --pod granite-thinking --device nvidia.com/gpu=all -v ~/models:/models:z vllm/vllm-openai:latest --model /models/granite-3.2-8b-instruct --dtype=half --max-model-len=8096 --served-model-name granite-3.2-8b-instruct --quantization bitsandbytes --load-format bitsandbytes

# Run the Gradio application container
podman run -d --name gradio --pod granite-thinking -e MODEL_ENDPOINT_URL=http://127.0.0.1:8000 -e MODEL_NAME=granite-3.2-8b-instruct gradio-granite-thinking:latest

# Run the NGINX proxy container
podman run -d --name nginx-gradio-proxy --pod granite-thinking -e SERVER_NAME=pgustafs.com nginx-gradio-proxy:latest

# Configure firewall
sudo firewall-cmd --permanent --add-port=8443/tcp
sudo firewall-cmd --reload

# Access the web interface
# Browse to: http://<your-server-ip>:8443
```