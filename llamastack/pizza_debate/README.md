# Llama Stack AI Pizza Debate

🍕 Robo Pizza Debate: A Llama Stack Multi-Agent Demo
Where AI chefs engage in a passionate debate about pineapple on pizza!

This demo showcases:
- Multi-agent reasoning and argumentation
- Tool usage (web search) to support arguments
- Agent coordination and turn-taking
- Consensus building through debate
- Humor and personality in AI interactions


In this demo I'm using:
- An optimized Qwen3-8b LLM from The Red Hat AI repository on Hugging Face https://huggingface.co/RedHatAI
- vLLM for highly efficient and optimized inference with the local Qwen3-8b model
- Llama Stack with an remote vLLM provider as the generative AI platform to build this multi-agent app
- podman to run all above containerized on my RHEL server with a 16GB Nvidia consumer grade GPU (4070) to showcase you can nowdays do quite cool GenAI things with small resources thanks to open-sorce.

## Prerequisites
- RHEL9 or equilanet 
- NVIDIA, AMD or Intel GPU, I will use NVIDIA in this setup
- NVIDIA GPU driver installed
- NVIDIA Container Toolkit installed
- Podman installed
- Python installed, I used 3.12

## Setup

**Clone the repository**
```bash
git clone https://github.com/pgustafs/genai-playground.git
cd genai-playground/pizza_debate
```

**Create virtual environment**
I know I should change to uv but i took me 4 years to change from yum to dnf :)
```bash
python3.12 -m venv venv
source venv/bin/activate 
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Build the a ModelCar container**
See: https://developers.redhat.com/articles/2025/01/30/build-and-deploy-modelcar-container-openshift-ai# for detailed instructions

```bash
podman build qwen3-8b-modelcar-container/ -t modelcar-qwen3-8b-fp8-dynamic --platform linux/amd64
```

**Starting Llama Stack and vLLM with podman play kube**
This will create a single pod with three interconnected containers:

1. vLLM Server (Port 8080)
Inference server
Runs the qwen-8b model
Provides OpenAI-compatible API for chat completions
Uses GPU acceleration for fast responses

2. Model Storage (ModelCar)
Contains the pre-loaded qwen3-8b model files
Shares model data with the VLLM server

3. LlamaStack Platform (Port 5000)
Provides enhanced AI platform features
Connects to the VLLM server for actual inference
Offers additional tools and capabilities beyond basic chat

```bash
podman play kube llamastack-vllm-qwen3-pod.yaml
```

list pods
```bash
podman pod ps
```

list containers
```bash
podman ps
```

To view logs:
```bash
podman pod logs llamastack-vllm-qwen3
```

or:
```bash
podman pod logs -f llamastack-vllm-qwen3
```

To stop everything:
```bash
podman play kube --down llamastack-vllm-qwen3-pod.yaml
```

**verify vLLM**
to list the models:
```bash
curl http://localhost:8080/v1/models
```

query the model with input prompts:
```bash
curl http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-8b",
        "prompt": "Stockholm is a",
        "max_tokens": 20,
        "temperature": 0
    }'
```
**verify Llama Stack **
to list models:
```bash
llama-stack-client --endpoint http://localhost:5000 models list
```

to list providers:
```bash
llama-stack-client --endpoint http://localhost:5000 providers list
```

You can request inference from the CLI in this manner:
```bash
llama-stack-client --endpoint http://localhost:5000 inference chat-completion --message "tell me a joke"
```

Now everything should be set for running the Pizza Debate Demo :) 

## running the demo

**CLI**
```bash
LLAMA_STACK_ENDPOINT=http://localhost:5000 INFERENCE_MODEL=qwen3-8b TAVILY_SEARCH_API_KEY="your_tavily_api_key" python pizza_debate_demo_cli.py
```

**Web (gradio)**
```bash
LLAMA_STACK_ENDPOINT=http://localhost:5000 INFERENCE_MODEL=qwen3-8b TAVILY_SEARCH_API_KEY="your_tavily_api_key" python pizza_debate_demo_gradio.py
```