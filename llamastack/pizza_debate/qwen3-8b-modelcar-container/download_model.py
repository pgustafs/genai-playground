from huggingface_hub import snapshot_download

# Specify the Hugging Face repository containing the model
model_repo = "RedHatAI/Qwen3-8B-FP8-dynamic"
snapshot_download(
    repo_id=model_repo,
    local_dir="/models",
    allow_patterns=["*.safetensors", "*.json", "*.txt"],
)
