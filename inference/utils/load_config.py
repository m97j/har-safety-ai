import json

from huggingface_hub import hf_hub_download


def download_config():
    return hf_hub_download(
        repo_id="m97j/har-safety-model",
        repo_type="model",
        filename="config.json",
    )

def load_config():
    path=download_config()
    with open(path, "r") as f:
        return json.load(f)
