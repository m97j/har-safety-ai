import json

from huggingface_hub import hf_hub_download


def download_labels():
    return hf_hub_download(
        repo_id="m97j/har-safety-model",
        repo_type="model",
        filename="class_label.json",
    )

def load_labels():
    path=download_labels()
    with open(path) as f:
        return json.load(f)