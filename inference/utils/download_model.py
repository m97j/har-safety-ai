from huggingface_hub import hf_hub_download


def download_model():
    return hf_hub_download(
        repo_id="m97j/har-safety-model",
        repo_type="model",
        filename="pytorch_model.bin",
    )