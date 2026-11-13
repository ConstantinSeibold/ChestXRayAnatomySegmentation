import gdown, os, torch
from ..label_mapper import id2label_dict

model_urls = {
    "UNet_ResNet50_default": "https://drive.google.com/file/d/1Y9zubvMzkYHoAqz-NvV6vniH5FKAF2iV/view?usp=drive_link"
}

# Weights are hosted on the Hugging Face Hub, pinned to the release tag this CXAS version was tested with; every
# downloaded file is checked against the repository's SHA256SUMS. Query-based (Mask2Former) models are directories
# of per-view bundles; the UNet is a single checkpoint (Google Drive remains as a fallback).
HF_MODEL_REPO = os.environ.get("CXAS_HF_REPO", "cmseibold/cxas-models")
HF_MODEL_REVISION = os.environ.get("CXAS_HF_REVISION", "v0.1.0")
bundle_models = {
    "M2F_anatomy_v2": "M2F_anatomy_v2",
}
hf_checkpoints = {
    "UNet_ResNet50_default": "UNet_ResNet50_default.pth",
}


def verify_sha256(root: str, sums_file: str, prefix: str = "") -> None:
    """Check every file listed in sums_file (sha256sum format) whose path starts with prefix, relative to root."""
    import hashlib

    bad = []
    for line in open(sums_file):
        digest, rel = line.strip().split(None, 1)
        if not rel.startswith(prefix):
            continue
        h = hashlib.sha256()
        with open(os.path.join(root, rel), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            bad.append(rel)
    if bad:
        raise RuntimeError("Checksum mismatch for downloaded weights (delete them and retry): {}".format(bad))


def weights_dir() -> str:
    root = os.environ["CXAS_PATH"] if "CXAS_PATH" in os.environ else os.environ["HOME"]
    return os.path.join(root, ".cxas", "weights")


def download_bundle(model_name: str) -> str:
    """Local directory of a bundle model; fetched from the Hugging Face Hub if not present."""
    local = os.path.join(weights_dir(), model_name)
    if os.path.isdir(local) and any(
        os.path.isfile(os.path.join(local, v, "meta.json")) for v in ("frontal", "lateral")
    ):
        return local
    from huggingface_hub import snapshot_download

    snapshot_download(
        HF_MODEL_REPO, revision=HF_MODEL_REVISION, local_dir=weights_dir(),
        allow_patterns=[bundle_models[model_name] + "/*", "SHA256SUMS"],
    )
    verify_sha256(weights_dir(), os.path.join(weights_dir(), "SHA256SUMS"), bundle_models[model_name] + "/")
    return local


def get_model(model_name, gpus=""):
    """
    Function to load a model by name and optionally move it to specified GPUs.

    Args:
        model_name (str): Name of the model to load.
        gpus (str): String containing GPU device IDs separated by commas.
                    If empty or 'cpu', the model will be loaded on CPU.

    Returns:
        torch.nn.Module: Loaded model.
    """
    assert model_name.split("_")[0] in list(model_getter.keys())

    if model_name in bundle_models:
        return get_bundle_model(model_name, gpus)

    model = model_getter[model_name.split("_")[0]](model_name)
    download_weights(model_name)
    model = load_weights(
        model, model_name, map_location="cpu" if "cpu" in gpus else "cuda"
    )

    if "cpu" not in gpus:
        gpus = [int(i) for i in gpus.split(",") if len(i) > 0]

        if len(gpus) > 1:
            assert torch.cuda.is_available()
            model.to(gpus[0])
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        elif len(gpus) == 1:
            assert torch.cuda.is_available()
            model.to(gpus[0])

    return model


def get_unet(model_name):
    """
    Function to get a U-Net model.

    Args:
        model_name (str): Name of the U-Net model.

    Returns:
        torch.nn.Module: U-Net model.
    """
    from .UNet.backbone_unet import BackboneUNet

    return BackboneUNet(model_name, len(id2label_dict.keys()))


def get_bundle_model(model_name: str, gpus: str = ""):
    """
    Load a query-based anatomy model (Mask2Former, one bundle per view) on the first requested device.

    Args:
        model_name (str): Key of `bundle_models`.
        gpus (str): "cpu", or comma-separated GPU ids; the first one is used.
    """
    from .qseg_backend import QsegAnatomy

    ids = [i for i in str(gpus).split(",") if i.strip().isdigit()]
    if "cpu" in str(gpus) or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda:{}".format(ids[0] if ids else 0)
    return QsegAnatomy(download_bundle(model_name), device=device)


def _is_valid_checkpoint(path: str) -> bool:
    """
    Check that a weight file is an actual torch checkpoint rather than, e.g., the
    HTML error page gdown saves when Google Drive blocks the download (quota or
    permission). Torch checkpoints are zip archives ("PK") or legacy pickles
    (0x80); HTML pages start with "<".

    Args:
        path (str): Path to the weight file.

    Returns:
        bool: True if the file looks like a torch checkpoint.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
    except OSError:
        return False
    return magic[:2] == b"PK" or magic[:1] == b"\x80"


def download_weights(model_name: str) -> None:
    """
    Function to download model weights.

    Args:
        model_name (str): Name of the model.
    """
    if "CXAS_PATH" in os.environ:
        store_path = os.path.join(os.environ["CXAS_PATH"], ".cxas")
    else:
        store_path = os.path.join(os.environ["HOME"], ".cxas")
    os.makedirs(os.path.join(store_path, "weights/"), exist_ok=True)
    out_path = os.path.join(store_path, "weights/{}".format(model_name + ".pth"))
    if os.path.isfile(out_path):
        if _is_valid_checkpoint(out_path):
            return
        # A previous run saved a corrupt file (e.g. a Google Drive error page);
        # drop it so it gets downloaded again.
        os.remove(out_path)
    if model_name in hf_checkpoints:
        try:
            from huggingface_hub import hf_hub_download

            weights = os.path.join(store_path, "weights")
            for fname in (hf_checkpoints[model_name], "SHA256SUMS"):
                hf_hub_download(HF_MODEL_REPO, fname, revision=HF_MODEL_REVISION, local_dir=weights)
            verify_sha256(weights, os.path.join(weights, "SHA256SUMS"), hf_checkpoints[model_name])
            if _is_valid_checkpoint(out_path):
                return
        except Exception as e:  # Hub unreachable or repo not available: fall back to Google Drive
            print("Hugging Face download of '{}' failed ({}); trying Google Drive.".format(model_name, e))
    result = gdown.download(model_urls[model_name], out_path, quiet=False, fuzzy=True)
    if result is None or not _is_valid_checkpoint(out_path):
        if os.path.isfile(out_path):
            os.remove(out_path)
        raise RuntimeError(
            "Failed to download valid model weights for '{}' from {}. The download "
            "was likely blocked by Google Drive (quota or permission). Please "
            "download the file manually and place it at '{}'.".format(
                model_name, model_urls[model_name], out_path
            )
        )
    return


def load_weights(model, model_name: str, map_location: str = "cuda:0"):
    """
    Function to load model weights.

    Args:
        model (torch.nn.Module): Model to load weights into.
        model_name (str): Name of the model.
        map_location (str): Location to map tensors to (default: 'cuda:0' if available, else 'cpu').

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    if "CXAS_PATH" in os.environ:
        store_path = os.path.join(os.environ["CXAS_PATH"], ".cxas")
    else:
        store_path = os.path.join(os.environ["HOME"], ".cxas")
    out_path = os.path.join(store_path, "weights/{}".format(model_name + ".pth"))
    assert os.path.isfile(out_path)

    checkpoint = torch.load(out_path, map_location=map_location, weights_only=False)

    if "module" in list(checkpoint["model"].keys())[0]:
        for i in list(checkpoint["model"].keys()):
            checkpoint["model"][i[len("module.") :]] = checkpoint["model"].pop(i)
    model.load_state_dict(checkpoint["model"], strict=False)
    return model


# Dictionary containing model getter functions
model_getter = {
    "UNet": get_unet,
    "M2F": get_bundle_model,
}
