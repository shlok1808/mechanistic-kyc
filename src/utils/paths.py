"""Result paths and cache identity.

Two rules live here.

1. The model goes in the DIRECTORY, never in the filename. `results/gemma-2-9b-it/
   advice.jsonl` rather than `results/advice_gemma-2-9b-it.jsonl`. Filenames stay
   readable and a 2B dev run can never overwrite a 9B production run.

2. An activation cache is identified by everything that can change its contents,
   not just the model name. The June pipeline keyed the cache on the model alone
   while the activations also depended on the prompt built in advice.py, so editing
   the prompt and re-running silently reused stale activations: the resume path
   skips vignette_ids it has already seen, and the ids had not changed.
"""

import hashlib
import json
import subprocess
from pathlib import Path


def model_tag(model_id):
    """Filesystem-safe short tag, e.g. google/gemma-2-9b-it -> gemma-2-9b-it."""
    return model_id.split("/")[-1]


def run_dir(cfg, model_id):
    """results/<model_tag>/ -- every model-specific artifact lives under here."""
    path = Path(cfg["paths"]["results_dir"]) / model_tag(model_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_digest(path, length=12):
    """Short content hash of a file, or 'missing'."""
    path = Path(path)
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def text_digest(*parts, length=12):
    """Short hash of arbitrary text/JSON-able parts."""
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode())
    return h.hexdigest()[:length]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def cache_fingerprint(model_id, prompt_text, data_paths, layers, positions, dtype):
    """Everything that can change a cached activation, as one dict + one short id.

    `cache_id` goes in the cache directory name, so changing the prompt, the data,
    the layer set, or the dtype writes to a NEW directory instead of silently
    resuming on top of activations that were produced differently.
    """
    parts = {
        "model": model_id,
        "prompt": text_digest(prompt_text),
        "data": {Path(p).name: file_digest(p) for p in data_paths},
        "layers": list(layers),
        "positions": list(positions),
        "dtype": dtype,
    }
    return parts, text_digest(parts, length=10)
