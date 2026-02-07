# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional


def hf_download(repo_id: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    if repo_id is None:
        raise ValueError("--repo_id is required")
    if hf_token is None:
        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )

    os.makedirs(f"checkpoints/{repo_id}", exist_ok=True)
    try:
        snapshot_download(repo_id, local_dir=f"checkpoints/{repo_id}", local_dir_use_symlinks=False, token=hf_token)
    except GatedRepoError:
        print(
            f"Cannot access gated repo: {repo_id}\n"
            "Make sure you've accepted the model license and authenticate via:\n"
            "  - export HF_TOKEN=...  (or HUGGINGFACE_HUB_TOKEN)\n"
            "  - or `huggingface-cli login`"
        )
        raise SystemExit(1)
    except HfHubHTTPError as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code in (401, 403):
            print(
                f"Unauthorized to download: {repo_id}\n"
                "Pass `--hf_token=...` or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
            )
            raise SystemExit(1)
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download data from HuggingFace Hub.')
    parser.add_argument('--repo_id', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Repository ID to download from.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token.')

    args = parser.parse_args()
    hf_download(args.repo_id, args.hf_token)
