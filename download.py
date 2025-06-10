# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
from huggingface_hub import snapshot_download
import os
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

save_dir = "/mnt/33t/cy/mllm_models/semantic_sam"
repo_id_list = ['HCMUE-Research/SAM-vit-h','openai/clip-vit-large-patch14','shi-labs/oneformer_ade20k_swin_large','shi-labs/oneformer_coco_swin_large','Salesforce/blip-image-captioning-large','CIDAS/clipseg-rd64-refined']
repo_id_list = ['shi-labs/oneformer_ade20k_swin_large','shi-labs/oneformer_coco_swin_large']
for repo_id in repo_id_list:
    cache_dir = save_dir + "/cache"
    #export HF_ENDPOINT="https://hf-mirror.com"
    snapshot_download(cache_dir=cache_dir,
    local_dir= os.path.join(save_dir , os.path.basename(repo_id)),
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    #allow_patterns=["*.json", "*.bin", "*.py", "*.md", "*.txt", '.safetensors', ],
    )
