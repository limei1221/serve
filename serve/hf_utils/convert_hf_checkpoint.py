# adapted from gpt-fast/convert_hf_checkpoint.py

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional
from safetensors.torch import load_file as load_safetensors_file  # type: ignore
import torch

from serve.model import ModelArgs
from requests.exceptions import HTTPError  # type: ignore


def hf_download(repo_id: str, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download  # type: ignore

    os.makedirs(f"checkpoints/{repo_id}", exist_ok=True)
    try:
        snapshot_download(repo_id, local_dir=f"checkpoints/{repo_id}", local_dir_use_symlinks=False, token=hf_token)
    except HTTPError as e:
        if e.response.status_code == 401:
            print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
        else:
            raise e


# todo: proper args, do for llama 3 1b for now
config_1b = {
    "dim": 2048,
    "n_layers": 16,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "ffn_dim_multiplier": 1.5,
    "rope_base": 10000,
    "rope_scaling": dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
}

config_8b = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "rope_base": 500000,
    "ffn_dim_multiplier": 21 / 16,
    "rope_scaling": dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
}


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path,
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    if model_name.endswith("1B"):
        config_to_use = config_1b
    else:
        config_to_use = config_8b

    config = ModelArgs(**config_to_use)  # type: ignore

    # Load the json file containing weight mapping
    model_map_json_safetensors = checkpoint_dir / "model.safetensors.index.json"
    model_map_json_pytorch = checkpoint_dir / "pytorch_model.bin.index.json"
    model_map_json = None

    if not model_name.endswith("1B"):
        try:
            assert model_map_json_safetensors.is_file()
            model_map_json = model_map_json_safetensors
            print(f"Found safetensors index at {model_map_json_safetensors}")
        except AssertionError:
            print(f"{model_map_json_safetensors} not found")
        if model_map_json is None:
            try:
                assert model_map_json_pytorch.is_file()
                model_map_json = model_map_json_pytorch
                print(f"Found pytorch index at {model_map_json_pytorch}")
            except AssertionError:
                print(f"{model_map_json_pytorch} not found")

        if model_map_json is None:
            raise Exception("No model map found!")

        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)

    else:
        bin_index = {"weight_map": {1: "model.safetensors"}}

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.wo.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    def permute(w, n_head):
        dim = config.dim
        head_dim = config.dim // config.n_heads
        return w.view(n_head, 2, head_dim // 2, dim).transpose(1, 2).reshape(head_dim * n_head, dim)

    merged_result = {}
    for file in sorted(bin_files):
        if "safetensors" in str(file):
            state_dict = load_safetensors_file(str(file), device="cpu")
            merged_result.update(state_dict)
        else:
            state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=True)
            merged_result.update(state_dict)
    final_result: dict[str, torch.Tensor] = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            match = re.search(r"\d+", key)
            assert match is not None
            layer_num = match.group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        assert new_key is not None
        final_result[new_key] = value

    # for l3 1b
    if model_name.endswith("1B"):
        final_result["output.weight"] = final_result["tok_embeddings.weight"]

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config.n_heads)
            k = permute(k, config.n_kv_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")
    if "llama-3-" in model_name.lower() or "llama-3.1-" in model_name.lower():
        if "llama-3.1-405b" in model_name.lower():
            original_dir = checkpoint_dir / "original" / "mp16"
        else:
            original_dir = checkpoint_dir / "original"
        tokenizer_model = original_dir / "tokenizer.model"
        tokenizer_model_tiktoken = checkpoint_dir / "tokenizer.model"
        print(f"Copying {tokenizer_model} to {tokenizer_model_tiktoken}")
        shutil.copy(tokenizer_model, tokenizer_model_tiktoken)


def main(repo_id: str):
    checkpoint_dir = Path(f"checkpoints/{repo_id}")
    hf_token = os.environ.get("HF_TOKEN")
    hf_download(repo_id, hf_token)
    convert_hf_checkpoint(
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main("meta-llama/Llama-3.2-1B")
