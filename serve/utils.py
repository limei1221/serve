from pathlib import Path

import torch
from torch import distributed as dist
from transformers import AutoTokenizer  # type: ignore

from serve.model import ModelArgs, LlamaModel
from serve.hf_utils.convert_hf_checkpoint import download_and_convert
from serve.tp import apply_tp

config_1b = {
    "dim": 2048,
    "n_layers": 16,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "ffn_dim_multiplier": 1.5,
    "rope_base": 500000,
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


name_to_config = {
    "meta-llama/Llama-3.2-1B": config_1b,
    "meta-llama/Llama-3.1-8B": config_8b,
}


def load_model(model_name, device, dtype, tp_dim: int = 1):
    print(f"loading model {model_name}...")
    path = Path(f"checkpoints/{model_name}/model.pth")
    if not path.exists():
        # raise FileNotFoundError(f"Model checkpoint not found at {path}")
        print("model not found, converting from hf")
        from torch import distributed as dist
        if dist.is_initialized():
            if dist.get_rank() == 0:
                download_and_convert(model_name)
            dist.barrier()
        else:
            download_and_convert(model_name)
    config = name_to_config[model_name]
    model = _load_model(config, path, device, dtype, tp_dim)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model finished loading")
    return model, tokenizer


def _load_model(config, checkpoint_path, device, dtype, tp_dim: int = 1):
    with torch.device("meta"):
        config = ModelArgs(**config)
        model = LlamaModel(config, dtype=dtype)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    if tp_dim > 1:
        apply_tp(model)

    model = model.to(device=device, dtype=dtype)

    model.compute_freqs_cis(device)

    if torch.cuda.is_available():
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"rank: {rank}; mem: {torch.cuda.memory_allocated()/1e9:.2f}")
    return model.eval()
