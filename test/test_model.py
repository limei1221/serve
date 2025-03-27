import sys
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

# for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serve.run import generate, Request
from serve.utils import load_model


def generate_toks(requests, model_str, device, dtype, temperature=0.0):
    model, _ = load_model(model_name=model_str, device=device, dtype=dtype)
    ret = generate(model, requests, return_logits=True, temperature=temperature)
    ret_toks = [r.generated_tokens for r in ret]
    ret_logits = [r.logits for r in ret]
    return ret_toks, ret_logits


def generate_hf_reference(requests, model_str, dtype, temperature=0.0):
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_str, device_map="auto", torch_dtype=dtype)
    # model.to(device)

    ret_seqs, ret_logits = [], []
    for req in requests:
        ret = model.generate(
            req.input_ids.unsqueeze(0).to(device),
            max_new_tokens=req.max_new_tokens,
            do_sample=False,  # equivalent to temp = 0
            temperature=0.0,
            top_p=None,
            return_dict_in_generate=True,
            output_logits=True,
        )
        ret_seqs.append(ret.sequences[:, len(req.input_ids) :])
        ret_logits.append(ret.logits[-1])
    return ret_seqs, ret_logits


def test_model_output_consistency():
    sequences = [
        "The capital of France is Paris",
        "The KV cache is",
        "In Germany, the sky is not the same colour as the",
    ]
    # max_new_tokens_per_seq = [300,180,130]
    max_new_tokens_per_seq = [50, 70, 40]

    # model_str = "meta-llama/Llama-3.2-1B"
    # dtype = torch.float16
    model_str = "meta-llama/Llama-3.1-8B"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    device = "cuda"

    requests = [
        Request(
            input_ids=torch.tensor(tokenizer.encode(seq), dtype=torch.long),
            input_length=len(tokenizer.encode(seq)),
            max_new_tokens=max_new_tokens,
        )
        for seq, max_new_tokens in zip(sequences, max_new_tokens_per_seq)
    ]

    toks, logits = generate_toks(requests, model_str, device, dtype)
    toks_hf, logits_hf = generate_hf_reference(requests, model_str, dtype)

    for t, tr in zip(toks, toks_hf):
        torch.testing.assert_close(t.flatten().cpu(), tr.flatten().cpu())
        print("./")
    print("tokens match")

    # todo: clean this up
    for req, l, lh in zip(requests, logits, logits_hf):
        ix = req.input_length
        print(f"Number of input tokens: {ix}")
        print(f"{l.shape=}")

        print(l)
        print(lh)
        print(l.argmax(dim=-1))
        print(lh.argmax(dim=-1))
        #
        delta = lh.to(l.dtype).to(l.device) - l
        print(f"{delta.abs().max()=}")
        torch.testing.assert_close(l, lh.to(l.dtype).to(l.device), atol=2e-2, rtol=1e-4)
        print("match")
    print("logits match")


def test_model_coherent():
    model_str = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    device = "cuda"
    dtype = torch.float16

    requests = ["The capital of France is known for", "A monotonic function is defined as"]
    requests = [
        Request(input_ids=torch.tensor(tokenizer.encode(seq), dtype=torch.long), input_length=len(tokenizer.encode(seq)), max_new_tokens=500)
        for seq in requests
    ]
    toks, logits = generate_toks(requests, model_str, device, dtype, temperature=0.6)
    print(tokenizer.decode(toks[0]))
    print(tokenizer.decode(toks[1]))

    toks, _ = generate_hf_reference(requests, model_str, dtype)
    print(tokenizer.decode(toks[0][0]))
    print(tokenizer.decode(toks[1][0]))


if __name__ == "__main__":
    test_model_output_consistency()
    # test_model_coherent()
