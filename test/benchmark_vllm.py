import os
import time
import subprocess
import random
import torch

from vllm import LLM, SamplingParams, TokensPrompt


def generate_requests(input_length, output_length, num_sequences):
    random.seed(1352)
    torch.manual_seed(1352)
    seqs = []
    params = []
    for _ in range(num_sequences):
        _input_len = random.randint(1, input_length)
        _output_len = random.randint(1, output_length)
        # _input_len = input_length
        # _output_len = output_length
        seq_sampling_params = SamplingParams(temperature=0.0, min_tokens=_output_len, max_tokens=_output_len)
        seq = torch.randint(0, 10000, (_input_len,), dtype=torch.long)
        seq = TokensPrompt(prompt_token_ids=seq)
        seqs.append(seq)
        params.append(seq_sampling_params)
    return seqs, params


NUM_SEQUENCES = 400
prompts, sampling_params = generate_requests(256, 256, NUM_SEQUENCES)


llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    tokenizer="meta-llama/Llama-3.2-1B",
    device="cuda",
    dtype="float16",
)


def fn():
    outputs = llm.generate(prompts, sampling_params)
    return outputs


# warmup
for _ in range(1):
    fn()

torch.cuda.synchronize()
st = time.perf_counter()
fn()
torch.cuda.synchronize()
et = time.perf_counter()
print(f"time taken: {(et - st)}s")


fn_name = "vllm_generate"

with torch.profiler.profile() as prof:
    for _ in range(1):
        result = fn()

timestamp = time.strftime("%H%M")
if not os.path.exists("/tmp/traces"):
    os.makedirs("/tmp/traces")

trace_path = f"/tmp/traces/vllm_trace_{fn_name}_{timestamp}.json"
prof.export_chrome_trace(trace_path)
subprocess.run(["tar", "-czvf", f"/tmp/traces/trace_{fn_name}_{timestamp}.tar.gz", trace_path])
print(f"Trace saved to {trace_path}.tar.gz")
