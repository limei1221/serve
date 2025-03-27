import time
import subprocess
import random
import os

import torch

from serve.run import generate, Request
from serve.utils import load_model


def generate_requests(input_length: int, output_length: int, num_sequences: int) -> list[Request]:
    random.seed(1352)
    torch.manual_seed(1352)
    seqs = []
    for _ in range(num_sequences):
        _input_len = random.randint(1, input_length)
        _output_len = random.randint(1, output_length)
        # _input_len = input_length
        # _output_len = output_length
        seq = torch.randint(0, 10000, (_input_len,), dtype=torch.long)
        req = Request(input_ids=seq, input_length=_input_len, max_new_tokens=_output_len)
        seqs.append(req)
    return seqs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    # model_str = "meta-llama/Llama-3.2-1B"
    # dtype = torch.float16
    model_str = "meta-llama/Llama-3.1-8B"
    dtype = torch.bfloat16

    device = "cuda"
    model, tokenizer = load_model(model_name=model_str, device=device, dtype=dtype)

    input_length = 256
    output_length = 256
    num_sequences = 2000
    temperature = 0.6

    requests = generate_requests(
        input_length=input_length,
        output_length=output_length,
        num_sequences=num_sequences,
    )

    def fn():
        return generate(model, requests, temperature)

    fn_name = f"generate_{input_length}_{output_length}_{num_sequences}"

    # warmup
    # print("warming up")
    # generate(model, requests[:10], temperature)

    print("doing first run")
    st = time.perf_counter()
    fn()
    et = time.perf_counter()
    print(f"Warmup time: {et - st:.4f} seconds")

    if not args.profile_only:
        torch.cuda.synchronize()
        st = time.perf_counter()
        responses = fn()
        torch.cuda.synchronize()
        et = time.perf_counter()
        elapsed_time = et - st
        total_generated_tokens = sum(len(req.generated_tokens) for req in responses)
        tokens_per_second = total_generated_tokens / elapsed_time
        print(f"Time taken: {elapsed_time:.4f} seconds")
        # todo prefill/decode throughput
        print(f"Throughput: {tokens_per_second:.2f} tokens/second")

    if args.profile:
        with torch.profiler.profile() as prof:
            with torch.autograd.profiler.record_function(fn_name):
                result = fn()

        timestamp = time.strftime("%H%M")
        if not os.path.exists("/tmp/traces"):
            os.makedirs("/tmp/traces")

        trace_path = f"/tmp/traces/trace_{fn_name}_{timestamp}.json"
        prof.export_chrome_trace(trace_path)
        subprocess.run(
            [
                "tar",
                "-czvf",
                f"/tmp/traces/trace_{fn_name}_{timestamp}.tar.gz",
                trace_path,
            ]
        )
        print(f"Trace saved to {trace_path}.tar.gz")

    print(torch._dynamo.utils.compile_times("str"))
