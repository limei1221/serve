# serve

## A minimal LLM inference engine.
I wanted to see how close you can get to SGLang/vLLM performance with a minimal codebase.


### Key points:
- Fully leverages `torch.compile`:
    - Both prefill and decode are fully compiled
    - Dynamic shapes supported without recompilation
- Handwritten Triton kernels for paged attention
- Matches SGLang performance on e.g. llama-8b on A100 offline batch inference


### Performance Benchmark

For llama-8b on A100 40GB

`python3 serve/benchmark.py` took 54.2s

sglang ([code from here](https://github.com/sgl-project/sglang/tree/52029bd1e3e30a1474ead6bddd20d79a162ebc6f/benchmark/blog_v0_2)) took 55.3s:

    python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B --enable-torch-compile --disable-radix-cache --mem-fraction-static 0.7

    python -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 2000 --random-input 256 --random-output 256

(todo: more benchmarks)


###  Future Steps:
- Chunked Prefill
- Tensor Parallelism
- Overlap decode steps/zero-overhead scheduler
- Decouple Attention kernels blocksize from paged attention blocksize
- More tests