from vllm import LLM, SamplingParams  # type: ignore


prompts = ["The capital of France is known for"]
sampling_params = SamplingParams(temperature=0.6, max_tokens=500)

llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    tokenizer="meta-llama/Llama-3.2-1B",
    device="cuda",
    dtype="float16",
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
