from dataclasses import dataclass
import os

import torch

from serve.model import LlamaModel
from serve.kv_cache_paged import KVCacheIndexPaged
from serve.kv_cache_paged import BlockTableData

assert torch.cuda.is_available()
COMPILE_DECODE = int(os.getenv("COMPILE_DECODE", 0)) == 1
COMPILE_PREFILL = int(os.getenv("COMPILE_PREFILL", 0)) == 1
MAX_ALLOWED_BATCH_SIZE = 4000


@dataclass
class Request:
    input_ids: torch.Tensor
    input_length: int
    max_new_tokens: int
    generated_tokens: torch.Tensor = torch.tensor([], dtype=torch.long)
    logits: torch.Tensor | None = None


@dataclass
class RequestData:
    request_id: int
    input_ids: torch.Tensor
    generated_tokens: list[int]
    input_length: int
    max_new_tokens: int
    prefill_done: bool = False
    completed: bool = False
    logits: torch.Tensor | None = None


def return_request(request_data: RequestData, return_logits: bool) -> Request:
    return Request(
        input_ids=request_data.input_ids,
        input_length=request_data.input_length,
        max_new_tokens=request_data.max_new_tokens,
        generated_tokens=torch.tensor(request_data.generated_tokens, dtype=torch.long),
        logits=request_data.logits if return_logits else None,
    )


# from gpt-fast
def multinomial_sample_one_no_sync(
    probs_sort,
):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_next_token(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 0.9):
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        return multinomial_sample_one_no_sync(probs)


def forward_prefill_sample(
    model: LlamaModel,
    tokens: torch.Tensor,
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table_data: BlockTableData,
    max_seqlen: int,
    temperature: float,
):
    logits = model(
        tokens,
        kv_caches,
        positions,
        seq_lens,
        block_table_data,
        max_seqlen,
        mode="prefill",
    )
    next_tokens = sample_next_token(logits, temperature=temperature).flatten()
    return logits, next_tokens


def forward_decode_sample(
    model: LlamaModel,
    tokens: torch.Tensor,
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    seq_lens: torch.Tensor,
    block_table_data: BlockTableData,
    max_seqlen: int,
    temperature: float,
):
    positions = (seq_lens - 1).unsqueeze(1)
    logits = model(
        tokens,
        kv_caches,
        positions,
        seq_lens,
        block_table_data,
        max_seqlen,
        mode="decode",
    )
    next_tokens = sample_next_token(logits, temperature=temperature).flatten()
    return logits, next_tokens


def prefill(
    model: LlamaModel,
    requests: list[RequestData],
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    kv_cache_ix: KVCacheIndexPaged,
    temperature: float,
):
    # todo: could make few % faster by more efficient setup
    # or overlapping disjoint prefills (don't need to wait for the last one)
    DEVICE = f"cuda:{torch.distributed.get_rank()}" if torch.distributed.is_initialized() else "cuda"

    req_ids = [r.request_id for r in requests]
    req_lens = [r.input_length for r in requests]
    with torch.autograd.profiler.record_function("allocate_requests"):
        kv_cache_ix.allocate_requests_batched(req_ids, req_lens)

    with torch.autograd.profiler.record_function("create_block_table"):
        block_table_data = kv_cache_ix.get_block_table_data(req_ids)

    tokens = torch.cat([r.input_ids for r in requests], dim=-1).to(DEVICE)
    seq_lens = torch.tensor(req_lens, dtype=torch.int32, device=DEVICE)
    positions = torch.cat([torch.arange(r.input_length) for r in requests], dim=-1).to(DEVICE)

    # avoid recompiles by marking dynamic from the start
    for _input_tensor in [tokens, positions, seq_lens, block_table_data.block_table_index, block_table_data.block_table]:
        torch._dynamo.mark_dynamic(_input_tensor, 0)
        # to avoid recompilation on batch_size == 1
        torch._dynamo.decorators.mark_unbacked(_input_tensor, 0)
    torch._dynamo.mark_dynamic(tokens, 1)
    torch._dynamo.mark_dynamic(positions, 1)
    torch._dynamo.mark_dynamic(seq_lens, 1)

    # todo: this causes a recompile, but calculating it within the compiled code
    # crashes dynamo (bug?)
    max_seqlen = int(torch.max(seq_lens))

    logits, next_tokens = forward_prefill_sample(
        model,
        tokens,
        kv_caches,
        positions,
        seq_lens,
        block_table_data,
        max_seqlen,
        temperature,
    )

    next_tokens_list = next_tokens.tolist()
    for req, tok in zip(requests, next_tokens_list):
        req.generated_tokens = [tok]
        req.prefill_done = True

    return logits


def decode_one_token(
    model: LlamaModel,
    requests: list[RequestData],
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    kv_cache_ix: KVCacheIndexPaged,
    temperature: float,
):
    DEVICE = model.device
    req_lens = [r.input_length + len(r.generated_tokens) for r in requests]
    req_ids = [r.request_id for r in requests]

    with torch.autograd.profiler.record_function("allocate_requests"):
        kv_cache_ix.allocate_requests_batched(req_ids, req_lens)

    with torch.autograd.profiler.record_function("create_block_table_index"):
        block_table_data = kv_cache_ix.get_block_table_data(req_ids)

    last_tokens = [r.generated_tokens[-1] for r in requests]
    tokens = torch.tensor(last_tokens, dtype=torch.int32, device=DEVICE).view(-1, 1)
    seq_lens = torch.tensor(req_lens, dtype=torch.int32, device=DEVICE)

    if COMPILE_DECODE:
        for _input_tensor in [tokens, seq_lens, block_table_data.block_table_index, block_table_data.block_table]:
            torch._dynamo.mark_dynamic(_input_tensor, 0)
            # to avoid recompilation on batch_size == 1
            torch._dynamo.decorators.mark_unbacked(_input_tensor, 0)

    max_seqlen = 1

    logits, next_tokens = forward_decode_sample(
        model,
        tokens,
        kv_caches,
        seq_lens,
        block_table_data,
        max_seqlen,
        temperature,
    )

    with torch.autograd.profiler.record_function("update_requests"):
        next_tokens_list = next_tokens.tolist()
        for ix, tok in enumerate(next_tokens_list):
            requests[ix].generated_tokens.append(tok)

    return logits


def process_completed_requests(
    current_step_requests: list[RequestData],
    finished_requests: list[RequestData],
    kv_cache_ix: KVCacheIndexPaged,
    logits=None,
    return_logits=False,
) -> int:
    num_free_tokens = 0
    completed_indices = []
    for ix, r in enumerate(current_step_requests):
        if len(r.generated_tokens) >= r.max_new_tokens:
            r.completed = True
            finished_requests.append(r)
            with torch.autograd.profiler.record_function("free_request"):
                kv_cache_ix.free_request(r.request_id)
            completed_indices.append(ix)
            num_free_tokens += len(r.input_ids) + r.max_new_tokens
            if return_logits and logits is not None:
                r.logits = logits[ix]

    for ix in sorted(completed_indices, reverse=True):
        current_step_requests.pop(ix)

    return num_free_tokens


@torch.inference_mode()
def generate(
    model: LlamaModel,
    requests: list[Request],
    temperature: float = 0.0,
    return_logits: bool = False,
) -> list[Request]:
    device = f"cuda:{torch.distributed.get_rank()}" if torch.distributed.is_initialized() else "cuda"

    blocksize_k = 64
    if not model.kv_caches_init:
        model.create_kv_cache(blocksize_k=blocksize_k, max_batch_size=MAX_ALLOWED_BATCH_SIZE, device=device)
    kv_caches, kv_cache_ix, num_kv_blocks = model.kv_caches, model.kv_cache_ix, model.num_kv_blocks

    finished_requests: list[RequestData] = []
    # todo: make this a deque
    prefill_queue = [
        RequestData(
            request_id=i,
            input_ids=r.input_ids,
            generated_tokens=[],
            input_length=r.input_length,
            max_new_tokens=r.max_new_tokens,
        )
        for i, r in enumerate(requests)
    ]

    max_prefill_tokens = 2048
    decode_queue: list[RequestData] = []
    max_decode_tokens = num_kv_blocks * blocksize_k // model.config.n_kv_heads
    spent_decode_tokens = 0

    while decode_queue or prefill_queue:
        spent_prefill_tokens = 0
        num_requests_to_prefill = 0
        for r in prefill_queue:
            spent_prefill_tokens += len(r.input_ids)
            required_decode_tokens = len(r.input_ids) + r.max_new_tokens
            # todo: the 90% threshold prevents lots of small bsz prefills, chunked prefill will supercede though
            if spent_decode_tokens + required_decode_tokens > 0.9 * max_decode_tokens or spent_prefill_tokens > max_prefill_tokens:
                break
            spent_decode_tokens += required_decode_tokens
            num_requests_to_prefill += 1

        # todo: can these slices be slow? (with mega long list)
        current_step_requests, prefill_queue = prefill_queue[:num_requests_to_prefill], prefill_queue[num_requests_to_prefill:]
        if current_step_requests:
            # print(f"Prefilling {len(current_step_requests)} requests")
            with torch.autograd.profiler.record_function("prefill"):
                logits = prefill(model, current_step_requests, kv_caches, kv_cache_ix, temperature)
            decode_queue.extend(current_step_requests)
        else:
            # print(f"Decoding {len(decode_queue)} requests")
            current_step_requests = decode_queue
            with torch.autograd.profiler.record_function("generate_next_token"):
                logits = decode_one_token(model, current_step_requests, kv_caches, kv_cache_ix, temperature)

        num_free_tokens = process_completed_requests(
            current_step_requests,
            finished_requests,
            kv_cache_ix,
            logits=logits,
            return_logits=return_logits,
        )
        spent_decode_tokens -= num_free_tokens

    returned_requests = [return_request(r, return_logits) for r in sorted(finished_requests, key=lambda x: x.request_id)]

    del kv_caches, kv_cache_ix
    torch.cuda.empty_cache()
    return returned_requests


if COMPILE_PREFILL:
    forward_prefill_sample = torch.compile(forward_prefill_sample)

if COMPILE_DECODE:
    forward_decode_sample = torch.compile(forward_decode_sample)
