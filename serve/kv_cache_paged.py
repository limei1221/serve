from dataclasses import dataclass

import torch


def cdiv(a, b):
    return -(a // -b)


@dataclass
class BlockTableData:
    block_table: torch.Tensor
    block_table_index: torch.Tensor
    block_table_size: int


class KVCacheIndexPaged:
    def __init__(self, num_blocks: int, blocksize_k: int, num_heads: int, num_layers: int, max_context_length: int, max_batch_size: int, device: str):
        self.num_blocks = num_blocks
        self.blocksize_k = blocksize_k
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        max_num_blocks = max_context_length // blocksize_k

        self.block_table = torch.empty(max_batch_size, num_heads, max_num_blocks, dtype=torch.int32, device=device)
        self.req_to_block_table_index: dict[int, int] = {}
        self.req_to_num_blocks: dict[int, int] = {}
        self.available_blocks = list(range(num_blocks))[::-1]
        self.available_batch_indexes = list(range(max_batch_size))[::-1]

    def allocate_requests_batched(self, request_ids: list[int], seq_lens: list[int]) -> None:
        req_block_table_ix = []
        req_new_blocks = []
        req_cur_blocks = []

        with torch.autograd.profiler.record_function("count_blocks"):
            for request_id, seq_len in zip(request_ids, seq_lens):
                if request_id not in self.req_to_block_table_index:
                    self.req_to_block_table_index[request_id] = self.available_batch_indexes.pop()
                    self.req_to_num_blocks[request_id] = 0

                cur_blocks = self.req_to_num_blocks[request_id]
                if seq_len > cur_blocks * self.blocksize_k:
                    req_block_table_ix.append(self.req_to_block_table_index[request_id])
                    new_blocks = cdiv(seq_len, self.blocksize_k) - cur_blocks
                    req_new_blocks.append(new_blocks)
                    req_cur_blocks.append(cur_blocks)
                    self.req_to_num_blocks[request_id] += new_blocks

        if not req_block_table_ix:
            return

        total_new_blocks = sum(req_new_blocks)

        new_block_ids = []
        num_new_blocks = self.num_heads * total_new_blocks
        with torch.autograd.profiler.record_function("get_new_block_ids"):
            new_block_ids = self.available_blocks[-num_new_blocks:]
            del self.available_blocks[-num_new_blocks:]

        new_block_ids_tensor = torch.tensor(new_block_ids, dtype=torch.int32, device=self.device)
        new_block_ids_tensor = new_block_ids_tensor.view(self.num_heads, total_new_blocks)
        new_block_ids_tensor_list = new_block_ids_tensor.split(req_new_blocks, dim=1)

        for index, new_blocks, cur_blocks, new_block_ids_tensor in zip(req_block_table_ix, req_new_blocks, req_cur_blocks, new_block_ids_tensor_list):
            self.block_table[index, :, cur_blocks : cur_blocks + new_blocks] = new_block_ids_tensor

    def free_request(self, request_id: int) -> None:
        # todo: batch this into one data transfer / more generally speed up
        req_batch_id = self.req_to_block_table_index[request_id]
        block_ids = self.block_table[req_batch_id, :, : self.req_to_num_blocks[request_id]].tolist()
        self.available_blocks.extend([x for xs in block_ids for x in xs])
        self.available_batch_indexes.append(req_batch_id)
        del self.req_to_block_table_index[request_id]
        del self.req_to_num_blocks[request_id]

    def get_block_table_data(self, request_ids: list[int]) -> BlockTableData:
        block_table_index = torch.tensor([self.req_to_block_table_index[r] for r in request_ids], dtype=torch.int32, device=self.device)
        return BlockTableData(self.block_table, block_table_index, self.blocksize_k)


def setup_kv_cache_paged(
    n_blocks: int,
    blocksize_k: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    max_context_length: int,
    max_batch_size: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], KVCacheIndexPaged]:
    kv_caches = [
        (
            torch.empty(n_blocks, blocksize_k, head_dim, dtype=dtype, device=device),
            torch.empty(n_blocks, blocksize_k, head_dim, dtype=dtype, device=device),
        )
        for _ in range(num_layers)
    ]
    kv_cache_ix = KVCacheIndexPaged(n_blocks, blocksize_k, num_heads, num_layers, max_context_length, max_batch_size, device)
    return kv_caches, kv_cache_ix
