import torch
from torch import nn
from torch import distributed as dist

from serve.model import LlamaModel

class RowParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, dtype: torch.dtype, tp_size: int, rank:int, input_parallel: bool, reduce_out: bool):
        super().__init__()
        assert d_in % tp_size == 0
        self.weight = nn.Parameter(
            torch.empty(d_out, d_in//tp_size, dtype=dtype)
        )
        self.input_parallel = input_parallel
        self.reduce_out = reduce_out
        self.tp_size = tp_size
        self.rank = rank

    def forward(self, x):
        # assumes input is already parallel
        if not self.input_parallel:
            x = x.chunk(self.tp_size, -1)[self.rank]
        out = x @ self.weight.T
        if self.reduce_out:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

class ColumnParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, dtype: torch.dtype, tp_size: int, rank: int, all_gather_out: bool):
        super().__init__()
        self.tp_size = tp_size
        self.rank = rank
        assert d_out % tp_size == 0
        self.weight = nn.Parameter(torch.empty(d_out//tp_size, d_in, dtype=dtype))
        self.d_out = d_out
        self.all_gather_out = all_gather_out

    def forward(self, x: torch.Tensor):
        out = x @ self.weight.T
        if self.all_gather_out:
            out_full = out.new_empty(*x.shape[:-1], self.d_out)
            dist.all_gather_into_tensor(out_full, out)
            out = out_full
        return out


def convert_row_parallel(layer: nn.Linear, tp_size: int, rank: int, input_parallel: bool, reduce_out: bool) -> RowParallelLinear:
    full_weight = layer.weight
    d_out, d_in = full_weight.shape
    local_weight = full_weight.chunk(tp_size, dim=1)[rank]
    new_layer = RowParallelLinear(d_in, d_out, full_weight.dtype, tp_size, rank, input_parallel, reduce_out)
    new_layer.weight = nn.Parameter(local_weight)
    return new_layer


def convert_col_parallel(layer: nn.Linear, tp_size: int, rank: int, all_gather_out: bool) -> ColumnParallelLinear:
    full_weight = layer.weight
    d_out, d_in = full_weight.shape
    local_weight = full_weight.chunk(tp_size, dim=0)[rank]
    new_layer = ColumnParallelLinear(d_in, d_out, full_weight.dtype, tp_size, rank, all_gather_out)
    new_layer.weight = nn.Parameter(local_weight)
    return new_layer


def convert_qkv_col_parallel(layer: nn.Linear, tp_size: int, rank: int, n_heads: int, n_kv_heads: int, head_dim: int) -> ColumnParallelLinear:
    full_weight = layer.weight
    d_out, d_in = full_weight.shape

    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    n_local_heads = n_heads // tp_size
    n_local_kv_heads = n_kv_heads // tp_size

    wq, wk, wv = torch.split(full_weight, [q_dim, kv_dim, kv_dim], dim=0)
    local_wq = wq[rank * n_local_heads * head_dim: (rank + 1) * n_local_heads * head_dim]
    local_wk = wk[rank * n_local_kv_heads * head_dim: (rank+1) * n_local_kv_heads * head_dim]
    local_wv = wv[rank * n_local_kv_heads * head_dim: (rank+1) * n_local_kv_heads * head_dim]
    local_wqkv = torch.cat([local_wq, local_wk, local_wv], dim=0).to(full_weight.device)
    new_layer = ColumnParallelLinear(d_in, d_out, full_weight.dtype, tp_size, rank, all_gather_out=False)
    new_layer.weight = nn.Parameter(local_wqkv)
    return new_layer



def apply_tp(model: LlamaModel):
    """ edit model in place to have those layers """

    # basic config for tp_size == world_size
    tp_size = dist.get_world_size()
    rank = dist.get_rank()

    for layer in model.layers:
        mlp = layer.mlp
        mlp.w1 = convert_col_parallel(mlp.w1, tp_size, rank, all_gather_out=False)
        mlp.w2 = convert_row_parallel(mlp.w2, tp_size, rank, input_parallel=True, reduce_out=True)
        mlp.w3 = convert_col_parallel(mlp.w3, tp_size, rank, all_gather_out=False)

        attn = layer.attn
        d_h = model.config.dim//model.config.n_heads
        attn.wqkv = convert_qkv_col_parallel(attn.wqkv, tp_size, rank, model.config.n_heads, model.config.n_kv_heads, d_h)
        attn.wo = convert_row_parallel(attn.wo, tp_size, rank, input_parallel=True, reduce_out=True)
