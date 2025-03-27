import os

os.environ["DEVICE"] = "cuda"
os.environ["COMPILE_DECODE"] = "1"
os.environ["COMPILE_PREFILL"] = "1"

# optional torch config
# import torch
# import torch._inductor.config
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.assert_indirect_indexing = False
