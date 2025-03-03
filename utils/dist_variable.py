import os
import functools
from contextlib import contextmanager

import torch
import torch.distributed as dist



class _DistVarible:
    rank = 0
    world_size = 1
    local_rank = 0
    master_addr = os.environ.get("MASTER_ADDR", "")
    master_port = os.environ.get("MASTER_ADDR", "")
    def __init__(self):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # if not dist.is_initialized():
        #     self._init_dist()
    
    def _init_dist(self):
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )    

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def is_local_main_process(self):
        return self.local_rank == 0

    @property
    def n_gpus(self):
        return torch.cuda.device_count()

    @property
    def master_addr(self):
        return self.master_addr
    
    @property
    def master_port(self):
        return self.master_port
    
DistVarible = _DistVarible()


def rank0_only(func):
    
    @functools.wraps(func)
    def warpper(*args, **kwargs):
        if DistVarible.is_main_process:
            return func(*args, **kwargs)
        else:
            return None
    
    return warpper

@contextmanager
def rank0_context():
    if DistVarible.is_main_process:
        try: # exec context code
            yield
        finally:
            pass
    else: # do nothing
        yield from ()

@contextmanager
def rankn_context(n):
    if DistVarible.rank == n:
        try:
            yield
        finally:
            pass
    else:
        yield from ()