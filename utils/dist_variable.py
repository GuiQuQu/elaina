import os
import torch

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