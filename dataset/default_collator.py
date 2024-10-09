
import torch

# elem type: list of int,float,str
# elem type: list of torch.Tensor
# elem type: list of [[1,2,3],[4,5,6],[7,8,9]]  => 
# elem type: list of dict

def default_collate(batch):
    """
        batch : list of dict
        return : dict
        stack all the torch.Tensor in the dict, and keep the other type as list
    """
    assert isinstance(batch, list)
    elem = batch[0]
    assert isinstance(elem, dict), f"elem type: {type(elem)}, expected type: dict"

    ret_batch = {}
    keys = list(elem.keys())
    for k in keys:
        if isinstance(elem[k], torch.Tensor):
            ret_batch[k] = torch.stack([d[k] for d in batch], dim=0)
        else:
            ret_batch[k] = [d[k] for d in batch]
    return ret_batch


def default_collate2(batch):
    assert isinstance(batch, list)
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, dim=0)
    if isinstance(elem, int) or isinstance(elem, float) or isinstance(elem, str):
        return batch
    if isinstance(elem, list):
        return [default_collate2(d) for d in batch]
    if isinstance(elem, dict):
        keys = list(elem.keys())
        ret = dict()
        for k in keys:
            ret[k] = default_collate2([d[k] for d in batch])
        return ret