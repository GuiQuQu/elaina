import torch

def internvl2_concat_collator(batch):
    assert isinstance(batch, list)

    elem = batch[0]
    assert isinstance(elem, dict), f"elem type: {type(elem)}, expected type: dict"
    ret_batch = {}
    keys = list(elem.keys())
    for k in keys:
        if isinstance(elem[k], torch.Tensor):
            shape = elem[k].shape
            if all([d[k].shape == shape for d in batch]) and k not in [
                "pixel_values",
                "test_pixel_values",
                "image_flags",
            ]:
                ret_batch[k] = torch.stack([d[k] for d in batch], dim=0)
            else:
                ret_batch[k] = torch.cat([d[k] for d in batch], dim=0)
        else:
            ret_batch[k] = [d[k] for d in batch]
    return ret_batch