import torch
from utils.register import Register

@Register(name="internvl2_concat_collator")
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

@Register(name="qwen2vl_concat_collator")
def qwen2vl_concat_collator(batch):
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
                "classify_pixel_values",
                "vqa_pixel_values",
                "test_pixel_values",
                "test_vqa_pixel_values",
                "test_classify_pixel_values",
            ]:
                ret_batch[k] = torch.stack([d[k] for d in batch], dim=0)
            else:
                ret_batch[k] = torch.cat([d[k] for d in batch], dim=0)
        else:
            ret_batch[k] = [d[k] for d in batch]
    return ret_batch


@Register(name="default_concat_collator")
def default_collator(batch):
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

@Register(name="triplet_collator")
def triplet_collator(batch):
    elem = batch[0]
    assert isinstance(elem, dict), f"elem type: {type(elem)}, expected type: dict"
    ret_batch = {}
    keys = list(elem.keys())
    for k in keys:
        if isinstance(elem[k], torch.Tensor):
            shape = elem[k].shape
            if all([d[k].shape == shape for d in batch]) and k not in [
                "negative_images"
            ]:
                ret_batch[k] = torch.stack([d[k] for d in batch], dim=0)
            else:
                ret_batch[k] = [d[k] for d in batch]
        else:
            ret_batch[k] = [d[k] for d in batch]
    return ret_batch