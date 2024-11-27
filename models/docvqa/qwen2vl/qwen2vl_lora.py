import os
from typing import List, Union,Sequence
from transformers import PreTrainedModel,PretrainedConfig



def print_rank0(msg):
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(msg)


# copy from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/model_utils/misc.py#L27
# 用于找到所有的Linear层用于lora训练
def find_all_linear_modules(model: "PreTrainedModel", freeze_vision_tower: bool) -> List[str]:
    r"""
    Finds all available modules to apply lora or galore.
    """
    model_type = getattr(model.config, "model_type", None)
    forbidden_modules = {"lm_head"}
    if model_type == "chatglm":
        forbidden_modules.add("output_layer")
    elif model_type == "internlm2":
        forbidden_modules.add("output")
    elif model_type in ["llava", "llava_next", "llava_next_video", "mllama", "paligemma", "video_llava"]:
        forbidden_modules.add("multi_modal_projector")
    elif model_type == "qwen2_vl":
        forbidden_modules.add("merger")

    if freeze_vision_tower:
        if model_type == "mllama":
            forbidden_modules.add("vision_model")
        elif model_type == "qwen2_vl":
            forbidden_modules.add("visual")
        else:
            forbidden_modules.add("vision_tower")

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    print_rank0("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)

# copy from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/model_utils/visual.py#L185
def patch_target_modules(
    config: "PretrainedConfig", freeze_vision_tower : bool, target_modules: Sequence[str]
) -> Union[str, List[str]]:
    r"""
    Freezes vision tower for VLM LoRA tuning.
    """
    model_type = getattr(config, "model_type", None)
    vit_model_type = getattr(getattr(config, "vision_config", None), "model_type", None)
    if freeze_vision_tower:
        if model_type in ["llava", "llava_next", "llava_next_video", "paligemma", "video_llava"]:
            return "^(?!.*vision_tower).*(?:{}).*".format("|".join(target_modules))
        elif model_type == "mllama":
            return "^(?!.*vision_model).*(?:{}).*".format("|".join(target_modules))
        elif model_type == "qwen2_vl":
            return "^(?!.*visual).*(?:{}).*".format("|".join(target_modules))
        else:
            return target_modules
    else:
        if model_type == "qwen2_vl":
            return "^(?!.*patch_embed).*(?:{}).*".format("|".join(target_modules))
        elif vit_model_type == "pixtral":
            return "^(?!.*patch_conv).*(?:{}).*".format("|".join(target_modules))
        else:
            return target_modules