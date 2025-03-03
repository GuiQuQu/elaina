import gradio as gr
import argparse
from transformers import set_seed
from functools import partial
import torch
from utils.register import registry_pycls_by_path
from arguments import get_config_from_args


from deploy.gradio_utils import (
    load_model_with_checkpoint,
    custom_model_inference,
    build_preprocessor,
)


def get_args():
    parser = argparse.ArgumentParser(description="elania gradio interface")
    parser.add_argument(
        "--config_file",
        "--config",
        type=str,
        default="deploy/config/internvl2_classify_qwen2vl_vqa_autodl.json",
        help="The config file",
    )
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    args = parser.parse_args()
    return args


def check_cuda_memory(device):
    # 获取GPU内存总量（字节）
    total_memory = torch.cuda.get_device_properties(device).total_memory
    # 获取已使用的GPU内存（字节）
    allocated_memory = torch.cuda.memory_allocated(device)
    # 获取缓存的GPU内存（字节）
    cached_memory = torch.cuda.memory_reserved(device)
    total_memory_str = "{:.2f}GB".format(total_memory / (1024**3))
    allocated_memory_str = "{:.2f}GB".format(allocated_memory / (1024**3))
    cached_memory_str = "{:.2f}GB".format(cached_memory / (1024**3))
    ret = (
        f"[alloc:{allocated_memory_str} cached:{cached_memory_str}|{total_memory_str}]"
    )
    return ret


def print_cuda0_memory(func):
    def warpper(*args, **kwargs):
        device = torch.device("cuda:0")
        print(f"Before {func.__name__} {check_cuda_memory(device)}")
        result = func(*args, **kwargs)
        print(f"After {func.__name__} {check_cuda_memory(device)}")
        return result

    return warpper


@print_cuda0_memory
def chat_fn(
    vqa_model, vqa_preprocessor, classify_model, classify_preprocessor, message, history
):
    question = message["text"]
    image_paths = message["files"]
    if len(image_paths) == 0:
        return "Please upload an image"
    elif len(image_paths) > 1 and (
        classify_model is None or classify_preprocessor is None
    ):
        return "classify model is None,not support multiple images"

    if len(image_paths) > 1:
        # 多图分类
        result = []
        for i, image_path in enumerate(image_paths):
            clasiify_input = dict(
                question=question,
                image_path=image_path,
            )
            score = custom_model_inference(
                classify_model, classify_preprocessor, clasiify_input
            )
            result.append((image_path, score, i))
        result = sorted(result, key=lambda x: x[1], reverse=True)
        image_path = result[0][0]
    else:
        image_path = image_paths[0]

    # 第二阶段回答
    vqa_input = dict(
        question=question,
        image_path=image_path,
    )
    resp = custom_model_inference(vqa_model, vqa_preprocessor, vqa_input)
    if len(image_paths) > 1:
        image_name = image_path.split("/")[-1]
        return (
            f"Pred Page Idx: {result[0][2]}\nImage Name: {image_name}\n Answer:{resp}"
        )
    else:
        return f"Answer:{resp}"


def _launch_demo(args, pred_fn):
    title = "MP-DocVQA Chatbot"
    with gr.Blocks() as demo:
        gr.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>{title}</h1>")

        chatbot = gr.Chatbot(
            label="mp-docvqa-chatbot", 
            elem_classes="control-height", 
            height=750,
            type="messages"
        )
        query = gr.MultimodalTextbox(
            file_count="multiple",
            file_types=["image"],
            placeholder="Ask Question About Document ...",
            container=False,
        )
        clear = gr.ClearButton([chatbot,query],label="🧹 Clear")

        query.submit(pred_fn, [query, chatbot], [query, chatbot])
