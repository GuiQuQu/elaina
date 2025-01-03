import gradio as gr
import argparse
from transformers import set_seed
from functools import partial
import torch
from utils.register import registry_pycls_by_path
from arguments import get_config_from_args
from logger import logger

from deploy.gradio_utils import (
    load_model_with_checkpoint,
    custom_model_inference,
    build_preprocessor,
)


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


def count_images(message, history):
    num_images = len(message["files"])
    total_images = 0
    for message in history:
        if isinstance(message["content"], tuple):
            total_images += 1
    return f"You just uploaded {num_images} images, total uploaded: {total_images+num_images}"


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
        text = f"Pred Page Idx: {result[0][2]}\nImage Name: {image_name}\n Answer:{resp}"
        ret = {
            "text": text,
            "files": [image_path],
        }
        return ret
    else:
        return f"Answer:{resp}"


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this message! => " + data.value)
    else:
        print("You downvoted this message! => " + data.value)


examples = [
    {
        "text": "What is the Log-in No. ?",
        "files": [
            "/root/elaina/examples/images/fryn0081_p0.jpg",
            "/root/elaina/examples/images/fryn0081_p1.jpg",
            "/root/elaina/examples/images/fryn0081_p2.jpg",
            "/root/elaina/examples/images/fryn0081_p3.jpg",
            "/root/elaina/examples/images/fryn0081_p4.jpg",
            "/root/elaina/examples/images/fryn0081_p5.jpg",
            "/root/elaina/examples/images/fryn0081_p6.jpg",
            "/root/elaina/examples/images/fryn0081_p7.jpg",
            "/root/elaina/examples/images/fryn0081_p8.jpg",
            "/root/elaina/examples/images/fryn0081_p9.jpg",
        ],
    },
    {
        "text": "What is the personnel costs in the 4th year?",
        "files": [
            "/root/elaina/examples/images/hrfw0227_p22.jpg",
            "/root/elaina/examples/images/hrfw0227_p23.jpg",
        ],
    },
    {
        "text": "What is the name of the person in the CC field ?",
        "files": ["/root/elaina/examples/images/lflm0081_p0.jpg"],
    },
]


if __name__ == "__main__":
    args = get_args()
    config: dict = get_config_from_args(args)
    registry_paths = config["registry_paths"]
    for path in registry_paths:
        registry_pycls_by_path(path)
    if config.get("seed", None) is not None:
        set_seed(config["seed"])
    else:
        logger.warning("seed is None")
    server_name = args.server_name

    # load vqa model
    vqa_model = load_model_with_checkpoint(
        config["vqa_model"], config.get("vqa_checkpoint_path", None)
    )
    # load classify model
    classify_model = load_model_with_checkpoint(
        config.get("classify_model", None), config.get("classify_checkpoint_path", None)
    )
    # load preprocessor
    vqa_preprocessor = build_preprocessor(config["vqa_preprocess_config"])
    classify_preprocessor = build_preprocessor(
        config.get("classify_preprocess_config", None)
    )

    fn = partial(
        chat_fn, vqa_model, vqa_preprocessor, classify_model, classify_preprocessor
    )
    demo = gr.ChatInterface(
        title="MP-DocVQA Chatbot",
        # description="Chatbot for MP-DocVQA",
        fn=fn,
        multimodal=True,
        chatbot=gr.Chatbot(type="messages", height=600),
        textbox=gr.MultimodalTextbox(
            file_count="multiple",
            file_types=["image"],
            placeholder="Ask Question About Document ...",
            container=False,
            scale=7,
        ),
        type="messages",
        theme="ocean",
        examples=examples,
    )

    demo.launch(server_name=server_name)
