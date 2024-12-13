import gradio as gr
import argparse
import random
from transformers import set_seed
from functools import partial

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
        "--config_file", "--config", type=str, default=None, help="The config file"
    )
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    args = parser.parse_args()
    return args


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
        return f"Pred Page Idx: {result[0][2]}\nPage Path: {image_path}\n Answer:{resp}"
    else:
        return f"Answer:{resp}"


if __name__ == "__main__":
    args = get_args()
    config: dict = get_config_from_args(args)
    registry_paths = config["registry_paths"]
    for path in registry_paths:
        registry_pycls_by_path(path)
    set_seed(config["seed"])
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
    vqa_preprocessor = build_preprocessor(config["vqa_preprocessor_config"])
    classify_preprocessor = build_preprocessor(
        config.get("classify_preprocessor_config", None)
    )

    fn = partial(
        chat_fn, vqa_model, vqa_preprocessor, classify_model, classify_preprocessor
    )

    demo = gr.ChatInterface(
        title="MP-DocVQA Chatbot",
        description="Chatbot for MP-DocVQA",
        fn=fn,
        multimodal=True,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.MultimodalTextbox(
            file_count="multiple",
            file_types=["image"],
            placeholder="Ask Question About Document ...",
            container=False,
            scale=7,
        ),
        type="messages",
        theme="ocean",
    )

    demo.launch(server_name=server_name)
