from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass, field
from typing_extensions import override

from PIL import Image


def handle_image(image):
    if image.startswith("file://"):
        image_path = image[7:]
        return Image.open(image_path).convert("RGB")
    else:
        raise ValueError("image should be a file path")


# 传入指定格式的输入，返回添加完标记的prompt
@dataclass
class Template(ABC):
    name: str
    system_message: str
    stop_words: List[str] = field(default_factory=list)
    roles: Tuple[str] = ("user", "assistant")
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"
    offset: int = 0
    sep: str = "\n"
    # add_generation_prompt: bool = True
    messages: List[Tuple[str, str]] = field(default_factory=list)

    def set_system_message(self, system_message):
        self.system_message = system_message

    def add_message(self, role: str, message: str):
        assert role in self.roles
        self.messages.append((role, message))

    def update_last_message(self, message):
        self.messages[-1] = message

    def get_prompt(self, add_generation_prompt: bool = False) -> str:
        system_prompt = self.system_message
        # system_prompt
        ret = self.format_system(system_prompt) + self.sep
        for i, (role, message) in enumerate(self.messages):
            if i % 2 == 0:
                ret += self.format_user(message) + self.sep
            else:
                ret += self.format_assistant(message) + self.sep
        if add_generation_prompt:
            ret += f"{self.im_start}{self.roles[1]}{self.sep}"
        return ret

    def clear_messages(self):
        self.messages = []

    def format_user(self, message):
        return f"{self.im_start}{self.roles[0]}{self.sep}{self.parse_message(message)}{self.im_end}"

    def format_assistant(self, message):
        return f"{self.im_start}{self.roles[1]}{self.sep}{self.parse_message(message)}{self.im_end}"

    def format_system(self, system_message=None):
        if system_message is None:
            system_message = self.system_message
        return f"{self.im_start}system{self.sep}{system_message}{self.im_end}"

    def parse_message(self, message):
        return message

    def to_openai_api_messages(self):
        ret = [{"role": "system", "content": self.system_message}]
        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                ret.append({"role": "assistant", "content": msg})
        return ret

    def dict(self):
        return {
            "name": self.name,
            "system_message": self.system_message,
            "stop_words": self.stop_words,
            "roles": self.roles,
            "im_start": self.im_start,
            "im_end": self.im_end,
            "offset": self.offset,
            "sep": self.sep,
            "add_generation_prompt": self.add_generation_prompt,
            "messages": self.messages,
        }


@dataclass
class Qwen2VLTemplate(Template):
    name: str = "qwen2-vl"
    system_message: str = "You are a helpful assistant."
    stop_words: List[str] = field(default_factory=lambda: ["<|im_end|>"])
    roles: Tuple[str] = ("user", "assistant")
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"
    sep: str = "\n"
    image_placeholder = "<|image|>"
    video_placeholder = "<|video|>"
    image_pad = "<|image_pad|>"
    video_pad = "<|video_pad|>"
    mm_start = "<|vision_start|>"
    mm_end = "<|vision_end|>"

    @property
    def assistant_start(self):
        return f"{self.im_start}{self.roles[1]}{self.sep}"

    @property
    def user_start(self):
        return f"{self.im_start}{self.roles[0]}{self.sep}"

    @override
    def parse_message(self, message):
        "按照openai 的 api格式进行msg的解析,解析成为string"
        if isinstance(message, str):
            return self.parse_msg_by_str(message)
        elif isinstance(message, list):
            if self.check_contain_image_placeholder(message):
                return self.parse_message_by_openai_with_imageplaceholder(message)
            else:
                self.parse_by_openai(message)
        else:
            raise ValueError("message should be a list or a string")

    def parse_msg_by_str(self, message):
        assert isinstance(message, str)

        parts = message.split(self.image_placeholder)
        ret = ""
        for i, part in enumerate(parts):
            ret += part
            if i < len(parts) - 1:
                ret += f"{self.mm_start}{self.image_pad}{self.mm_end}"

        parts = ret.split(self.video_placeholder)
        ret = ""
        for i, part in enumerate(parts):
            ret += part
            if i < len(parts) - 1:
                ret += f"{self.mm_start}{self.video_pad}{self.mm_end}"
        return ret

    def parse_message_by_openai_with_imageplaceholder(self, message):
        ret_text, images = "", []
        need_image_cnt = 0
        if not isinstance(message, list):
            raise ValueError("message should be a list")
        if len(message) == 0:
            return ret_text, images
        if not isinstance(message[0], dict):
            raise ValueError("each element in message should be a dict")

        for msg in message:
            assert isinstance(msg, dict)
            msg_type = msg["type"]
            if msg_type == "image" or msg_type == "image_url":
                images.append(handle_image(msg["image"]))
            if msg_type == "text":
                need_image_cnt += msg["text"].count(self.image_placeholder)
                t = self.parse_msg_by_str(msg["text"])
                ret_text += t
        if len(images) != need_image_cnt:
            raise ValueError(
                f"image count should be equal to the number of image placeholders in text, but get {len(images)} images and {need_image_cnt} placeholders"
            )
        return ret_text, images

    def check_contain_image_placeholder(self, message):
        texts = [msg["text"] for msg in message if msg["type"] == "text"]
        return any([self.image_placeholder in text for text in texts])

    def parse_by_openai(self, message):
        """
        按照固定的格式解析message,文本中不应该包含图像占位符
        """

        ret = ""
        if isinstance(message, list):
            picture_idx = 1
            video_idx = 1
            # 寻找msg中的文本输入
            texts = [msg["text"] for msg in message if msg["type"] == "text"]
            if any([self.image_placeholder in text for text in texts]):
                raise ValueError(
                    "text should not contain image placeholder, if you want organize the format input, please use 'parse_message_by_openai_with_imageplaceholder' method"
                )
            for msg in message:
                assert isinstance(msg, dict)
                msg_type = msg["type"]
                if msg_type == "image" or msg_type == "image_url":
                    ret += f"Picture {picture_idx}: {self.mm_start}{self.image_pad}{self.mm_end}"
                    picture_idx += 1
                elif msg_type == "video":
                    ret += f"Video {video_idx}: {self.mm_start}{self.video_pad}{self.mm_end}"
                    video_idx += 1
                if msg_type == "text":
                    ret += self.parse_msg_by_str(msg["text"])
        else:
            raise ValueError(
                "message should be a list, and each element should be a dict"
            )
        return ret


if __name__ == "__main__":
    tem = Qwen2VLTemplate(add_generation_prompt=False)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "file:///path/to/image1.jpg"},
                {"type": "image", "image": "file:///path/to/image2.jpg"},
                {
                    "type": "text",
                    "text": "Identify the similarities between these images.",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The images are both of the same type of fruit.",
                }
            ],
        },
    ]

    messages = [
        {
            "role": "user",
            "content": "Image 1:<|image|>\nImage 2:<|image|>\nDescribe the two images.",
        },
        {
            "role": "assistant",
            "content": "The two images are of the same type of fruit.",
        },
    ]
    for msg in messages:
        tem.add_message(msg["role"], msg["content"])

    prompt = tem.get_prompt()
    print(prompt)
    # <|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # Image 1:<|image|>
    # Image 2:<|image|>
    # Describe the two images.<|im_end|>
    # <|im_start|>assistant
    # The two images are of the same type of fruit.<|im_end|>
