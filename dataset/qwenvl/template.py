from typing import List, Tuple
from dataclasses import dataclass, field

# 传入指定格式的输入，返回添加完标记的prompt


@dataclass
class QwenVLTemplate(object):
    name: str = "qwenvl"
    system_message: str = "You are a helpful assistant."
    stop_words: List[str] = field(default_factory=list)
    roles: Tuple[str] = ("user", "assistant")
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"
    img_start = "<img>"
    img_end = "</img>"
    offset: int = 0
    sep: str = "\n"
    # add_generation_prompt: bool = True
    messages: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def assistant_start(self):
        return f"{self.im_start}{self.roles[1]}{self.sep}"

    @property
    def user_start(self):
        return f"{self.im_start}{self.roles[0]}{self.sep}"

    def set_system_message(self, system_message):
        self.system_message = system_message

    def add_message(self, role, message):
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

    def system_message_with_role(self, system_message=None) -> str:
        if system_message is None:
            system_message = self.system_message
        return f"system{self.sep}{system_message}"

    def parse_message(self, message):
        "按照openai 的 api格式进行msg的解析,解析成为string"
        if isinstance(message, str):
            return message
        elif isinstance(message, list):
            return self.parse_by_openai(message)
        else:
            raise ValueError("message should be a list or a string")

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
            "messages": self.messages,
        }

    def parse_by_openai(self, message):
        ret = ""
        if isinstance(message, list):
            picture_idx = 1
            for msg in message:
                assert isinstance(msg, dict)
                msg_type = msg["type"]
                if msg_type == "image" or msg_type == "image_url":
                    content = msg[msg_type]
                    ret += f"Picture {picture_idx}: {self.img_start}{content}{self.img_end}{self.sep}"
                    picture_idx += 1
                if msg_type == "text":
                    ret += msg["text"]
        else:
            raise ValueError(
                "message should be a list, and each element should be a dict"
            )
        return ret


if __name__ == "__main__":
    tem = QwenVLTemplate()
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
            "content": "Image 1:<img>file:///path/to/image1.jpg</img>\nImage 2:<img>file:///path/to/image2.jpg</img>\nDescribe the two images.",
        },
        {
            "role": "assistant",
            "content": "The two images are of the same type of fruit.",
        },
        {
            "role": "user",
            "content": "What is the fruit?",
        },
        {
            "role": "assistant",
            "content": "The fruit is an apple.",
        }
    ]
    for msg in messages:
        tem.add_message(msg["role"], msg["content"])

    prompt = tem.get_prompt(add_generation_prompt=False)
    print(prompt)
