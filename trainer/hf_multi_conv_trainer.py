"""
    支持llm多轮对话的训练, 可以在训练过程中让多轮对话只跑一次forward
    而不需要拆解为多个单轮对话
"""

from typing import Callable, List, Optional, Tuple, Union, Dict

from transformers import PreTrainedModel, TrainingArguments, PreTrainedTokenizerBase
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback

import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset

from utils.register import Register


# def compute_1d_non_zero_block_sum(input: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
#     # 按照求解前缀和的思路,求完前缀和之后求解所有的区间左右端点,
#     # 然后按照s(r) - s(l-1)的方式求解区间和,通过遍历解出所有区间和
#     device = input.device
#     assert input.dim() == 1
#     prefix_sum = torch.cumsum(input, dim=-1)

#     # 求解开始坐标(当前元素非0,且前一个元素为0)
#     non_zero_mask = input != 0
#     temp_mask = torch.cat([torch.zeros(1, dtype=torch.bool, device=device), non_zero_mask], dim=0)
#     temp_mask = temp_mask[:-1]
#     block_start_mask = non_zero_mask & ~temp_mask
#     start_idx = torch.nonzero(block_start_mask, as_tuple=False).squeeze(1)
#     # 求解结束坐标(当前元素非0,且下一个元素为0)
#     temp_mask = torch.cat([non_zero_mask, torch.zeros(1, dtype=torch.bool, device=device)], dim=0)
#     temp_mask = temp_mask[1:]
#     block_end_mask = non_zero_mask & ~temp_mask
#     end_idx = torch.nonzero(block_end_mask, as_tuple=False).squeeze(1)
#     block_sum = []
#     for st, ed in zip(start_idx, end_idx):
#         s = prefix_sum[ed] - prefix_sum[st-1] if st > 0 else prefix_sum[ed]
#         block_sum.append(s)
#     return torch.tensor(block_sum, device=device, requires_grad=requires_grad)

def get_loss_token_num_turn_num(losses) -> Tuple[torch.Tensor,torch.Tensor]:
    assert losses.dim() == 1
    s = 0
    loss_token_num = []
    cur_token_num = 0
    turn_num = 0
    for i, token in enumerate(losses):
        if token == 0: 
            loss_token_num.append(1)
            cur_token_num = 0
        else:
            if cur_token_num == 0:
                breakpoint()
                 # 计算当前区间的token数量
                for j, n in enumerate(losses[i:]):
                    if n == 0:
                        break
                    cur_token_num += 1
                turn_num += 1
            loss_token_num.append(cur_token_num)
    loss_token_num = torch.tensor(loss_token_num, device=losses.device, dtype=torch.long)
    turn_num = torch.tensor(turn_num, device=losses.device, dtype=torch.long)
    return loss_token_num, turn_num

# 参考：https://zhuanlan.zhihu.com/p/721652210
def loss_func(losses, loss_mask, loss_token_num, turn_num):
    """
        Args:
         turn_num,这个batch内每一条数据的对话轮数
    """
    losses = losses.view(-1)
    loss_mask = loss_mask.view(-1)
    loss_token_num = loss_token_num.view(-1)
    turn_num = turn_num.sum()
    loss = torch.sum(losses * loss_mask / loss_token_num)
    # 对对话轮数求平均
    return loss / turn_num

@Register(name="hf_multi_conv_trainer")
class HFMultiConvTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        
        if 'labels' in inputs:
            labels = inputs.pop("labels")
        _, outputs = model(**inputs)
        if isinstance(outputs, dict) and 'logits' not in outputs:
            raise ValueError(
                "The model did not return a logits from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        
        # [bsz, seq_len, vocal_size]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
        loss = None
        assert labels is not None
        # Shift so that tokens < n predict n
        bsz, seq_len, vocab_size = logits.size()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        losses = loss_fct(shift_logits, shift_labels)
        loss_mask = losses != 0
        loss_token_num = loss_mask.sum()
        loss = torch.sum(losses * loss_mask) / loss_token_num
        # 按照bsz切分
        losses = losses.view(bsz, seq_len - 1)
        loss_mask = losses != 0
        loss_token_num, turn_num = [], []
        for l in losses:
            t1,t2 = get_loss_token_num_turn_num(l)
            loss_token_num.append(t1)
            turn_num.append(t2)
        loss_token_num = torch.stack(loss_token_num)
        turn_num = torch.stack(turn_num)
        loss =loss_func(losses, loss_mask, loss_token_num, turn_num)
        return (loss, outputs) if return_outputs else loss