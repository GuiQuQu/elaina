import os
import sys
import json

import torch
import deepspeed

import transformers
from transformers import Trainer, TrainingArguments
from transformers import HfArgumentParser
from transformers import set_seed
from transformers import AutoTokenizer


from mpdocvqa_classify.internvl2_for_classify.internvl2.tokenization_internlm2 import InternLM2Tokenizer
from mpdocvqa_classify.internvl2_for_classify.arguments import TotalArguments
from mpdocvqa_classify.internvl2_for_classify.model import InternVL2ClassifyModel
from logger import logger

# init_dist
def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    # dist.init_process_group(backend=backend, **kwargs)
    deepspeed.init_distributed(dist_backend=backend)

def init_dist(backend='nccl', **kwargs):
    _init_dist_pytorch(backend, **kwargs)

############################################

def get_cls_or_func(path_str:str):
    parts = path_str.split('.')
    module_path = '.'.join(parts[:-1])
    cls_name = parts[-1]
    logger.debug(f'Loading submodule {cls_name} from module {module_path}')
    module = __import__(module_path, fromlist=[cls_name])
    cls = getattr(module, cls_name)
    return cls

def get_last_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    return last_checkpoint

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = HfArgumentParser((TotalArguments, TrainingArguments))
    total_args, training_args = parser.parse_args_into_dataclasses()
    config_file = load_config(total_args.config_file)
    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = get_last_checkpoint(training_args)
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load model and dataset
    model_config = config_file['model_config']
    model_type = model_config.pop('type')
    model = get_cls_or_func(model_type)(**model_config)
    dataset_config = config_file['dataset_config']
    dataset_type = dataset_config.pop('type')
    train_dataset = get_cls_or_func(dataset_type)(**dataset_config)
    


if __name__ == "__main__":
    main()