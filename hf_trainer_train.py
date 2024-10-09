"""
    旧的训练脚本, 在训练docvqa分类模型时有使用，现在已经整合
    trainer.hf_trainer.py 文件中，可以通过build_hf_trainer来使用
"""
import os
import sys
import json
import argparse

import torch
import deepspeed

import transformers
from transformers import Trainer, TrainingArguments
from transformers import HfArgumentParser
from transformers import set_seed

from utils.utils import get_cls_or_func, load_config

from logger import logger

def get_args():
    parser = argparse.ArgumentParser(description='elania Training')
    parser.add_argument('--config_file', type=str, default=None, help='The config file')
    args = parser.parse_args()
    return args

def get_config_from_args(args):
    config_path = args.config_file
    return load_config(config_path)


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

def load_dataset(dataset_config, split='none'):
    if dataset_config is None:
        logger.warning(f'[{split}] No dataset provided.')
        return None
    _type = dataset_config.pop('type', None)
    if _type is None:
        raise ValueError(f'Dataset type not provided in {dataset_config}.')
    return get_cls_or_func(_type)(**dataset_config)

def load_model(model_config):
    if model_config is None:
        raise ValueError('No model provided.')
    _type = model_config.pop('type', None)
    if _type is None:
        raise ValueError(f'Model type not provided in {model_config}.')
    return get_cls_or_func(_type)(**model_config)

def main():
    args = get_args()
    config = get_config_from_args(args)
    hfparser = HfArgumentParser(TrainingArguments)
    training_args = hfparser.parse_dict(config['training_args'],allow_extra_keys=True)[0]
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

    # load dataset
    train_dataset_config = config.get('train_dataset', None)
    train_dataset = load_dataset(train_dataset_config,split='train')
    eval_dataset_config = config.get('eval_dataset', None)
    eval_dataset = load_dataset(eval_dataset_config,split='eval')
    if train_dataset is None:
        raise ValueError('No training dataset provided.')
    
    # load model
    model_config = config.get('model', None)
    model = load_model(model_config)
    # TODO load model checkpoint if available
    pass

    # print training parameters
    # if DistVarible.is_main_process:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             logger.info(f'{name}: {param.size()}')
    
    from dataset.default_collator import default_collate
    data_collator = default_collate
    if config.get('data_collator', None) is not None:
        data_collator = get_cls_or_func(config['data_collator'])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
    )
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
        trainer._save(output_dir=os.path.join(training_args.output_dir, 'checkpoint-final'))

if __name__ == "__main__":
    main()