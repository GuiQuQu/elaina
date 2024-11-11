import os

import transformers
from transformers import Trainer,HfArgumentParser,TrainingArguments,set_seed

from trainer.base_trainer import BaseTrainer
from utils.utils import get_cls_or_func
from trainer.trainer_utils import load_dataset,load_model
from dataset.build import build_dataset
from logger import logger

def hf_get_last_checkpoint(training_args):
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

def build_training_args_dict(config):
    training_args_dict = config.get('training_args', {})
    training_args_dict['output_dir'] = config.get('output_dir', None)
    training_args_dict['seed'] = config.get('seed', 42)
    return training_args_dict

def build_hf_trainer(config):
    hfparser = HfArgumentParser(TrainingArguments)
    training_args = hfparser.parse_dict(build_training_args_dict(config),allow_extra_keys=True)[0]
    # Log on each process the small summary:
    logger.info(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu},'
        + f'distributed training: {bool(training_args.local_rank != -1)}, fp16 training: {training_args.fp16}, bf16 training: {training_args.bf16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')
    set_seed(training_args.seed)

    # load dataset
    train_dataset_config = config.get('train_dataset', None)
    train_dataset = build_dataset(train_dataset_config,split='train')
    eval_dataset_config = config.get('val_dataset', None)
    eval_dataset = build_dataset(eval_dataset_config,split='val')
    if train_dataset is None:
        raise ValueError('No training dataset provided.')

    # load model
    model_config = config.get('model', None)
    model = load_model(model_config)

    from dataset.default_collator import default_collate
    data_collator = default_collate
    if config.get('data_collator', None) is not None:
        data_collator = get_cls_or_func(config['data_collator'])

    return HFTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
    )

from utils.register import Register

@Register(name='hf_trainer')
class HFTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        training_args,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        **kwargs
    ):
        self.train_dataset = train_dataset
        self.training_args = training_args
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            data_collator=data_collator,
            **kwargs
        )

    def train(self):
        hf_last_checkpoint = hf_get_last_checkpoint(self.training_args)
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif hf_last_checkpoint is not None:
                checkpoint = hf_last_checkpoint
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(self.train_dataset)
        except:
            metrics['train_samples'] = -1
        self.trainer.log_metrics('train', metrics)
        self.trainer.save_metrics('train', metrics)
        self.trainer.save_state()
        self.trainer._save(output_dir=os.path.join(self.training_args.output_dir, f'checkpoint-{self.trainer.state.global_step}'))

