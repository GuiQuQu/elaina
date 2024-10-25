
from transformers import set_seed

from arguments import get_args,get_config_from_args
from trainer.build import build_trainer
from tester.build import build_tester

from utils.utils import check_environment

def main():
    check_environment()
    args = get_args()
    config = get_config_from_args(args)
    set_seed(config['seed'])

    if args.do_train:
        trainer = build_trainer(config)
        trainer.train()
    if args.do_test:
        tester = build_tester(config)
        tester.test()
    
if __name__ == "__main__":
    main()