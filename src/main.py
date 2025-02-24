import argparse
from scripts.train import main as train_main
from scripts.evaluate import main as eval_main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    args = parser.parse_args()
    if args.mode == 'train':
        train_main()
    else:
        eval_main()

if __name__ == '__main__':
    main()
