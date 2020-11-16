import argparse
import datetime
from .data_utils import MolTokenizer


def add_args(parser):
    parser.add_argument(
        "data_file",
        nargs='+',
        type=str,
        help="Source file to generate vocabulary.",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=None,
        help="Maximum vocabulary size.",
    )

def main():
    parser = argparse.ArgumentParser(description='Generate vocab file (as torchtext.Vocab) based on given data.')
    add_args(parser)
    args = parser.parse_args()

    tokenizer = MolTokenizer(source_files=args.data_file, mask_token='<mask>',
                             vocab_size=args.max_size)
    tokenizer.save_vocabulary(str(datetime.datetime.now()).split()[0].replace('-','_')
                              +'_vocab.pt')


if __name__ == "__main__":
    main()