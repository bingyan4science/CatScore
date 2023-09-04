import argparse
import sys, os
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer

def add_args(parser):
    parser.add_argument(
        "--pretrain",
        default='',
        help="Path to a pretrained model. If not given, we will train from scratch",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./pruned',
        help="The input data dir. Should contain train.source, train.target, val.source, val.target, test.source, test.target",
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    # load model
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain)
    num_params = count_parameters(model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)

    num_layers = len(model.encoder.block)
    assert num_layers == len(model.decoder.block)
    print (f'orig #params: {num_params}')

    #import pdb; pdb.set_trace()
    for to_del in range(1, num_layers):
        model.encoder.block = model.encoder.block[:-1]
        model.decoder.block = model.decoder.block[:-1]
        model.config.num_layers = model.config.num_layers - 1
        model.config.num_decoder_layers = model.config.num_decoder_layers - 1
        num_params = count_parameters(model)
        print (f'after removing {to_del} layers, #params: {num_params}')
        output_dir = os.path.join(args.output_dir, args.pretrain, f'{to_del}_removed')
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    #import pdb; pdb.set_trace()
