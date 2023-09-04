import copy
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
        default='./sliced',
        help="The input data dir. Should contain train.source, train.target, val.source, val.target, test.source, test.target",
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch
from transformers import AutoModel, AutoConfig

def reduce_model_size(model, factor):
    new_config = copy.deepcopy(model.config)
    new_config.update({'d_model': new_config.d_model // factor})
    new_config.update({'d_ff': new_config.d_ff // factor})
    new_config.update({'d_kv': new_config.d_kv // factor})
    new_model = AutoModel.from_config(new_config)
    
    # Iterate over original and new model parameters
    for (name, param), (_, new_param) in zip(model.named_parameters(), new_model.named_parameters()):
        
        # Check if current parameter's dimensions can be reduced
        sliced_param = param
        if len(param.size()) >= 2 and param.size(-1) == model.config.hidden_size:
            sliced_param = param.data[..., :new_config.hidden_size]
            #new_param.data = sliced_param
        else:
            pass
            #new_param.data = param.data
        if new_param.shape != sliced_param.shape:
            #if sliced_param.size(0) != model.config.hidden_size:
            #    import pdb; pdb.set_trace()
            sliced_param = sliced_param.data[:new_param.shape[0]]
        if new_param.shape != sliced_param.shape:
            sliced_param = sliced_param.data[..., :new_param.shape[-1]]
        if new_param.shape != sliced_param.shape:
            #import pdb; pdb.set_trace()
            pass
        #new_param.data = sliced_param
        #import pdb; pdb.set_trace()

    return new_model


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
    for factor in [2, 4, 8, 16, 32, 64]:
        smaller_model = reduce_model_size(model, factor)
        num_params = count_parameters(smaller_model)
        print (f'after dividing hidden size by {factor}, #params: {num_params}')
        output_dir = os.path.join(args.output_dir, os.path.basename(args.pretrain), f'shrink_by_{factor}')
        os.makedirs(output_dir, exist_ok=True)
        smaller_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    #import pdb; pdb.set_trace()
