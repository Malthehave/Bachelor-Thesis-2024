from __future__ import annotations
import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from flops_profiler.profiler import get_model_profile
from optimized_model import SubNetworkWithProjection # Needed 


def bert_input_constructor(batch_size, seq_len, tokenizer, device):
    fake_seq = ''
    # ignore the two special tokens [CLS] and [SEP]
    for _ in range(seq_len - 2):
        fake_seq += tokenizer.pad_token
    inputs = tokenizer(
        [fake_seq] * batch_size,
        padding=True,
        truncation=True,
        return_tensors='pt',
    ).to(device)
    inputs = dict(inputs)
    return inputs


# name = 'textattack/bert-base-uncased-imdb'
name = "lvwerra/distilbert-imdb"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(name)
config = AutoConfig.from_pretrained(name)
model = AutoModel.from_config(config)
current_script_dir = os.path.dirname(__file__)

def print_output(flops, macs, params):
    print('{:<30}  {:<8}'.format('Number of flops: ', flops))
    print('{:<30}  {:<8}'.format('Number of MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def load_and_report(file):
    model_path = os.path.join(current_script_dir, 'optimized_models', f'{file}.pt')
    model = torch.load(model_path, map_location=device )    
    model = model.to(device)

    batch_size = 1
    seq_len = 128

    flops, macs, params = get_model_profile(
        model,
        kwargs=bert_input_constructor(batch_size, seq_len, tokenizer, device),
        print_profile=False,
        detailed=False,
        as_string=True,
    )
    print("Model: ", file)
    print_output(flops, macs, params)


files = [5, 8, 0, 6, 11, 7, 3, 4, 10, 1, 2, 9]
for file in files:
    load_and_report(file)