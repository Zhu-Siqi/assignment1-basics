from cs336_basics.model import CustomTransformerLM
from cs336_basics.config import config
from cs336_basics.decoding import low_temperature_sampling_step
from cs336_basics.Tokenizer.tokenizer import Tokenizer
import torch
import json
import base64
MAX_LEN = 4000

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = CustomTransformerLM(**config['model'])
    check_point = torch.load('check_point/checkpoint_owt.pt')
    print(f'train iterations: {check_point['iteration']}')
    
    model.load_state_dict(check_point['model_state_dict'])
    with open('data/owt-vocab.json', 'r', encoding = 'utf-8') as f:
        vocab = json.load(f)
    vocab = {int(idx): base64.b64decode(encode_b) for idx, encode_b in vocab.items()}

    merges = []
    with open('data/owt-merges.txt', 'r', encoding = 'utf-8') as f:
        for row in f:
            if not row:
                continue
            parts = row.split()
            if len(parts) == 2:
                first = base64.b64decode(parts[0])
                second = base64.b64decode(parts[1])
                merges.append((first, second))

    tkn = Tokenizer(
        vocab, merges, ['<|endoftext|>']
    )
    prompt = tkn.encode('The key did\'t open any door in Elara\'s house, but she kept it anyway, hanging on a silver chain around her neck. It was cold against her skin, a constant, mysterious weight.')

    model = model.to(device)
    prompt = torch.tensor(prompt).long().to(device)

    end_token = b'<|endoftext|>'
    for idx, token in tkn.vocab.items():
        if token == end_token:
            end_idx = idx
            break
    
    while prompt[-1] != end_idx and len(prompt) < MAX_LEN:
        prompt = low_temperature_sampling_step(model, prompt, temperature=1)
    
    prompt = prompt.cpu().numpy().tolist()
    
    print(tkn.decode(prompt))