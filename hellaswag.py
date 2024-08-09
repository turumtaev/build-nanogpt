"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import numpy as np
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from tokenizer import get_char_to_tokens_for_text

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def get_token_len(x):
    if x == eot:
        return 1
    return len(enc.decode([x]))

def get_pos_and_mask(x, ls):
    B1, B2, T = x.size()

    pos = torch.zeros(x.size(), dtype=torch.long)
    mask = torch.zeros(x.size() + (T, ), dtype=torch.bool)
    # Create a grid of indices for the first and second dimensions
    batch_i_0 = torch.arange(B1).unsqueeze(1).expand(B1, B2)
    batch_i_1 = torch.arange(B2).unsqueeze(0).expand(B1, B2)

    for i in range(T):
        current_i = torch.full((B1, B2), i)
        prev_i = i - ls[:, :, i]
        if any(prev_i.view(-1) >= 0):
            cbi0, cbi1, cpi, cci = \
                  batch_i_0[prev_i >= 0], batch_i_1[prev_i >= 0], prev_i[prev_i >= 0], current_i[prev_i >= 0]
            pos[cbi0, cbi1, i] = pos[cbi0, cbi1, cpi] + 1
            # pos[(prev_i < 0), i] = 0
            mask[cbi0, cbi1, cci] = mask[cbi0, cbi1, cpi]
        mask[batch_i_0, batch_i_1, current_i, current_i] = True
    return pos, mask

def render_example(example, B, T):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size B//4,4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctxs = [e['ctx'] for e in example] # (B//4, )
    all_endings = [e['endings'] for e in example] # (B//4, 4)
    labels = [e['label'] for e in example] # (B//4, )
    ctx_tokens = [enc.encode(ctx) for ctx in ctxs] # (B//4, N_i)
    ctx_tokens_lens = [[get_token_len(x) for x in ctx] for ctx in ctx_tokens]

    tok_rows_batch = []
    tok_len_rows_batch = []
    target_rows_batch = []
    for bi, endings in enumerate(all_endings):
        tok_rows = []
        tok_len_rows = []
        target_rows = []
        for end in endings:
            char2tokens = get_char_to_tokens_for_text(" " + end)
            char2tokens = char2tokens
            end_tokens = [c for c, ts in char2tokens[1:]] # (N_o)
            end_token_lens = [get_token_len(x) for x in end_tokens] # (N_o)
            targets = [[]] * (len(ctx_tokens[bi]) - 1) + [ts for c, ts in char2tokens[:-1]] + [[]] * T # (T, N_e)
            targets = targets[:T]
            tok_rows.append(ctx_tokens[bi] + end_tokens) # (4, N_i+N_o)
            tok_len_rows.append(ctx_tokens_lens[bi] + end_token_lens) # (4, N_i+N_o)

            target_rows.append(targets) # (4, N_o, N_e)
        tok_rows_batch.append(tok_rows) # (B // 4, 4, N_i+N_o)
        tok_len_rows_batch.append(tok_len_rows) # (B // 4, 4, N_i+N_o)
        target_rows_batch.append(target_rows)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = T # min(T, max(len(row) for tok_rows in tok_rows_batch for row in tok_rows))
    # TBC
    tokens = torch.zeros((B//4, 4, max_len), dtype=torch.long)
    token_lens = torch.zeros((B//4, 4, max_len), dtype=torch.long)
    for i, (tok_rows, tok_len_rows) in enumerate(zip(tok_rows_batch, tok_len_rows_batch)):
        for j, (row, len_row) in enumerate(zip(tok_rows, tok_len_rows)):
            tokens[i, j, :len(row[:max_len])] = torch.tensor(row[:max_len])
            token_lens[i, j, :len(len_row[:max_len])] = torch.tensor(len_row[:max_len])
    pos, mask = get_pos_and_mask(tokens, token_lens)

    target_lens = np.array([[[len(t) for t in targets] for targets in tr] for tr in target_rows_batch ])  # (B//4, 4, T])
    target_lens = np.transpose(target_lens, (2, 0, 1)).reshape(-1)
    target_clens = np.cumsum(target_lens)
    target_indices = np.insert(target_clens, 0, 0)
    target_poses = np.arange(len(target_lens))[target_lens>0]
    target_lens = target_lens[target_lens>0]
    target_poses = np.repeat(target_poses, target_lens)
    target_lens = np.repeat(target_lens, target_lens)
    assert max(target_indices) <= len(target_lens), f'{max(target_indices), len(target_lens)}'

    new_targets = []
    for t in range(T):
        for b1 in range(B//4):
            for b2 in range(4):
                new_targets += target_rows_batch[b1][b2][t]
    ctx_len = torch.tensor([min(max_len, len(ct)) for ct in ctx_tokens], dtype=torch.long) # (B//4,)
    tok_len = torch.tensor([[min(max_len, len(tok)) for tok in tr] for tr in tok_rows_batch ], dtype=torch.long) # (B//4, 4)
    poses  = torch.tensor(target_poses, dtype=torch.long)
    indices = torch.tensor(target_indices, dtype=torch.long)
    lens = torch.tensor(target_lens, dtype=torch.long)
    targets = torch.tensor(new_targets, dtype=torch.long)

    y = ctx_len, tok_len, indices, poses, targets, lens# (B//4,), (B//4, 4), (B//4,4, T), (N_t,), (N_t,), (N_t,)
    assert len(poses) == len(targets) == len(lens), f'{len(poses), len(targets), len(lens)}'

    return tokens, pos, mask, y, labels

def iterate_examples(split, B):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        examples = []
        for line in f:
            example = json.loads(line)
            examples.append(example)
            if len(examples) >= B // 4:
                examples_to_yield, examples = list(examples), []
                yield examples_to_yield
        if len(examples) >= B // 4:
            yield examples

def get_most_likely_row(logits, y, T, B):
    B1, B2 = B // 4, 4
    ctx_len, tok_len, indices, poses, targets, lens = y # (B//4,), (B//4, 4), (B//4,4, T), (N_t,), (N_t,), (N_t,)
    logits = logits.permute(2, 0, 1, 3)
    logits = logits.reshape(-1, logits.size(-1))  # (T*B//4*4, C)
    props = torch.zeros((T, B1, B2), dtype=torch.float32, device=logits.device)
    # print(props.shape, ctx_len.shape, tok_len.shape, indices.shape, poses.shape, targets.shape, lens.shape)
    props[ctx_len - 1, torch.arange(B1, device=logits.device), :] = 1.0
    props = props.view(-1)
    for t in range(T):
        start_i, end_i = indices[t * B1*B2], indices[(t+1)*B1*B2]
        cur_pos = poses[start_i:end_i]
        next_pos = torch.clamp(cur_pos + lens[start_i:end_i] * B1 * B2, max=(T-1)*B1*B2 + cur_pos % (B1*B2))
        cur_props = props[cur_pos]
        cur_targets = targets[start_i:end_i]
        cur_logits = F.softmax(logits[cur_pos, cur_targets], dim=-1)
        props[next_pos] = props[next_pos] + cur_props * cur_logits
    batch_i_0 = torch.arange(B1, device=logits.device).unsqueeze(1).expand(B1, B2)
    batch_i_1 = torch.arange(B2, device=logits.device).unsqueeze(0).expand(B1, B2)
    props = props.view((T, B1, B2))
    props_end = props[tok_len-1, batch_i_0, batch_i_1] # (B//4, 4)
    pred_norm = props_end.argmax(dim=1).tolist() # (B//4)
    return pred_norm