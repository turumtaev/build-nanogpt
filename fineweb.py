"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
from tokenizer import get_char_to_tokens_for_text

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
nprocs = max(1, os.cpu_count()//2)
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", num_proc=nprocs)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    char2tokens = get_char_to_tokens_for_text(doc['text'])
    char2tokens[0][0] = eot
    char2tokens[-1][1] = [eot]
    x, y = zip(*char2tokens)
    x, y = list(x), list(y)
    x_np = np.array(x)
    assert (0 <= x_np).all() and (x_np < 2**16).all(), "token dictionary too large for uint16"
    x_np_uint16 = x_np.astype(np.uint16)
    y_np = np.array(y, dtype=object)
    return x_np_uint16, y_np

def write_datafile(filename, data):
    np.save(filename, data)

def save_in_background(filename, data):
    process = mp.Process(target=write_datafile, args=(filename, data))
    process.start()
    return process  # Return the process in case you want to join it later

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
def main():
    nprocs = max(1, os.cpu_count()//2)
    processes = []
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_x_np = np.empty((shard_size,), dtype=np.uint16)
        all_y_np = np.empty((shard_size,), dtype=object)
        token_count = 0
        progress_bar = None
        for x, y in pool.imap(tokenize, fw, chunksize=4):
            if shard_index >= 100:
                break
            # is there enough space in the current shard for the new tokens?
            if token_count + len(x) < shard_size:
                # simply append tokens to current shard
                all_x_np[token_count:token_count+len(x)] = x
                all_y_np[token_count:token_count+len(x)] = y
                token_count += len(x)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(x))
            else:
                # write the current shard and start a new one
                # 10 times smaller shards -> 10 val shards
                split = "val" if shard_index < 1 else "train"
                x_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_x_{split}_{shard_index:06d}")
                y_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_y_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_x_np[token_count:token_count+remainder] = x[:remainder]
                all_y_np[token_count:token_count+remainder] = y[:remainder]
                process_x = save_in_background(x_filename, all_x_np)
                process_y = save_in_background(y_filename, all_y_np)
                processes.extend([process_x, process_y])
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_x_np[0:len(x)-remainder] = x[remainder:]
                all_y_np[0:len(x)-remainder] = y[remainder:]
                token_count = len(x)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0 and shard_index < 100:
            split = "val" if shard_index < 1 else "train"
            x_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_x_{split}_{shard_index:06d}")
            y_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_y_{split}_{shard_index:06d}")
            process_x = save_in_background(x_filename, all_x_np[:token_count])
            process_y = save_in_background(y_filename, all_y_np[:token_count])
            processes.extend([process_x, process_y])
        for process in processes:
            process.join()

if __name__ == '__main__':
    main()