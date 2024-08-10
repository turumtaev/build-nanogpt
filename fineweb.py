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

def transform_y(y):
    npy = np.array(y, dtype=object)
    # get array of lengths
    npl = np.vectorize(len)(npy)
    csl = np.cumsum(npl)
    # get array of indices
    npi = np.insert(csl[:-1], 0, 0)
    nprows = np.repeat(np.arange(len(npl)), npl)
    npy = np.concatenate(npy)
    npl_repeated = np.repeat(npl, npl)
    return nprows, npy, npl_repeated, npi

def get_token_len(x):
    if x == eot:
        return 1
    return len(enc.decode_bytes([x]))

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    char2tokens = get_char_to_tokens_for_text(doc['text'])
    char2tokens[0][0] = eot
    char2tokens[-1][1] = [eot]
    x, y = zip(*char2tokens)
    x, y = list(x), list(y)
    x_np = np.array(x)
    x_len = np.vectorize(get_token_len)(x_np)
    assert (0 <= x_np).all() and (x_np < 2**16).all(), "token dictionary too large for uint16"
    return x_np.astype(np.uint16), x_len.astype(np.uint16), transform_y(y)

def write_datafile(filename, data):
    try:
        np.save(filename, data)
    except Exception as e:
        print(f"Failed to save {filename}: {e}")

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
        all_np = {
            "x": np.empty((shard_size,), dtype=np.uint16),
            "xlen": np.empty((shard_size,), dtype=np.uint16),
            "y": np.empty((shard_size * 10,), dtype=np.uint16),
            "rows": np.empty((shard_size * 10,), dtype=np.int64),
            "len": np.empty((shard_size * 10,), dtype=np.int64),
            "indices": np.empty((shard_size,), dtype=np.int64),
        }
        token_count = 0
        y_count = 0
        progress_bar = None
        for x, x_len, y in pool.imap(tokenize, fw, chunksize=4):
            if shard_index >= 100:
                break
            # is there enough space in the current shard for the new tokens?
            if token_count + len(x) < shard_size:
                # simply append tokens to current shard
                all_np["x"][token_count:token_count+len(x)] = x
                all_np["xlen"][token_count:token_count+len(x)] = x_len

                nprows, npy, npl, npi = y
                npi = npi + y_count
                nprows = nprows + token_count
                all_np["y"][y_count:y_count+len(npy)] = npy
                all_np["rows"][y_count:y_count+len(npy)] = nprows
                all_np["len"][y_count:y_count+len(npy)] = npl
                all_np["indices"][token_count:token_count+len(x)] = npi

                token_count += len(x)
                y_count += len(npy)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(x))
            else:
                # write the current shard and start a new one
                # 10 times smaller shards -> 10 val shards
                split = "val" if shard_index < 1 else "train"
                filenames = {k: os.path.join(DATA_CACHE_DIR, f"edufineweb_{k}_{split}_{shard_index:06d}") for k in all_np}
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_np["x"][token_count:token_count+remainder] = x[:remainder]
                all_np["xlen"][token_count:token_count+remainder] = x_len[:remainder]

                nprows, npy, npl, npi = y
                if remainder != len(x):
                    npy_remainder = len(npy)
                else:
                    npy_remainder = npi[remainder]
                    
                npi = npi + y_count
                nprows = nprows + token_count
                all_np["y"][y_count:y_count+npy_remainder] = npy[:npy_remainder]
                all_np["rows"][y_count:y_count+npy_remainder] = nprows[:npy_remainder]
                all_np["len"][y_count:y_count+npy_remainder] = npl[:npy_remainder]
                all_np["indices"][token_count:token_count+remainder] = npi[:remainder]
                k2counts = {
                    "x": token_count,
                    "xlen": token_count,
                    "y": y_count,
                    "rows": y_count,
                    "len": y_count,
                    "indices": token_count
                }
                cur_processes = [save_in_background(filenames[k], all_np[k][:k2counts[k]]) for k in all_np]
                processes.extend(cur_processes)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_np["x"][0:len(x)-remainder] = x[remainder:]
                all_np["xlen"][0:len(x)-remainder] = x_len[remainder:]
                all_np["y"][0:len(npy)-npy_remainder] = npy[npy_remainder:]
                all_np["rows"][0:len(npy)-npy_remainder] = nprows[npy_remainder:]
                all_np["len"][0:len(npy)-npy_remainder] = npl[npy_remainder:]
                all_np["indices"][0:len(x)-remainder] = npi[remainder:]
                token_count = len(x)-remainder
                y_count = len(npy)-npy_remainder

        # write any remaining tokens as the last shard
        if token_count != 0 and shard_index < 100:
            split = "val" if shard_index < 1 else "train"
            filenames = {k: os.path.join(DATA_CACHE_DIR, f"edufineweb_{k}_{split}_{shard_index:06d}") for k in all_np}
            k2counts = {
                "x": token_count,
                "xlen": token_count,
                "y": y_count,
                "rows": y_count,
                "len": y_count,
                "indices": token_count
            }
            cur_processes = [save_in_background(filenames[k], all_np[k][:k2counts[k]]) for k in all_np]
            processes.extend(cur_processes)
        for process in processes:
            process.join()

if __name__ == '__main__':
    main()