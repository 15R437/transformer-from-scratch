import tiktoken
import numpy as np
import torch
from typing import List, Literal

SEED = 15
rng = np.random.default_rng(SEED)

def get_batch(data_set : str, batch_size : int, n_ctx : int, split : Literal['train', 'val']) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        data = np.memmap(f"data/{data_set}/{split}.bin", dtype=np.uint16, mode='r')
    except FileNotFoundError:
        try:
            enc = tiktoken.get_encoding("gpt2")
            text = open(f"data/{data_set}/input.txt", encoding="utf-8").read()
            ids = enc.encode_ordinary(text)
            arr = np.array(ids, dtype=np.uint16)
            train_idx = int(0.9 * len(arr))
            arr[:train_idx].tofile(f"data/{data_set}/train.bin")
            arr[train_idx:].tofile(f"data/{data_set}/val.bin")
        except FileNotFoundError:
            raise FileNotFoundError("Data file not found")

    n = min(n_ctx, len(data) - 1)


    offset = rng.choice(len(data) - n, size=(batch_size,), replace=False)
    window = data[offset[:, None] + np.arange(n + 1)]

    input = torch.from_numpy(window[:, :-1].astype(np.int64))
    target = torch.from_numpy(window[:, 1:].astype(np.int64))

    return input, target

if __name__ == "__main__":
    input, target = get_batch("tiny_shakespeare", 4, 8, "train")
    print(f"Input: {input[:10]} \nTarget: {target[:10]}")