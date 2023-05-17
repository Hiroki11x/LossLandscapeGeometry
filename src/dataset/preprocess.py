import torch

from torch.nn.utils.rnn import pad_sequence

def preprocess(batch, tokenizer, vocab, window_size, pad_idx):

    idxs = []

    for text in batch:
        token_ids = vocab(tokenizer(text))
        length = len(token_ids)

        if length < 2:
            continue

        elif length > window_size + 1:
            token_ids = token_ids[:window_size+1]

        idxs.append(torch.tensor(token_ids, dtype=torch.int64))

    if len(idxs) == 0:
        idxs = [torch.tensor([pad_idx], dtype=torch.int64)]
        
    idxs_tensor = pad_sequence(idxs, batch_first=True, padding_value=pad_idx)
    return idxs_tensor