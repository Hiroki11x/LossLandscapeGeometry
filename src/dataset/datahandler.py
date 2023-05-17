import torchtext
import torch
import numpy as np
import random

from collections import Counter
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class DataHandler:

    def __init__(self, datasets):

        self.datasets = datasets

    def load_data(self, preprocess, tokenizer, batch_size, window_size, min_freq, num_workers, shuffle):
        
        train, val, _ = self.datasets
        
        train_map = to_map_style_dataset(train)
        vocab, counter = self.build_vocab(train_map, tokenizer, min_freq)
        pad_idx = vocab["<pad>"]
        wrapper = lambda batch: preprocess(batch, tokenizer, vocab, window_size, pad_idx)

        train_loader = self.dataloader(train, batch_size, wrapper, num_workers, shuffle=shuffle)
        val_loader = self.dataloader(val, batch_size, wrapper, num_workers, shuffle=False)

        return train_loader, val_loader, vocab, counter


    @staticmethod
    def dataloader(dataset, batch_size, collate_fn, num_workers, shuffle):

        fixed_generator = torch.Generator()
        fixed_generator.manual_seed(0)
        
        return DataLoader(to_map_style_dataset(dataset), 
                          batch_size, 
                          shuffle=shuffle, 
                          drop_last=True,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          worker_init_fn=seed_worker,
                          generator=fixed_generator)

    def build_vocab(self, dataset_map, tokenizer, min_freq):

        tokens = map(tokenizer, dataset_map)
        counter = Counter()
        for token in tokens:
            counter.update(token)

        vocab = torchtext.vocab.vocab(counter, min_freq=min_freq, 
                                      specials=["<unk>", "<pad>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab, counter

    