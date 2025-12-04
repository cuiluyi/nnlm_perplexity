import re
from collections import Counter
from tqdm import tqdm
from pathlib import Path

import jieba
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import read_file, clean_text
from constants import MIN_FREQ

corpus_files = ["data/train_corpus.txt", "data/test_corpus.txt"]

word_counts = Counter()
for file in corpus_files:
    text = read_file(file)
    words = [w for w in jieba.lcut(text) if re.search(r"[\u4e00-\u9fff]", w)]
    # Build vocabulary (filter words with frequency < min_freq)
    word_counts.update(words)

vocab = [word for word, count in word_counts.most_common() if count >= MIN_FREQ]
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}


def get_loader(
    data_path: str | Path,
    context_size: int,
    batch_size: int,
):
    # tokenize data into contexts and targets
    data_contexts = []
    data_targets = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            words = [w for w in jieba.lcut(line) if re.search(r"[\u4e00-\u9fff]", w)]
            # Option1: non-overlapping to create (context, target) pairs
            for i in tqdm(range(context_size, len(words), context_size)):
            # Option2: slidding window to create (context, target) pairs
            # for i in tqdm(range(context_size, len(words))):
                context = [word_to_ix[words[j]] for j in range(i - context_size, i) if words[j] in word_to_ix]
                if len(context) == context_size and words[i] in word_to_ix:
                    target = word_to_ix[words[i]]
                    data_contexts.append(context)
                    data_targets.append(target)

    # prepare  dataset
    dataset = TensorDataset(
        torch.tensor(data_contexts, dtype=torch.long),
        torch.tensor(data_targets, dtype=torch.long),
    )
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader