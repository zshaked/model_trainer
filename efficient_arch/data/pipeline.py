"""
Shared data pipeline — identical for all model variants.

Uses HuggingFace datasets + GPT-2 tokenizer. Streams data to avoid
downloading the full dataset upfront (important for free-tier machines).
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np


def get_tokenizer(name: str = "gpt2"):
    """Load tokenizer. Downloads once, then cached."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class StreamingTextDataset(IterableDataset):
    """
    Streams tokenized text from HuggingFace datasets.
    Concatenates all documents and yields fixed-length chunks.
    No padding waste — every token is real data.
    """

    def __init__(self, dataset_name: str, tokenizer_name: str, seq_len: int,
                 split: str = "train", seed: int = 42):
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.seq_len = seq_len
        self.split = split
        self.seed = seed

    def __iter__(self):
        from datasets import load_dataset
        tokenizer = get_tokenizer(self.tokenizer_name)

        # Stream dataset — no full download needed
        if self.dataset_name == "openwebtext":
            ds = load_dataset("openwebtext", split=self.split,
                              streaming=True, trust_remote_code=True)
        elif self.dataset_name == "roneneldan/TinyStories":
            ds = load_dataset("roneneldan/TinyStories", split=self.split,
                              streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(self.dataset_name, split=self.split,
                              streaming=True, trust_remote_code=True)

        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        # Accumulate tokens across documents, yield fixed-length chunks
        buffer = []
        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            tokens = tokenizer.encode(text)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


class PreTokenizedDataset(Dataset):
    """
    For pre-tokenized data stored as numpy memmap.
    Faster than streaming after initial tokenization.
    """

    def __init__(self, data_path: str, seq_len: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def tokenize_and_save(dataset_name: str, output_path: str,
                      tokenizer_name: str = "gpt2",
                      max_tokens: int = None):
    """
    One-time tokenization: download dataset, tokenize, save as memmap.
    Run this once, then use PreTokenizedDataset for fast loading.

    Usage:
        python -m data.pipeline --tokenize openwebtext --output data/owt.bin
    """
    from datasets import load_dataset
    tokenizer = get_tokenizer(tokenizer_name)

    ds = load_dataset(dataset_name, split="train", streaming=True,
                      trust_remote_code=True)

    # First pass: count tokens to allocate memmap
    # (or just use a large pre-allocated file and truncate)
    all_tokens = []
    total = 0
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total += len(tokens)
        if max_tokens and total >= max_tokens:
            all_tokens = all_tokens[:max_tokens]
            break
        if total % 1_000_000 < 1000:
            print(f"  Tokenized {total:,} tokens...", flush=True)

    print(f"Total tokens: {len(all_tokens):,}")
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(output_path)
    print(f"Saved to {output_path} ({os.path.getsize(output_path) / 1e9:.2f} GB)")


def get_dataloader(config, split="train"):
    """Create dataloader from config. Uses streaming by default."""
    dataset = StreamingTextDataset(
        dataset_name=config.dataset,
        tokenizer_name=config.tokenizer,
        seq_len=config.max_seq_len,
        split=split,
        seed=config.seed,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=0,  # streaming datasets need 0 workers
        pin_memory=True,
    )


def get_eval_batches(config, num_tokens: int = 500_000):
    """Pre-load eval batches into memory for fast repeated evaluation."""
    batches = []
    tokens_so_far = 0
    loader = get_dataloader(config, split="validation")
    for x, y in loader:
        batches.append((x, y))
        tokens_so_far += x.numel()
        if tokens_so_far >= num_tokens:
            break
    return batches


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenize", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()
    tokenize_and_save(args.tokenize, args.output, max_tokens=args.max_tokens)
