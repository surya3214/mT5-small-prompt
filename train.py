"""Fine-tune mT5-small for slot extraction under different prompt conditions."""

import json
import os
import random
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ── Prompt templates ──────────────────────────────────────────────────────────

BARE_PROMPT = "extract: {text}"
SCHEMA_PROMPT = (
    "extract fields "
    "(add=address, act=action, record=item type, time=time period): "
    "{text}"
)


def format_input(text, prompt_mode):
    """Format input text with the appropriate prompt prefix."""
    template = BARE_PROMPT if prompt_mode == "bare" else SCHEMA_PROMPT
    return template.format(text=text)


def get_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────


class SlotDataset(Dataset):
    def __init__(self, data, tokenizer, prompt_mode, max_input_len=64, max_target_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_mode = prompt_mode
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = format_input(item["input"], self.prompt_mode)
        target_text = json.dumps(item["target"], sort_keys=True)

        enc_in = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc_tgt = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = enc_tgt["input_ids"].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc_in["input_ids"].squeeze(),
            "attention_mask": enc_in["attention_mask"].squeeze(),
            "labels": labels,
        }


# ── Training ──────────────────────────────────────────────────────────────────


def train(data_path, prompt_mode, output_dir, epochs=10, batch_size=16, lr=3e-4, seed=42):
    """Train mT5-small and return per-epoch average loss list."""
    random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    print(f"  Device: {device}")

    with open(data_path) as f:
        data = json.load(f)
    print(f"  Training examples: {len(data)}")

    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use bfloat16 on MPS/CUDA for speed without float16 numerical overflow (NaNs)
    dtype = torch.bfloat16 if device.type in ("mps", "cuda") else torch.float32
    model = MT5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)

    dataset = SlotDataset(data, tokenizer, prompt_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_log = []
    t0 = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_log.append(avg_loss)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>2}/{epochs} | Loss: {avg_loss:.4f} | Elapsed: {elapsed:.0f}s")

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "loss_log.json"), "w") as f:
        json.dump(loss_log, f)

    print(f"  Checkpoint → {output_dir}/")
    return loss_log


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.json")
    p.add_argument("--mode", choices=["bare", "schema"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    train(args.data, args.mode, args.output, args.epochs, args.batch_size, args.lr)
