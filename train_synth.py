import os
import time
import math
import dataclasses
from typing import Iterator, Tuple

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader

import tiktoken
from datasets import load_dataset

from model.models import GPT
from model.config import TransformerConfig


# ==========================
# Hyperparameters
# ==========================

MODEL_EMBED_DIM = 768
MODEL_NUM_HEADS = 12
MODEL_ATTENTION_DIM = 64
MODEL_BLOCKS = 6
MODEL_FF_DIM = 2048
SEQUENCE_LENGTH = 512

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1  # set >1 to emulate larger batch

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
MAX_STEPS = 200
LOG_INTERVAL = 50

SAVE_INTERVAL = 500

MAX_TRAIN_EXAMPLES = 100_000  # set to None to use full SYNTH train split

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "gpt_synth_latest.pt")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SynthTokenDataset(IterableDataset):
    """
    Lazily tokenizes the SYNTH dataset into fixed-length language modeling sequences.

    Each yielded sample is a pair (input_ids, target_ids), both of shape [sequence_length].
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        sequence_length: int,
    ) -> None:
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        # Use GPT-2 EOS token
        self.eos_id = self.tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        token_buffer = []

        # We iterate over the (already shuffled) HF dataset and lazily tokenize rows.
        for row in self.hf_dataset:
            text = (
                row["query"]
                + "\n\n"
                + row["synthetic_reasoning"]
                + "\n\n"
                + row["synthetic_answer"]
            )

            tokens = self.tokenizer.encode(
                text,
                allowed_special={"<|endoftext|>"},
            )

            token_buffer.extend(tokens)
            token_buffer.append(self.eos_id)

            # Emit as many full sequences as possible from the buffer
            # We advance by sequence_length tokens each time (non-overlapping chunks).
            while len(token_buffer) >= self.sequence_length + 1:
                chunk = token_buffer[: self.sequence_length + 1]
                token_buffer = token_buffer[self.sequence_length :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)

                yield input_ids, target_ids


def build_model(vocab_size: int) -> GPT:
    config = TransformerConfig(
        embedding_dim=MODEL_EMBED_DIM,
        num_attention_heads=MODEL_NUM_HEADS,
        attention_dim=MODEL_ATTENTION_DIM,
        block_num=MODEL_BLOCKS,
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=vocab_size,
        feed_forward_dim=MODEL_FF_DIM,
        flash_attention=True,
    )
    return GPT(config)


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    tokens_processed: int,
) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config = dataclasses.asdict(model.config)

    # Timestamped filename for history
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"gpt_synth_step{step}_{timestamp}.pt"
    timestamp_path = os.path.join(CHECKPOINT_DIR, filename)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
        "step": step,
        "tokens_processed": tokens_processed,
    }

    # Save checkpoint with timestamped name
    torch.save(
        payload,
        timestamp_path,
    )

    # Also update "latest" symlink-style file for resuming
    torch.save(
        payload,
        CHECKPOINT_PATH,
    )
    print(
        f"Saved checkpoint to {timestamp_path} "
        f"(latest -> {CHECKPOINT_PATH}, step={step}, tokens={tokens_processed})"
    )


def train() -> None:
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    device = get_device()
    print(f"Using device: {device}")

    # --------------------------
    # Tokenizer & dataset
    # --------------------------
    enc = tiktoken.get_encoding("gpt2")
    print(f"Vocabulary size: {enc.n_vocab}")

    print("Loading SYNTH dataset ...")
    raw_train = load_dataset("PleIAs/SYNTH", split="train")
    print(f"Loaded full train split with {len(raw_train)} examples")

    # Optionally downsample to a manageable subset before any shuffling/selects
    if MAX_TRAIN_EXAMPLES is not None and len(raw_train) > MAX_TRAIN_EXAMPLES:
        print(f"Subsampling to first {MAX_TRAIN_EXAMPLES} examples for training")
        raw_train = raw_train.select(range(MAX_TRAIN_EXAMPLES))

    # Shuffle once so our iterable is roughly randomized
    raw_train = raw_train.shuffle(seed=42)
    print(f"Using {len(raw_train)} examples after optional subsample & shuffle")

    # Optional: small validation split (take a tail slice)
    val_size = min(2_000, len(raw_train) // 20)  # up to 5% of data
    if val_size > 0:
        raw_val = raw_train.select(range(len(raw_train) - val_size, len(raw_train)))
        raw_train = raw_train.select(range(0, len(raw_train) - val_size))
        print(f"Using {len(raw_train)} train examples, {len(raw_val)} val examples")
    else:
        raw_val = None
        print("Using full dataset for training only (no validation split)")

    train_dataset = SynthTokenDataset(
        hf_dataset=raw_train,
        tokenizer=enc,
        sequence_length=SEQUENCE_LENGTH,
    )

    # Use a small number of workers to overlap CPU preprocessing with GPU,
    # but keep it modest to avoid excessive disk thrash.
    num_workers = 2 if device.type == "cuda" else 0

    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # not allowed for IterableDataset
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # Build an (optional) validation iterator that yields a finite number
    if raw_val is not None:
        val_dataset = SynthTokenDataset(
            hf_dataset=raw_val,
            tokenizer=enc,
            sequence_length=SEQUENCE_LENGTH,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
    else:
        val_loader = None

    # --------------------------
    # Model & optimizer
    # --------------------------
    model = build_model(vocab_size=enc.n_vocab)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Resume from checkpoint if present
    step = 0
    tokens_processed = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint at {CHECKPOINT_PATH}, loading ...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt.get("step", 0)
        tokens_processed = ckpt.get("tokens_processed", 0)
        print(f"Resuming from step={step}, tokens_processed={tokens_processed}")

    print(
        f"Training for {MAX_STEPS} steps | "
        f"batch_size={BATCH_SIZE}, seq_len={SEQUENCE_LENGTH}, "
        f"tokens_per_step={BATCH_SIZE * SEQUENCE_LENGTH}"
    )

    running_loss = 0.0
    running_tokens = 0
    start_time = time.time()

    data_iter = iter(dataloader)

    try:
        for global_step in range(step + 1, step + 1 + MAX_STEPS):
            # Gradient accumulation loop
            for micro_step in range(GRAD_ACCUM_STEPS):
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    # Re-create the iterator when we exhaust the underlying iterable
                    data_iter = iter(dataloader)
                    inputs, targets = next(data_iter)

                inputs = inputs.to(device, non_blocking=True)  # [B, T]
                targets = targets.to(device, non_blocking=True)  # [B, T]

                with torch.amp.autocast('cuda'):
                    _, loss = model(inputs, targets)
                    loss = loss / GRAD_ACCUM_STEPS

                scaler.scale(loss).backward()

                batch_tokens = inputs.numel()
                tokens_processed += batch_tokens
                running_tokens += batch_tokens
                running_loss += loss.item() * GRAD_ACCUM_STEPS  # undo division for logging

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / LOG_INTERVAL
                ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                toks_per_sec = running_tokens / max(elapsed, 1e-6)
                sec_per_step = elapsed / LOG_INTERVAL

                total_target_step = step + MAX_STEPS
                steps_done = global_step - step
                steps_remaining = max(total_target_step - global_step, 0)
                eta_seconds = steps_remaining * sec_per_step
                eta_h = int(eta_seconds // 3600)
                eta_m = int((eta_seconds % 3600) // 60)
                eta_s = int(eta_seconds % 60)

                print(
                    f"Step {global_step:6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"ppl {ppl:.2f} | "
                    f"tokens {tokens_processed:,} | "
                    f"tokens/s {toks_per_sec:,.0f} | "
                    f"step_time {sec_per_step:.3f}s | "
                    f"eta {eta_h:02d}:{eta_m:02d}:{eta_s:02d} "
                    f"({steps_remaining} steps left)"
                )

                running_loss = 0.0
                running_tokens = 0
                start_time = time.time()

            if global_step % SAVE_INTERVAL == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    tokens_processed=tokens_processed,
                )

    except KeyboardInterrupt:
        print(f"\nInterrupted, saving checkpoint ...")
    finally:
        # Save at the last global_step reached
        last_step = global_step if "global_step" in locals() else step
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=last_step,
            tokens_processed=tokens_processed,
        )


if __name__ == "__main__":
    train()


