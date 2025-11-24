import time
import math

import torch
from torch import Tensor

from model.models import GPT
from model.config import TransformerConfig


# ==========================
# Hyperparameters (match train_synth as much as possible)
# ==========================

MODEL_EMBED_DIM = 768
MODEL_NUM_HEADS = 12
MODEL_ATTENTION_DIM = 64
MODEL_BLOCKS = 6
MODEL_FF_DIM = 2048
SEQUENCE_LENGTH = 512

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1  # keep simple here; adjust if you want

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
MAX_STEPS = 200
LOG_INTERVAL = 50

# GPT-2 vocab size; matches tiktoken "gpt2" encoding used in train_synth
VOCAB_SIZE = 50257


def get_device() -> torch.device:
    if torch.cuda.is_available():
        # Match your training script setting for flash attention
        torch.backends.cuda.enable_flash_sdp(True)
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def benchmark_compute() -> None:
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    device = get_device()
    print(f"[compute benchmark] Using device: {device}")

    # --------------------------
    # Model & optimizer
    # --------------------------
    model = build_model(vocab_size=VOCAB_SIZE)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # --------------------------
    # Synthetic "DataLoader"
    # --------------------------
    # Pre-allocate one batch of random tokens on the device and reuse it.
    # This removes all CPU / I/O overhead and isolates pure model compute.
    inputs: Tensor = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(BATCH_SIZE, SEQUENCE_LENGTH),
        dtype=torch.long,
        device=device,
    )
    # Targets can be arbitrary; for throughput, structure doesn't matter.
    targets: Tensor = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(BATCH_SIZE, SEQUENCE_LENGTH),
        dtype=torch.long,
        device=device,
    )

    print(
        f"[compute benchmark] "
        f"batch_size={BATCH_SIZE}, seq_len={SEQUENCE_LENGTH}, "
        f"tokens_per_step={BATCH_SIZE * SEQUENCE_LENGTH}"
    )

    running_loss = 0.0
    running_tokens = 0
    tokens_processed = 0
    start_time = time.time()

    try:
        for step in range(1, MAX_STEPS + 1):
            for micro_step in range(GRAD_ACCUM_STEPS):
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
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

            if step % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / LOG_INTERVAL
                ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                toks_per_sec = running_tokens / max(elapsed, 1e-6)
                sec_per_step = elapsed / LOG_INTERVAL

                print(
                    f"[compute benchmark] "
                    f"step {step:5d} | "
                    f"loss {avg_loss:.4f} | "
                    f"ppl {ppl:.2f} | "
                    f"tokens {tokens_processed:,} | "
                    f"tokens/s {toks_per_sec:,.0f} | "
                    f"step_time {sec_per_step:.4f}s"
                )

                running_loss = 0.0
                running_tokens = 0
                start_time = time.time()

    except KeyboardInterrupt:
        print("\n[compute benchmark] Interrupted.")

    total_time = time.time() - start_time
    overall_toks_per_sec = tokens_processed / max(total_time, 1e-6)
    print(
        f"[compute benchmark] Finished {MAX_STEPS} steps, "
        f"processed {tokens_processed:,} tokens "
        f"in {total_time:.2f}s "
        f"({overall_toks_per_sec:,.0f} tokens/s overall)."
    )


if __name__ == "__main__":
    benchmark_compute()


