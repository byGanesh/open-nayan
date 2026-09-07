import torch
import torch.nn.functional as F

from config import *
from data   import get_tokenizer, load_tinystories, TextDataset, get_loader


# ── Known Engineering Challenges ───────────────────────────────────────────
#
# 1. SEQUENTIAL LOOP (main bottleneck)
#    The model processes tokens one at a time in a sequential loop.
#    This is O(s) in Python — no GPU parallelism across the sequence.
#    The internal state update is associative, so a parallel prefix scan
#    (like those used in Mamba/S4) would reduce this to O(log s) steps.
#    A Triton kernel for this is the single highest-impact contribution.
#
# 2. SINGLE GPU ONLY
#    The sequential state C cannot be split across GPUs with DataParallel.
#    Multi-GPU training requires a different parallelism strategy
#    (tensor parallelism or pipeline parallelism, not data parallelism).
#    Current workaround: single GPU, larger batch size.
#
# 3. COMPLEX GRADIENTS + AMP
#    torch.amp with float16 + GradScaler does not support complex gradients.
#    Workaround: bfloat16 autocast (no scaler needed).
#    bf16 has the same exponent range as float32 — no overflow issues.
#
# 4. GRAPH DEPTH WITH LONG SEQUENCES
#    Without the parallel scan fix, seq_len directly determines
#    how deep the computation graph is during backprop.
#    seq_len=64 caused OOM on 2x T4 (14.5GB each).
#    Current workaround: seq_len=32, which halves graph depth.
#    Proper fix: parallel scan removes the graph depth issue entirely.
#
# 5. TOKEN / PARAMETER RATIO
#    Chinchilla suggests ~20 tokens per parameter for optimal training.
#    Current experiments are severely undertrained by that standard.
#    This is a compute/data constraint, not an architectural one.
#    Results improve significantly with more tokens.
#
# ───────────────────────────────────────────────────────────────────────────


def train(model, dataset, epochs, lr, batch_size, device):
    model  = model.to(device)

    # NOTE: no DataParallel — sequential state C doesn't split across GPUs
    # see challenge #2 above
    print(f"device: {device}  GPUs available: {torch.cuda.device_count()}")
    print(f"using: 1 GPU (multi-GPU requires tensor/pipeline parallelism)")

    loader = get_loader(dataset, batch_size)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_loss = float('inf')
    print(f"dataset: {len(dataset):,}  batch: {batch_size}  seq: {SEQ_LEN}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches  = 0

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # bfloat16 autocast — no GradScaler needed
            # float16 + GradScaler raises NotImplementedError on complex grads
            # see challenge #3 above
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss   = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            if batch_idx % 200 == 0:
                print(
                    f"  batch {batch_idx}/{len(loader)}"
                    f"  loss: {loss.item():.4f}"
                )

        scheduler.step()
        avg = total_loss / n_batches

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "best_nayan.pt")

        print(
            f"Epoch {epoch+1:3d}/{epochs}"
            f"  loss: {avg:.4f}"
            f"  best: {best_loss:.4f}"
        )

    print("done.")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc     = get_tokenizer()
    ids     = load_tinystories(MAX_TOKENS, enc)
    dataset = TextDataset(ids, SEQ_LEN)

    # import your model here
    # from model import Nayan
    # model = Nayan(VOCAB, D, N_LAYERS)
    # print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # model = train(model, dataset, EPOCHS, LR, BATCH, device)
