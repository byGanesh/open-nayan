# Nayan

An attention-free sequence architecture with fixed-memory complexity.

## What is Nayan?

Nayan is a novel language model architecture that processes sequences
without attention mechanisms. It maintains a fixed-size internal state
regardless of sequence length.

## Key Properties

- No attention
- O(d²) memory, fixed regardless of sequence length
- Causal by design
- Complex-valued representations
- No separate positional encoding needed
- Theoretically supports 10M+ context windows

## Architecture

Each layer contains exactly two learned complex matrices:
- W_M of shape (dc, dc)
- W_R of shape (dc, dc)

where dc = d // 2.

Each layer also maintains a dynamic state C of shape (batch, dc, dc) that accumulates information causally across the sequence.

C is initialized to zero at the start of each sequence.  
C is updated once per token, after that token is processed.
  
Input:  token ids  (batch, seq_len)  
Output: logits     (batch, seq_len, vocab_size)  

## Results

| Config            | Params | Tokens | Loss | Time       |
|-------------------|--------|--------|------|------------|
| d=128, layers=2   | 13M    | 200K   | 3.48 | 42 min (2× T4) |
| d=256, layers=4   | 39M    | 5M     | 2.91 | 43 min (2× T4) |

Generation after 10 epochs on TinyStories (200K tokens, 13M params):

```
prompt: 'Once upon a time'
output: Once upon a time there was a little girl named Lily.
        She loved to play. One day, she went to the park.

prompt: 'He went to'
output: He went to the hospital.
```

## Open Problems (help wanted)

### 1. Parallel Scan
The architecture processes tokens sequentially — one token at a time.
The internal state update is associative, which means it can in principle be parallelized using a parallel prefix scan in O(log s) steps instead of O(s) sequential steps.

A Triton or CUDA kernel for this would make training competitive with transformers in wall-clock time.

### 2. Benchmarking
Honest head-to-head comparison against a transformer of equivalent parameter count, token budget, and hardware.

### 3. Long-context validation
Empirical validation of stable behavior at 10K, 100K, 1M+ token sequences.

## What is not public

The mechanism inside each layer — how W_M, W_R, and C interact — is proprietary.

Research collaborators can access the full specification.  
Contact: [byganesh.com@gmail.com](byganesh.com@gmail.com)

## Collaboration

If you want to help with the open problems above:
- You do not need the formula to work on the parallel scan
- You do not need the formula to run benchmarks
- You do not need the formula to test long-context stability

Open a GitHub issue or email directly.

## License
Apache 2.0  
[Ganesh Kumar](https://www.byganesh.com).
