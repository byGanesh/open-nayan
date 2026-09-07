# Contributing

## What you can help with

### Parallel Scan (most urgent)
The state update inside Nayan is associative.
This means it can be parallelized using a standard
parallel prefix scan — O(log s) instead of O(s).

You do not need the formula to implement this.
The interface is:

    input:  sequence of matrices, each (dc, dc) complex
    output: their prefix sums, shape (seq_len, dc, dc) complex

A Triton kernel for this would remove the main
training bottleneck.

### Benchmarking
Train a standard transformer on identical:
- dataset (TinyStories)
- token budget
- parameter count
- hardware

Compare loss curves and generation quality.

### Long-context experiments
Run inference at 1K, 10K, 100K token sequences.
Does the architecture remain stable?

## What you cannot help with
The layer internals are not public.
Everything else is.

## Contact
[byganesh.com@gmail.com](byganesh.com@gmail.com)
