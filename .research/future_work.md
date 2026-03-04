# Future Work

## F1 - Allow reporting via wandb
- update training command to do wandb training
- implement metric reporting to use wandb
- check it works
- make sure that pre and post training have different prefixes

## F2 - Sweeps


## F3 Code to save best model version to hugging face in regular and gguf format



## E2/R2 — Test training and evaluation batch sizes independently

Training stores activations for the backward pass; eval only needs a forward pass. These have
different memory profiles, so the max eval batch size may be significantly larger than the max
training batch size. HuggingFace Trainer's `auto_find_batch_size` had a known issue where it
reduced training batch size when OOM occurred during evaluation.

Future: add a separate eval batch size probing pass after finding training batch size, and report
both. This would allow the user to set `per_device_eval_batch_size` independently for maximum
throughput.

---

## R4 — CUDA allocator fragmentation mitigation

Memory fragmentation can cause OOM even when sufficient free VRAM exists in aggregate. Best
practice:

- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` before the probing loop, or document it as
  a user-level environment variable to configure.
- Call `torch.cuda.reset_peak_memory_stats()` between probes for diagnostic reporting of peak
  memory per batch size.

---

## P5 — Gradient accumulation guidance in output

After reporting the max per-device batch size, it would be useful to suggest how to achieve
common effective batch sizes (e.g. 128) using `gradient_accumulation_steps`:

```
gradient_accumulation_steps = target_effective_batch_size / per_device_batch_size
```

Future: add an optional `--target-effective-batch-size` flag that auto-computes and prints the
recommended `gradient_accumulation_steps` value.
