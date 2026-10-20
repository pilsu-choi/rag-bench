---
type: Lesson
title: hf download flag pitfalls
description: "--exclude takes multiple values greedily, and bge-m3 only has .bin weights."
tags: [lesson, huggingface]
timestamp: 2026-10-20T09:15:00Z
---

# hf download flag pitfalls

- `hf download repo --exclude "a" "b"` treats `"b"` as a **filename to download**. It then ignores
  `--exclude` and fails with "File not found". Repeat the flag instead:
  `--exclude a --exclude b`.
- `BAAI/bge-m3` has no safetensors on `main`; the weights are `pytorch_model.bin` (2.2 GB).
  Excluding `*.bin` downloads only 24 MB of config. Fetch `pytorch_model.bin` explicitly.
- `uvx` is in `~/.local/bin`, which is not on PATH in a non-login `ssh host 'cmd'` shell. Use the
  full path.
