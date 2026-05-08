# AGENTS.md

## Quick Start

- Install editable package from the repo root with `pip install -e .`.
- Launch the local Gradio demo with `python -m qwen_tts.cli.demo <model-or-path> --device cuda:0 --dtype float16 --no-flash-attn`.
- Use the scripts in `examples/` as manual capability checks. They are not pytest tests.

## Read First

- Start with [README.md](README.md) for the official model matrix, public API examples, and CLI usage.
- Read [memory-bank/architecture.md](memory-bank/architecture.md) before making structural changes.
- Read [memory-bank/tech-stack.md](memory-bank/tech-stack.md) before changing dependencies or model-loading behavior.
- Read [memory-bank/progress.md](memory-bank/progress.md) for local fork status, verified behavior, and known limitations.
- Read [memory-bank/gradio-demo-test-guide.md](memory-bank/gradio-demo-test-guide.md) for demo launch flags and troubleshooting.
- Read [finetuning/README.md](finetuning/README.md) before touching the fine-tuning pipeline.

## Repo Map

- `qwen_tts/inference/` is the public inference layer. Prefer editing here when behavior changes belong to the external API surface.
- `qwen_tts/core/models/` and `qwen_tts/core/tokenizer_12hz/` are the main low-level implementation layers.
- `qwen_tts/core/tokenizer_25hz/` is legacy compatibility code. Do not route new feature work here unless the task is explicitly about 25Hz support.
- `qwen_tts/cli/demo.py` is the Gradio demo entrypoint and the closest thing to an app layer in this repo.
- `examples/` contains runnable examples and manual validation scripts.
- `memory-bank/` contains maintainer knowledge for this fork and should be treated as authoritative local context.

## Project Conventions

- Keep changes minimal and preserve the Hugging Face style `from_pretrained` workflows already used throughout the repo.
- Treat `transformers==4.57.3` as pinned. Do not change it casually.
- Prefer the 12Hz tokenizer path for current work unless a task explicitly targets 25Hz.
- Remember the three model families map to different APIs:
  - `CustomVoice` -> `generate_custom_voice`
  - `VoiceDesign` -> `generate_voice_design`
  - `Base` -> `generate_voice_clone`
- The 0.6B CustomVoice variant does not support `instruct` the same way as the 1.7B instruct-capable models. Check the current code path before assuming parity.

## Validation Expectations

- There is no formal pytest suite in this repo.
- Prefer the narrowest runnable validation available:
  - an `examples/test_*.py` script for inference or tokenizer changes
  - `python -m qwen_tts.cli.demo ...` for demo-layer behavior
  - the documented fine-tuning commands for `finetuning/` work
- If you change model loading or audio decode behavior, explicitly preserve the current fix that keeps the speech tokenizer on `float32` when the main model is loaded in half precision.

## Environment Notes

- This fork already has a working Python environment. Do not spend time re-litigating environment setup unless the task is explicitly about it.
- Do not anchor guidance to older host-specific GTX 1660 notes unless the current task is specifically about that machine.
- `--no-flash-attn` is the safe default for local Windows demo work unless the task confirms compatible hardware and FlashAttention support.

## Documentation Rules

- Link to `memory-bank` documents instead of duplicating their content in new instructions or summaries.
- When maintainer docs and upstream README disagree, prefer the fork-local `memory-bank` docs for this workspace and call out the discrepancy explicitly.
- Keep `memory-bank` content focused on verified local behavior, not speculative plans presented as facts.