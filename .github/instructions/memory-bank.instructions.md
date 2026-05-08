---
applyTo: "memory-bank/**/*.md"
description: "Use when editing or summarizing memory-bank docs, progress notes, local test guides, architecture notes, or implementation plans for this fork."
---

# Memory-Bank Instructions

- Treat `memory-bank/` as fork-maintainer documentation, not upstream product docs.
- Keep these files concise, factual, and grounded in verified repository behavior.
- Prefer updating an existing memory-bank file over creating a new one when the topic already has a natural home.
- Keep the writing language and tone consistent with the surrounding file. Most current memory-bank docs are written in Chinese and should stay that way unless the file already uses English.
- Link back to authoritative workspace docs such as `README.md` or `finetuning/README.md` instead of copying long command blocks unless the file is specifically a runbook.
- Preserve the distinction between:
  - upstream repository facts
  - this fork's local fixes and validations
  - future plans that are still undecided
- When documenting local runtime behavior, prefer the current verified environment from `memory-bank/progress.md` and `memory-bank/implementation-plan.md`.
- Do not keep repeating older host-specific GPU narratives unless they are still required for the current machine or the current troubleshooting task.
- If a note describes a fix, include the symptom, root cause, affected file or layer, and how it was validated.
- If a note becomes stale, update or remove it rather than appending a conflicting paragraph.