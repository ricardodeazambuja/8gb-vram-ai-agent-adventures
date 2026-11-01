# Overnight state (started 2026-10-31 evening, user asleep)

Rules: no push, PR, deletion, `claude -p`, or peekexec (nobody to answer; batch pibench for later).
New models only through the security routine. Commit each finished piece. Analyse before rerunning.

Queue (from docs/up-next.md "Faster and better fine-tuning"):
- [x] qwen-codecheck: container by default (--on-host to opt out)
- [x] input test repeatable, saved results
- [ ] train_style: preflight, checkpoints
- [ ] sd-cli: try our LoRA; SDXL norm bake
- [ ] automatic style score (no new model if avoidable)
- [ ] person: DoRA, LoRA + norm, Klein -r; leak fix
- [ ] Textual Inversion on person (SDXL)
- [ ] steering test 1 and 2 (cvector-generator on LFM3)

Next: first unchecked item.

User back at 07:00 on 2026-11-01: keep planning and working until then. Terminal safety: every heavy job in its
own scope (`systemd-run --user --scope -p MemoryMax=20G -p OOMPolicy=stop --unit=...`), one GPU job at a time,
nothing >15 GB while downloading, watch /proc/pressure/memory, no broad pkill, `timeout` on every job.
