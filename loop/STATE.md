# Overnight state (started 2026-10-31 evening, user asleep)

Rules: no push, PR, deletion, `claude -p`, or peekexec (nobody to answer; batch pibench for later).
New models only through the security routine. Commit each finished piece. Analyse before rerunning.

Queue (from docs/up-next.md "Faster and better fine-tuning"):
- [x] qwen-codecheck: container by default (--on-host to opt out)
- [x] input test repeatable, saved results
- [x] train_style: preflight, checkpoints, dora, lora+norm
- [x] sd-cli: LoRA loads (1120/1120); bake_sdxl.py works, watercolour likeness kept
- [ ] automatic style score (no new model if avoidable)
- [ ] person: DoRA, LoRA + norm, Klein -r; leak fix
- [ ] Textual Inversion on person (SDXL)
- [ ] steering test 1 and 2 (cvector-generator on LFM3)

Next: first unchecked item.

User back at 07:00 on 2026-11-01: keep planning and working until then. Terminal safety: every heavy job in its
own scope (`systemd-run --user --scope -p MemoryMax=20G -p OOMPolicy=stop --unit=...`), one GPU job at a time,
nothing >15 GB while downloading, watch /proc/pressure/memory, no broad pkill, `timeout` on every job.
- bake_sdxl save is slow (10+ min, /backup disk near full); needs 30 min timeout
- Running: loop/person.sh (dora, lora+norm, ckpts); logs in loop/logs. Then run invert.py --fit, full invert, eval_sdcli.py on subject-word/-lora/-norm. Sweep of concise vector on CPU in bench/steering. Next after: compare, Klein -r on person, inversion.
- 00:05 concise vector: +-0.5 and up breaks LFM3 completely (blank or word salad), so sweep +-0.02..0.3 next, on the GPU after the image queue. CPU sweep during training halves training speed (laptop power); never overlap them.
- Queue after person.sh: loop/queue2.sh
