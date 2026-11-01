# Overnight state (started 2026-10-31 evening, user asleep)

Rules: no push, PR, deletion, `claude -p`, or peekexec (nobody to answer; batch pibench for later).
New models only through the security routine. Commit each finished piece. Analyse before rerunning.

Queue (from docs/up-next.md "Faster and better fine-tuning"):
- [x] qwen-codecheck: container by default (--on-host to opt out)
- [x] input test repeatable, saved results
- [x] train_style: preflight, checkpoints, dora, lora+norm
- [x] sd-cli: LoRA loads (1120/1120); bake_sdxl.py works, watercolour likeness kept
- [ ] automatic style score (no new model if avoidable)
- [x] person: DoRA, LoRA + norm, Klein -r (leak fix: word yes; reg LoRA see log)
- [x] Textual Inversion on person (SDXL)
- [x] steering tests 1, 2, 4; written up in docs/fine-tuning.md

Next: first unchecked item.

User back at 07:00 on 2026-11-01: keep planning and working until then. Terminal safety: every heavy job in its
own scope (`systemd-run --user --scope -p MemoryMax=20G -p OOMPolicy=stop --unit=...`), one GPU job at a time,
nothing >15 GB while downloading, watch /proc/pressure/memory, no broad pkill, `timeout` on every job.
- bake_sdxl save is slow (10+ min, /backup disk near full); needs 30 min timeout
- Running: loop/person.sh (dora, lora+norm, ckpts); logs in loop/logs. Then run invert.py --fit, full invert, eval_sdcli.py on subject-word/-lora/-norm. Sweep of concise vector on CPU in bench/steering. Next after: compare, Klein -r on person, inversion.
- 00:05 concise vector: +-0.5 and up breaks LFM3 completely (blank or word salad), so sweep +-0.02..0.3 next, on the GPU after the image queue. CPU sweep during training halves training speed (laptop power); never overlap them.
- Queue after person.sh: loop/queue2.sh
- DoRA train timed out at 1800 s (CPU sweep slowed it), eval then silently ran the base model (images moved to models/image-gen/invalid/; eval now refuses). Rerun DoRA with 3000 s after queue2. Never overlap.
- 00:55 LoRA+norm done, about level with LoRA, 250 steps already has glasses and beard; leak remains. Queue after queue2: loop/queue3.sh (klein -r, reg pictures, reg LoRA and norm, DoRA rerun, Van Gogh 3000 steps). No overlap.
- 01:15 fine sweep: +0.05 cuts answers 217 -> 151 tokens, 2/2 facts; +0.1 breaks; -0.05 lengthens. Window is narrow. queue3 running (ETA ~05:00). After it: doom-loop vector (bench/steering), pibench if firewall preflight is clean (no peekexec), then docs and final report by 07:00.
- 01:40 (DST ended: clock repeated 01:xx, 07:00 is later than it looks) queue3: word 8 KB strong and LEAK-FREE; Klein -r good with no training; DoRA strong, leaks. reg runs and Van Gogh 3k failed on my argparse (--tag needs =; mamba ate --n). Order now: retry vector sanity, pibench (queue4), then loop/queue5.sh (reg LoRA, word on styles, stack, 3k), docs, report by 07:00.
- 03:40 pibench with retry vector at +0.03: 0/6, repeats 93% -> 80%. Docs done. queue5 running; after it: view reg leak, word on styles, 3k; docs; stack test; final report at 07:00.
- 04:45 8 KB style word on Van Gogh looks stronger than the 1000-step LoRA, and leak-free, with a green cast (by eye). Next: gris word, 3k LoRA, then loop/queue6.sh (stack person + style words), docs, report.
- 06:15 all queues done. Stack works on Van Gogh, weak on Gris; word weight 0.7 no effect. Docs updated. Only the final report is left at 07:00.
- 06:12 last test: Van Gogh word from "art" instead of "painting" (run vangogh-word-art), to see if the green cast comes from the start word. Then final report.
- 06:57 green cast is not the start word (art start greener). Overnight work complete; final report given.
