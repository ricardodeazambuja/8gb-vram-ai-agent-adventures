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

## Plan from 2026-11-01 morning: LFM3-VL beyond vision (user awake, approved; see docs/up-next.md top)
Pilots, think before cutting, read results with sample size in mind. No push/PR/deletion.
Order: replay test, graft, vision fine-tune on graft or LoRA on VL, two-model fallback.
- 09:55 replay on 13 stuck points, 4 samples each: exact repeats VL 15/52 (6 points), VL temp 0.7 1/52 (1 point), window 11/52, both 0/52, 1.7B 0/52 (0 points). VL new calls often near-copies; 1.7B changes approach. Replay only partly reproduces real loops (no thinking text, rebuilt tool schemas). Graft pibench equals plain 1.7B unless images are used, so only vision is tested. Next: graft glance (queue8) then pibench VL TEMP=0.7, 6 runs.
- 10:32 graft is blind: think off 14/150 (VL 109), think on 10/48 before end; answers restate the prompt. Weights: cos ~0.8, relative distance ~0.6, token_embd cos 0.46 (weight_gap.py), so blend and short fine-tune dropped. Next: read pibench VL TEMP=0.7, then LoRA on VL from solved 1.7B runs (sandbox), or two-model fallback.
- 10:50 found: real prompts contained thinking (recorded tokens match with-think plus ~450, not no-think), so replay v1 understated loops. replay.py fixed. Next after pibench-t: loop/queue9.sh (replay v2, 1.7B replies), make_agent_data.py data/stuck.json, fit test train_agent via SCRIPT=/bench/beyond-vision/train_agent.py sandbox.sh, train, convert to GGUF, score replay held-out points, glance, pibench.
- 11:20 pibench VL TEMP=0.7: 0/6 solved but loops gone: repeats 24% (was 93%), worst streak 3 (was 769), 3 timeouts, tests 30/54; scrape-books 2/5 was always 0. Graft think on: 10/48 before timeout. Running queue9.
- 11:50 replay v2 (thinking in history): VL default 32/52 repeats (11/13 points), temp 0.7 8/52 (5 points), 1.7B 0/52 (0). Weights cause confirmed. Training agent LoRA: 177 examples, 4 epochs, cap 2300 tokens, 5.5 GiB. Odd stuck points held out. Then convert, replay held-out, glance n30, pibench at TEMP=0.7.
- 13:25 agent LoRA: replay held-out default 21/24 -> 7/24, vision 110 -> 101 of 150, pibench 0/6 at temp 0.7, tests 27/54 (temp alone 30/54), repeats 48% (24%). Likely overfit (loss 0.84 -> 0.25 on 177 examples). Last item: eyes probe (queue11), then docs and report.
