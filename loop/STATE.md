# Overnight state (started 2026-10-31 evening, unattended)

Rules: no push, PR, deletion or root prompts (nobody to answer; batch pibench for later).
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

Stop at 07:00 on 2026-11-01: keep planning and working until then. Terminal safety: every heavy job in its
own scope (`systemd-run --user --scope -p MemoryMax=20G -p OOMPolicy=stop --unit=...`), one GPU job at a time,
nothing >15 GB while downloading, watch /proc/pressure/memory, no broad pkill, `timeout` on every job.
- bake_sdxl save is slow (10+ min, /backup disk near full); needs 30 min timeout
- Running: loop/person.sh (dora, lora+norm, ckpts); logs in loop/logs. Then run invert.py --fit, full invert, eval_sdcli.py on subject-word/-lora/-norm. Sweep of concise vector on CPU in bench/steering. Next after: compare, Klein -r on person, inversion.
- 00:05 concise vector: +-0.5 and up breaks LFM3 completely (blank or word salad), so sweep +-0.02..0.3 next, on the GPU after the image queue. CPU sweep during training halves training speed (laptop power); never overlap them.
- Queue after person.sh: loop/queue2.sh
- DoRA train timed out at 1800 s (CPU sweep slowed it), eval then silently ran the base model (images moved to models/image-gen/invalid/; eval now refuses). Rerun DoRA with 3000 s after queue2. Never overlap.
- 00:55 LoRA+norm done, about level with LoRA, 250 steps already has glasses and beard; leak remains. Queue after queue2: loop/queue3.sh (klein -r, reg pictures, reg LoRA and norm, DoRA rerun, Van Gogh 3000 steps). No overlap.
- 01:15 fine sweep: +0.05 cuts answers 217 -> 151 tokens, 2/2 facts; +0.1 breaks; -0.05 lengthens. Window is narrow. queue3 running (ETA ~05:00). After it: doom-loop vector (bench/steering), pibench if firewall preflight is clean (no root prompt), then docs and final report by 07:00.
- 01:40 (DST ended: clock repeated 01:xx, 07:00 is later than it looks) queue3: word 8 KB strong and LEAK-FREE; Klein -r good with no training; DoRA strong, leaks. reg runs and Van Gogh 3k failed on my argparse (--tag needs =; mamba ate --n). Order now: retry vector sanity, pibench (queue4), then loop/queue5.sh (reg LoRA, word on styles, stack, 3k), docs, report by 07:00.
- 03:40 pibench with retry vector at +0.03: 0/6, repeats 93% -> 80%. Docs done. queue5 running; after it: view reg leak, word on styles, 3k; docs; stack test; final report at 07:00.
- 04:45 8 KB style word on Van Gogh looks stronger than the 1000-step LoRA, and leak-free, with a green cast (by eye). Next: gris word, 3k LoRA, then loop/queue6.sh (stack person + style words), docs, report.
- 06:15 all queues done. Stack works on Van Gogh, weak on Gris; word weight 0.7 no effect. Docs updated. Only the final report is left at 07:00.
- 06:12 last test: Van Gogh word from "art" instead of "painting" (run vangogh-word-art), to see if the green cast comes from the start word. Then final report.
- 06:57 green cast is not the start word (art start greener). Overnight work complete; final report given.

## Plan from 2026-11-01 morning: LFM3-VL beyond vision (approved; see docs/up-next.md top)
Pilots, think before cutting, read results with sample size in mind. No push/PR/deletion.
Order: replay test, graft, vision fine-tune on graft or LoRA on VL, two-model fallback.
- 09:55 replay on 13 stuck points, 4 samples each: exact repeats VL 15/52 (6 points), VL temp 0.7 1/52 (1 point), window 11/52, both 0/52, 1.7B 0/52 (0 points). VL new calls often near-copies; 1.7B changes approach. Replay only partly reproduces real loops (no thinking text, rebuilt tool schemas). Graft pibench equals plain 1.7B unless images are used, so only vision is tested. Next: graft glance (queue8) then pibench VL TEMP=0.7, 6 runs.
- 10:32 graft is blind: think off 14/150 (VL 109), think on 10/48 before end; answers restate the prompt. Weights: cos ~0.8, relative distance ~0.6, token_embd cos 0.46 (weight_gap.py), so blend and short fine-tune dropped. Next: read pibench VL TEMP=0.7, then LoRA on VL from solved 1.7B runs (sandbox), or two-model fallback.
- 10:50 found: real prompts contained thinking (recorded tokens match with-think plus ~450, not no-think), so replay v1 understated loops. replay.py fixed. Next after pibench-t: loop/queue9.sh (replay v2, 1.7B replies), make_agent_data.py data/stuck.json, fit test train_agent via SCRIPT=/bench/beyond-vision/train_agent.py sandbox.sh, train, convert to GGUF, score replay held-out points, glance, pibench.
- 11:20 pibench VL TEMP=0.7: 0/6 solved but loops gone: repeats 24% (was 93%), worst streak 3 (was 769), 3 timeouts, tests 30/54; scrape-books 2/5 was always 0. Graft think on: 10/48 before timeout. Running queue9.
- 11:50 replay v2 (thinking in history): VL default 32/52 repeats (11/13 points), temp 0.7 8/52 (5 points), 1.7B 0/52 (0). Weights cause confirmed. Training agent LoRA: 177 examples, 4 epochs, cap 2300 tokens, 5.5 GiB. Odd stuck points held out. Then convert, replay held-out, glance n30, pibench at TEMP=0.7.
- 13:25 agent LoRA: replay held-out default 21/24 -> 7/24, vision 110 -> 101 of 150, pibench 0/6 at temp 0.7, tests 27/54 (temp alone 30/54), repeats 48% (24%). Likely overfit (loss 0.84 -> 0.25 on 177 examples). Last item: eyes probe (queue11), then docs and report.
- 13:40 eyes probe: LFM3-1.7B driving, VL reporting: 57/60 (VL alone 60/60), 5.0 GB VRAM for both, median 11 s eye + 2 s driver. No 12B mmproj on disk. Running queue12 (Gemma, Groot n=6 think on). Then projector pilot: 1.7B weights from GGUF in VL's code, sandbox, train only multi_modal_projector, fit test first. Then docs and report.
- 13:50 gguf_to_state.py verified 286/286 tensors (cos >=0.99 against VL safetensors) and exported 1.7B to proj/. train_projector.py and queue13 ready; run after queue12.
- 13:55 Groot glance think on n=6: 27/30, mean 18 s, 7.2 GB. Gemma failed (wrong script), queue14 fixes it after queue13.
- 14:20 projector run: 35/150, but answers are near constant per scene (blind, loss flat at 0.3 from step 30). Base-rate result, not reading. queue15: overfit test after Gemma tells "cannot carry info" from "needs more data". Then docs, report.
- 14:45 plan done: Gemma glance 30/30 6.2 GB 17 s, overfit loss 0.15 not clear. Written up in docs/fine-tuning.md. Only the report remains.
- 2026-11-01 repro audit: docs/reproduce.md is now the map; bench/check_numbers.py checks the numbers. Clean-clone replay matched 32/52, 8/52, 0/52 at every point. This file stays a journal; see loop/README.md.

## Night of 2026-11-01 (user asleep, unattended)
Goal from the user: every model runs on our llama.cpp fork; keep fixing problems and improving; if time, add VibeThinker.
Rules: no push (repo or fork), no PR, no root prompts, no deletions. Commit each finished piece locally. One GPU job at a time, own scope, MemoryMax=20G, timeouts.
Queue:
- [x] Commit fork-by-default change once the fresh build (scratchpad/forkbuild) passes
- [x] Read audit WARNs for the 12 unaudited models, make audit-gguf read PTQ1_0, record rows in docs/security.md
- [x] bench/recheck: launch every chat model on the fork, save qwen-bench and codecheck output, compare with docs
- [x] VibeThinker added: launcher, pinned download, page, shim. Verify agent mode after recheck
- [x] Bonsai 2 on the fork: QMODEL=bonsai2q2, Q2_0, 60 layers, 14.3 tok/s, 0/500 wrong. No port needed.
- [x] Fix stale items; final report
Not possible tonight: pibench (firewall may need root), pushing, and running VibeThinker as an agent on the host (blocked, correctly: first shell belongs in pibench).
