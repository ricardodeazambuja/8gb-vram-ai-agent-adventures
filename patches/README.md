# Patches

| File | What | Applies to |
|---|---|---|
| `llama.cpp-dory-pr30306.patch` | Dory support (upstream PR #30306). Already in [our fork](https://github.com/ricardodeazambuja/llama.cpp/tree/adventures); kept for reference. | upstream llama.cpp |
| `buun-flashmap-drafter-c1d1a39.patch` | The FlashMap flow-map speculative drafter, ported as a DFlash variant (local experiment, 2026-10-20). Unfinished: see [the graveyard](../docs/graveyard.md#flashmap-drafter-unfinished). | [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) at `34ef0ef7c6eaa58afd2011cf4385e8b2d76b225b` |
| `buun-nvcc12.2-moe-cpu-share-fix.patch` | Two lines that let nvcc 12.2 build `moe-cpu-share.cu` next to gcc 12.4 (avoids `<immintrin.h>`). | the same buun commit, after the FlashMap patch |

The buun checkout these came from was deleted on 2026-11-02 with the rest of `vendor/`; both patches were checked to
apply cleanly to `34ef0ef` first:

```bash
git clone https://github.com/spiritbuun/buun-llama-cpp && cd buun-llama-cpp && git checkout 34ef0ef7c6ea
git am ../patches/buun-flashmap-drafter-c1d1a39.patch
git apply ../patches/buun-nvcc12.2-moe-cpu-share-fix.patch
```

These stay out of our fork on purpose: nothing here uses them.
