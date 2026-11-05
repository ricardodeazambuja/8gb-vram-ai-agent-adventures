import numpy as np

# measured: llama-bench tg64, Qwen3.6-35B-A3B UD-IQ4_XS, -fa 1 -t 6, ncmoe -> tok/s
n = np.array([40,36,32,28]); ts = np.array([22.21,24.62,26.98,29.82])
ms = 1000/ts
b, a = np.polyfit(n, ms, 1)
print(f"fit: ms/token = {a:.2f} + {b:.3f} * n_cpu_layers   (resid {np.abs(np.polyval([b,a],n)-ms).max():.2f} ms)")
L, K, E = 40, 8, 256
exp_MB = 1.412*2**20/1e6
per_layer_MB = K*exp_MB
print(f"expert {exp_MB:.3f} MB; one layer's active experts {per_layer_MB:.1f} MB -> effective CPU expert bw {per_layer_MB/b:.1f} GB/s")
pcie = 6.45  # GB/s measured, pinned or pageable
cache_frac = 8/40  # same VRAM as -ncmoe 32
print(f"\nVRAM for experts = 8 layers = {8*E*exp_MB/1e3:.2f} GB = {cache_frac:.0%} of experts")
print(f"{'miss rate':>10} {'static':>8} {'cache+upload':>13} {'cache+CPU':>10}  (tok/s)")
static = 1000/(a+b*32)
for m in [0.15,0.22,0.27,0.33,0.45]:
    misses = m*L*K
    up = 1000/(a + misses*exp_MB/1e3/pcie*1e3)
    cpu = 1000/(a + misses*b/K)
    print(f"{m:>10.0%} {static:>8.1f} {up:>13.1f} {cpu:>10.1f}")
print(f"\nupload cost per expert {exp_MB/pcie:.3f} ms vs CPU cost per expert {b/K:.3f} ms -> CPU is {exp_MB/pcie/(b/K):.1f}x cheaper per miss")
# hybrid split f=p/(p+c) with c effective cpu bw
c = per_layer_MB/b
print(f"hybrid f = p/(p+c) = {pcie/(pcie+c):.2f}")
