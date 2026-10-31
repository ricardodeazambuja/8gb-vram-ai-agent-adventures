#!/usr/bin/env python3
"""Compare two GGUFs tensor by tensor and list metadata differences: cmp_gguf.py A.gguf B.gguf"""
import sys

from gguf import GGUFReader

a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ta = {t.name: t for t in a.tensors}
tb = {t.name: t for t in b.tensors}
print("only in A:", sorted(set(ta) - set(tb))[:5], "only in B:", sorted(set(tb) - set(ta))[:5])
diff = [n for n in ta if n in tb and (ta[n].tensor_type != tb[n].tensor_type or ta[n].data.tobytes() != tb[n].data.tobytes())]
print(f"{len(ta)} tensors, {len(diff)} differ:", diff[:12])
for n in diff[:3]:
    print(" ", n, ta[n].tensor_type.name, tb[n].tensor_type.name)
meta = sorted(set(a.fields) ^ set(b.fields))
print("metadata keys in only one:", meta[:12])
