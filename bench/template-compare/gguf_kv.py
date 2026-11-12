"""Print selected GGUF metadata from the header only (works on a truncated file). Usage: gguf_kv.py FILE [OUT_TEMPLATE]"""
import struct, sys
f = open(sys.argv[1], "rb")
def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def s(): n = rd("Q"); return f.read(n).decode("utf-8")
SC = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
def val(t):
    if t == 8: return s()
    if t == 9:
        et, n = rd("I"), rd("Q")
        return [val(et) for _ in range(n)]
    return rd(SC[t])
assert f.read(4) == b"GGUF"; ver = rd("I"); nt = rd("Q"); nkv = rd("Q")
kv = {}
for _ in range(nkv):
    k = s(); t = rd("I"); kv[k] = val(t)
for k, v in kv.items():
    if isinstance(v, list): v = f"[{len(v)} items]"
    elif isinstance(v, str) and len(v) > 80: v = f"<str {len(v)} chars>"
    if not k.startswith("tokenizer.ggml.") or "token" not in k: print(f"{k} = {v}")
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(kv["tokenizer.chat_template"])
