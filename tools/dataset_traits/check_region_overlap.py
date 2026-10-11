import csv, os
from collections import defaultdict
base = r"E:\code\code tools\data"
regs = ["huabei","huazhong","huanan_sw","huadong","dongbei"]
occ = {}
for r in regs:
    p = os.path.join(base, r, "dataset.csv")
    if not os.path.exists(p): continue
    s = set()
    with open(p, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            k = (row.get("occurrence_key") or "").strip()
            if k: s.add(k)
    occ[r] = s
    print(f"{r:<10s} {len(s)} 个唯一 occurrence_key")
print()
names = list(occ)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = names[i], names[j]
        n = len(occ[a] & occ[b])
        if n: print(f"重叠 {a} ∩ {b} = {n} 条")
