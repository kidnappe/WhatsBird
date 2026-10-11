r"""离线行政区划数据生成（观鸟记录"点→省市区地址"端上反查用）

数据源：阿里 DataV.GeoAtlas 公开边界接口（免费公开，无需 key）
    https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json
输出：android/app/src/main/assets/geo/admin_county.ndjson
    每行一个县级（或无下属区的市级）面要素：
    {"a":adcode, "p":省, "c":市, "d":区县(null=不设区市), "bb":[minLon,minLat,maxLon,maxLat],
     "g":[[lon0,lat0,lon1,lat1,...], 洞环...]} —— 坐标为整数（度×1e4，≈11m 量化）

流程：国家→省→市→区县三级展开；市层 _full 404/为空（东莞、中山、嘉峪关等不设区地级市）
时把市自身当叶子；RDP 简化 tolerance 默认 0.002°（体积换准确，别超过 0.005°）。

用法（AGENTS.md 规矩：脚本 + bat 同目录）：
    python tools/admin_geo/generate_admin_geo.py [--tolerance 0.002]
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://geo.datav.aliyun.com/areas_v3/bound/{code}_full.json"
UA = {"User-Agent": "ObserveBird-admin-geo/1.0 (offline region index)"}
OUT = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "android", "app", "src", "main", "assets", "geo", "admin_county.ndjson",
)


def fetch(code: str):
    req = urllib.request.Request(BASE.format(code=code), headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # 不设区市：无下级
        raise


# ---------- RDP 简化（纯 stdlib，拓扑不敏感：环独立简化，共边缝隙可忽略——判内用 11m 量化容差足够） ----------

def _rdp(pts, tol):
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        (sx, sy), (ex, ey) = pts[s], pts[e]
        dx, dy = ex - sx, ey - sy
        norm = (dx * dx + dy * dy) ** 0.5 or 1e-12
        far, idx = -1.0, s
        for i in range(s + 1, e):
            dist = abs(dy * pts[i][0] - dx * pts[i][1] + ex * sy - ey * sx) / norm
            if dist > far:
                far, idx = dist, i
        if far > tol:
            keep[idx] = True
            stack.append((s, idx))
            stack.append((idx, e))
    return [p for p, k in zip(pts, keep) if k]


def rings_of(geom):
    """Polygon/MultiPolygon → 环列表（每环 [(lon,lat)...] 首尾未重复）。"""
    gtype = geom["type"]
    polys = geom["coordinates"] if gtype == "MultiPolygon" else [geom["coordinates"]]
    out = []
    for poly in polys:
        for ring in poly:
            pts = [(lon, lat) for lon, lat in ring[:-1]] if len(ring) > 3 else ring
            if len(pts) >= 3:
                out.append(pts)
    return out


def quantize(rings, tol):
    out = []
    for ring in rings:
        sim = _rdp(ring, tol)
        if len(sim) < 3:
            continue
        flat = []
        for lon, lat in sim:
            flat.append(int(round(lon * 10000)))
            flat.append(int(round(lat * 10000)))
        out.append(flat)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.002)
    args = ap.parse_args()

    leaves = []
    nat = fetch("100000")
    provinces = [f for f in nat["features"]
                 if str(f["properties"].get("adcode", "")).isdigit()]
    print(f"省级要素 {len(provinces)} 个")
    t0 = time.time()
    for pf in provinces:
        pp = pf["properties"]
        pname, pcode = pp["name"], pp["adcode"]
        cities = fetch(pcode)
        children = (cities or {}).get("features") or []
        # 直辖市：children 直接是 district
        for cf in children:
            cp = cf["properties"]
            level, cname, ccode = cp.get("level"), cp["name"], cp["adcode"]
            if level == "district":  # 直辖市下辖区
                leaves.append((pname, pname, cname, ccode, cf["geometry"]))
                continue
            subs = fetch(ccode)
            sub_feats = (subs or {}).get("features") or []
            dist_level = [s for s in sub_feats if s["properties"].get("level") == "district"]
            if dist_level:
                for sf in dist_level:
                    leaves.append((pname, cname, sf["properties"]["name"],
                                   sf["properties"]["adcode"], sf["geometry"]))
            else:
                # 不设区地级市（东莞/中山/嘉峪关…）或子层为空：市自身当叶子
                leaves.append((pname, cname, None, ccode, cf["geometry"]))
            time.sleep(0.15)
        time.sleep(0.15)
        print(f"  {pname} ✓ 累计叶子 {len(leaves)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_pts = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for pname, cname, dname, adcode, geom in leaves:
            if not geom:
                continue
            qr = quantize(rings_of(geom), args.tolerance)
            if not qr:
                continue
            flat = [v for ring in qr for v in ring]
            n_pts += len(flat) // 2
            xs, ys = flat[0::2], flat[1::2]
            rec = {
                "a": adcode, "p": pname, "c": cname, "d": dname,
                "bb": [min(xs), min(ys), max(xs), max(ys)],
                "g": qr,
            }
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    size_mb = os.path.getsize(OUT) / 1e6
    print(f"[DONE] 叶子 {len(leaves)} 个 · 顶点 {n_pts} · {size_mb:.1f} MB · "
          f"{time.time() - t0:.0f}s -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    sys.exit(main())
