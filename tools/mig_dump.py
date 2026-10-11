# -*- coding: utf-8 -*-
"""列出旧包数据库里的观测记录，确认迁移价值（迁移前人工过一眼）。

用法：python tools/mig_dump.py <db路径>
"""
import sqlite3
import sys


def main(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select id, speciesZh, speciesSci, round(confidence,3) conf, "
        "region, address, observedAt, photoPath, confirmed "
        "from observations order by observedAt").fetchall()
    print(f"共 {len(rows)} 条记录\n")
    for r in rows:
        photo = (r["photoPath"] or "")[-38:]
        print(f"#{r['id']:<3} {r['observedAt']}  {r['speciesZh']} ({r['speciesSci']})"
              f"  conf={r['conf']}  确认={r['confirmed']}")
        print(f"      地区={r['region'] or '-'}  地址={r['address'] or '-'}")
        if r["photoPath"]:
            print(f"      照片={r['photoPath']}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))