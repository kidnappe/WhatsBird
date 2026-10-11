# 离线行政区划反查（AdminGeoIndex）

> 2026-10-08 上线。给"记录此鸟"的地址字段做无网保底：坐标→"省 市 区县"，
> 端上多边形点内判断，零网络、零 key、零限流。方案讨论正本见本文件；
> 接入点：`CameraViewModel.reverseGeocode()` 降级链第一级。

## 为什么

系统 Geocoder 在无 GMS 的国产 ROM（测试机小米 15 实测）`isPresent()`=false，地址恒空；
在线第三方（Nominatim/BigDataCloud）有共享限流、可达性和"数据发第三方"的隐私让步，
而且野外观鸟经常没信号。结论来自四种免费方案的对比：离线区划是唯一的
"确定 + 免费 + 离线 + 无合规风险"组合，街道级留给 Geocoder 兜底当锦上添花。

## 数据链路（PC 侧，tools/admin_geo/）

```
DataV.GeoAtlas 公开边界 API（阿里，免费无 key）
   国家 100000_full → 34 省 → 每省 _full → 市 → 每市 _full → 区县
        │ 不设区市（东莞/中山/嘉峪关…）_full 404 → 市自身当叶子，d=null
        │ 直辖市（沪京津渝）children 直接是 district
   python 侧 RDP 简化（tolerance 0.002°，≈200m，别超 0.005°）
   坐标量化：度×1e4（≈11m）
        ↓
android/app/src/main/assets/geo/admin_county.ndjson   （实测 7.62MB / 2026-10-08 产物，非本文原写的 2-4MB；不进 git，由本节 bat 重跑生成）
   每行：{"a":adcode,"p":省,"c":市,"d":区县|null,"bb":[minLon,minLat,maxLon,maxLat],
          "g":[[lon,lat,...] 外环+洞环...]}
```

- 一键重跑：`tools\admin_geo\start_generate.bat`（全量 ~400 请求，约 3 分钟；区划调整时换数据）。
- ⚠️ **待决策（10-09 记）**：7.62MB 已占 APK 预算。若要更准（贴县界不误判邻县）需把 RDP tolerance
  从 0.002° 降到 0.001°，体积约翻倍到 ~15MB——先评估 APK 影响再定。
- ⚠️ **坐标系**：DataV 边界是 **GCJ-02 火星坐标**（随高德对齐）。GPS 出的是 WGS-84。
  端上不做数据侧纠偏，改为**查询点前变换 WGS→GCJ**（`AdminGeoIndex.wgs84ToGcj02`，
  eviltransform 公开算法）——数学等价，代码只在端侧一处。

## 查询算法（App 侧 vision/AdminGeoIndex.kt，~200 行）

1. **bbox 预筛**：**2,874** 个量化矩形线性扫（μs 级），不需要空间索引。
   （2026-10-09 实测 `assets/geo/admin_county.ndjson` 行数；本文原写 3400，
    `AdminGeoIndex.kt:21/99` 与 `CameraViewModel.kt:98` 的注释也照抄了 3400，一并作废。）
2. **射线法判内**：对通过预筛的 1~3 个候选，逐环偶奇计数（外环+洞环同规则，
   GeoJSON 环不嵌套所以奇偶翻转成立）；交点比较用 double，防整式符号翻转坑。
3. **最近命中缓存**：观鸟一次 outing 不挪窝，连续记录通常首候选即中。
4. 输出组装：直辖市去重（市名=省名时只播一次），形如"湖北省 武汉市 洪山区"。

- 懒加载 + `preload()`：`CameraViewModel` init 在 IO 线程预热，首次保存不等解析。
- 失败口径：资产缺失/境外/未覆盖 → null → Geocoder 兜底 → 再 null → 经纬度降级展示。

## 验收

- PC 侧快速回归（生成后必跑）：北京站 (39.90,116.48)→"北京市 北京市 东城区"级、
  武汉光谷 (30.51,114.42)→ 洪山区/东湖高新邻近县、东莞 (23.02,113.75)→"广东省 东莞市"。
  用同一 RDP 参数在 python 里跑 point-in-polygon 对样例点自测（生成脚本同款数学）。
- 真机：记录此鸟后日志详情"📍"行应出现区县地址（小米 15 无 Geocoder 后端时
  **只有这条链能给出地址**——它即验收主体）。

## 已知边界

- 粒度到区县（观鸟记录够用）；街道/POI 级在 GMS 机型由 Geocoder 兜底补。
- RDP 0.002° 下紧贴县界的点可能判错邻县（误差量级=简化容差）；接受，
  如要更好精度把 tolerance 调到 0.001 重新生成（体积翻倍）。
- 海域/无人区（西沙等）有名义辖区面但基本不会有观测点，无需特判。
- 数据是 2026-10 快照，区划变更年度换文件。
