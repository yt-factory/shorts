# CHANGES: xhs_cover_pencil.py — 铅笔专用封面渲染器

## 最终方案：sigmoid 背景替换（方向 C）

"舍"字以原始铅笔灰度呈现（非黑色），笔画深浅完全由灰度值表达——重按处深灰、
石墨反光处浅灰。纸面被 sigmoid 过渡替换为精确的 BG_COLOR (245,240,235)，
无矩形方块、无纸面纹理残留。

## 关键设计决策

### 1. 输入源：raw cover_frame（非 thumb）

`generate_calligraphy_thumbnail` 的 pencil-curves 会把铅笔灰度（100-180）
压到近黑（0-80），导致 cover 失去铅笔感。

修复：`phone_ink_processor.py` 在生成 thumb 之前，把 `cover_processed`
（crop+scale+color+sharpen，未经 pencil-curves）另存为 `xxx_cover_frame.png`。
Makefile 传给 `xhs_cover_pencil.py` 的是 `cover_frame.png`，不是 `thumb.jpg`。

### 2. 渲染：sigmoid 背景替换 + 压暗，无 alpha

```
旧方案（多轮迭代后放弃）          最终方案
─────────────────────────       ─────────────────────────
RGB = 统一 ink_color (45,42,40)  RGB = 原始灰度（保留铅笔质感）
alpha 表浓淡 → 石墨反光变透明     无 alpha → 石墨反光正确显示为浅灰
render_ink / CLAHE / 暖色偏移     sigmoid + *0.8 压暗，无其他处理
flat_field_correct → 近乎 no-op   不依赖 flat_field
```

### 3. 纸面清理三层策略

```
sigmoid       → 纸面推向 BG_COLOR（但非精确，差 3-15 灰度）
灰度 snap     → result_mean > bg_mean-45 的像素 snap 到精确 BG_COLOR
形态学开运算   → 5×5 椭圆核去除 snap 后残余的孤立灰点
边缘羽化       → bbox 边缘 8% 宽度向 BG_COLOR 渐变，消除方块边界
```

### 4. 连通域空间距离过滤

面积 <200 删除。面积 200-800 检查到最大连通域的距离——近的保留（字的细笔画），
远的删除（纸面噪点）。面积 >800 无条件保留。

解决了面积阈值的两难：200 保留"口"的碎片，800 去杂点——空间距离让两者兼得。

## 调参历程

| 轮次 | darkening | center | min_cc | snap | 结果 |
|------|-----------|--------|--------|------|------|
| v1 (alpha+ink_color) | N/A | N/A | 200 | N/A | 灰色方块（flat_field no-op） |
| v2 (alpha+ink_color) | N/A | N/A | 200 | alpha_floor | 散点噪声云 |
| v3 (ink_weight+alpha) | N/A | N/A | 200 | N/A | 空心描边（石墨反光） |
| v4 (sigmoid, thumb) | 0.8 | p95-15 | 200 | N/A | 字太黑（输入源错误） |
| v5 (sigmoid, raw) | 0.8 | p95-15 | 200 | N/A | 铅笔感对，稍淡+杂点 |
| v6 | 0.65 | p95-8 | 800 | N/A | "口"被截断 |
| v7 | 0.70 | p95-15 | 200+距离 | N/A | 字完整，有方块 |
| v8 | 0.80 | p95-15 | 200+距离 | bg-30 | 方块消失，有灰点 |
| **v9 (final)** | **0.80** | **p95-15** | **200+距离** | **bg-45+morph** | **干净** |

## 文件变更

| 文件 | 动作 | 说明 |
|------|------|------|
| `xhs_cover_pencil.py` | 新建 | 铅笔专用封面，sigmoid 背景替换 |
| `phone_ink_processor.py` | 改动 | 保存 `xxx_cover_frame.png` 供封面使用 |
| `Makefile` | 改动 | phone-full 路由 + pencil-cover target + cover_frame 导出 |
| `CHANGES_PENCIL_COVER.md` | 新建 | 本文件 |
| `xhs_cover.py` | 不动 | 毛笔版保持现状 |
| `webcam_ink_processor.py` | 不动 | webcam 走毛笔版不受影响 |
| `ink_extraction.py` | 不动 | 仅 import remove_ruled_lines |
