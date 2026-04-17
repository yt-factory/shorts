# v5 — 审美精修：手机管线进入"品牌 ready"

## 目标

v4 功能已通但视觉风格像「碑刻拓片」。v5 的六项精修把封面调整到「极客禅」调性：
极致留白、干净纸感、柔光边缘、单色系、字就是主角。

聚焦 phone pipeline + xhs_cover。webcam 受 P2 间接受益（共用
`build_color_correction_filter`）。

## 实施顺序与结果

按视觉影响力从大到小（P2→P1→P5→P4→P3→P6），she.mp4 重跑指标：

| 指标 | v4 | v5 | 变化 |
|---|---|---|---|
| xhs_cover 笔迹对比度 | 228 | 227 | ≈ 持平（hqdn3d 没伤字） |
| render_ink `p5_ink` | 169 | 171 | ≈ 持平 |
| 字缩放尺寸 | 325×581 | **204×365** | ↓ 37% |
| 字占画布宽度 | 26% | **16%** | ↓ 10 pct |
| 字中心 y 位置 | ~43% | **37.95%** | 精准落在 38% 黄金分割 |
| cover 整体 mean | 236 | 238 | ≈ 持平，表示留白比例更大 |
| cover 整体 std | 更高 | **14.8** | 更均匀（少色块波动） |

## 六项修复详述

### P2 — 纸面去噪（`ink_video_processor.py::build_color_correction_filter`）

v4 的 `gain=1.43` 把 she.mp4 纸面从 168 拉到 240，同时也把纸面纤维和光照不均
一起放大，输出像"水泥墙"。在 lutyuv 之后、下游锐化之前接 `hqdn3d=4:3:6:4.5`：

- `luma_spatial=4`，`chroma_spatial=3` — 空间低频纹理平掉
- `luma_tmp=6`，`chroma_tmp=4.5` — 静态纸面利用时域冗余额外平滑

```python
return (
    f"lutyuv=y='clip(val*{gain:.4f}\\,0\\,255)',"
    f"hqdn3d=4:3:6:4.5"
)
```

视频管线时域分量生效明显。thumb 路径只处理一张中位数合成帧、时域分量
退化为空间，所以 thumb 上纸面纹理仍残留一些（std=38）；后续如果需要
更干净的 thumb 可在 `generate_calligraphy_thumbnail` 里再加一遍空间 hqdn3d。
本轮未动该函数。

### P1 — 去字周白色光晕（`xhs_cover.py::extract_calligraphy`）

旧版 feather 4% + 线性 ramp 让 alpha 在字周 12-20 px 内线性过渡，
composite 阶段这条 ramp 被当成"字的一部分"渲染，形成外发光。

```python
# feather: 4% → 1%；ramp: linear → smoothstep (y = x² (3-2x))
feather = max(2, int(min(ch_w, ch_h) * 0.01))
ramp = np.linspace(0, 1, feather, dtype=np.float32)
ramp_ss = ramp * ramp * (3.0 - 2.0 * ramp)
```

smoothstep 两端导数为 0，过渡柔但中段陡，宽度只有原来的 1/4，
消除了 ramp 尾部的"字边缘轻 alpha 带"。

### P5 — 缩小字占比（`xhs_cover.py::generate_cover`）

单字 0.48 → **0.28**，多字 0.70 → **0.45**。`max_h` 同步下调到 0.22。

小红书高赞书法封面的字普遍占 15-25%。she.mp4 从 26% 降到 **16%**，
留白从 74% 增加到 84%。视觉上从"课本插图"变"禅意海报"。

### P4 — 黄金分割锚点（`xhs_cover.py::generate_cover`）

重构锚点逻辑：**从 `DIVIDER_Y_RATIO` 常量改为 `CHAR_CENTER_Y_RATIO=0.38`**，
divider_y 从 target_h 反推（不再是常量）。

**语义变化**：旧版先定字底边（52%-3%=49%），字中心随 target_h 漂移；
对方字中心在 42%、对扁字中心在 47%。新版字中心固定在 38%，不同 aspect
的字视觉重心一致。

```python
# v5
char_center_y = int(cover_height * 0.38)
paste_y = char_center_y - target_h // 2  # 字中心锚住
char_bottom = paste_y + target_h          # 字底派生
divider_y = char_bottom + char_gap        # 圆点派生
```

she.mp4 实测 paste_y=448, target_h=365 → char_center_y = 630.5 = **37.95% × 1660**，
精准落在设计点。

### P3 — 纸块硬切 → alpha-blend paste（**与 spec 方案差距最大**）

这是 v5 六项里离 spec 最远的一项。spec 原意是"给显式纸块加 2.5% 羽化外边缘"，
但代码里没有显式纸块——可见的"纸块"是 paste 逻辑的副产品：

```python
# 旧版把 composite（bbox 内的完整复合图）硬覆盖到 canvas 上
canvas_gray[py1:py2, px1:px2] = gray_arr[sy1:sy2, sx1:sx2]
```

`canvas_gray` 在 paste 之前已经加了 `GaussianBlur(sigma=40)` 的纸面噪声纹理
（`canvas_gray = canvas_gray * 0.94 + noise * 0.06`）。paste 把这块区域的
纹理完全擦掉，换成 composite_gray —— composite_gray 中 alpha=0 的区域是
纯 bg_gray_val=240（无纹理）。结果是：带纹理的米白底 vs 带无纹理的纸白区，
形成可见矩形的"纸块"边缘。

**alpha-blend 的具体实现**：不引入显式纸块，而是让 composite 以
smoothstep 方式"融入"已纹理化的 canvas：

```python
bh, bw = (py2 - py1), (px2 - px1)
ramp_px = max(2, int(min(bh, bw) * 0.025))   # 2.5%
spatial_blend = np.ones((bh, bw), dtype=np.float64)
if 0 < ramp_px * 2 < min(bh, bw):
    r = np.linspace(0, 1, ramp_px, dtype=np.float64)
    r_ss = r * r * (3.0 - 2.0 * r)  # smoothstep
    # 上、下、左、右四边 ramp_px 宽度内乘以 smoothstep ramp
    spatial_blend[:ramp_px, :] *= r_ss[:, None]
    spatial_blend[-ramp_px:, :] *= r_ss[::-1, None]
    spatial_blend[:, :ramp_px] *= r_ss[None, :]
    spatial_blend[:, -ramp_px:] *= r_ss[None, ::-1]

composite_in = gray_arr[sy1:sy2, sx1:sx2]
canvas_gray[py1:py2, px1:px2] = (
    composite_in * spatial_blend
    + canvas_gray[py1:py2, px1:px2] * (1.0 - spatial_blend)
)
```

**关键设计点**：
- `spatial_blend` 是**空间** ramp（bbox 位置上的距边距离），不是**字形** alpha 的再 feather
- 中心区 `spatial_blend=1` → composite 完整保留（字干净呈现）
- 边缘 2.5% → `spatial_blend` 从 1 滑到 0 → canvas 的纹理从无到有透出来
- smoothstep (`3x²-2x³`) 在 0 和 1 处导数为 0，视觉上"边界自然消失"
- 左右上下四边相乘保证角点过渡最快（而非突兀直角）

**与 spec 「给纸块加羽化外边缘」的等价性**：
视觉效果达成了 spec 目标（边缘柔化、纸块感保留）。但实现路径完全不同——
spec 假设代码里有纸块元素可以贴上软 alpha 蒙版，实际是 paste 本身的
过渡策略变成 smoothstep 混合。

### P6 — 标题字体：霞鹜文楷 Bold（`xhs_cover.py::load_title_font`）

系统已装 `/usr/share/fonts/truetype/lxgw-wenkai/LXGWWenKai-Bold.ttf`，
没有下载需要。实现为优先级链：

```python
TITLE_FONT_CANDIDATES = [
    ('/usr/share/fonts/truetype/lxgw-wenkai/LXGWWenKai-Bold.ttf', 0),
    ('/usr/share/fonts/opentype/source-han-serif/SourceHanSerifSC-Heavy.otf', 0),
    (SERIF_BOLD, CJK_SC_INDEX),
    (SERIF_FONT, CJK_SC_INDEX),
    (WIN_SIMKAI, 0),
]
```

首次命中非首选时一次性打印提示（`sudo apt install fonts-lxgw-wenkai`），
此后静默。`load_default()` 作为最后兜底（不会实际触达）。

视觉对比：she.mp4 的"舍得之间"从 NotoSerifCJK Bold 的均匀衬线改为
霞鹜文楷 Bold 的手写楷书感。笔画粗细有变化、字形不方正、和书法
内容气质呼应。

## 视觉验收（肉眼）

| 项 | 状态 | 备注 |
|---|---|---|
| 纸面干净，无水泥墙感 | ⚠️ 部分 | 视频帧（时域去噪）干净；thumb 纸面仍有颗粒感（单帧 hqdn3d 效果有限），本轮未加 thumb 专用空间去噪 |
| 字边缘无白色光晕 | ✅ | smoothstep feather 1% 后边缘锐利 |
| 纸块无硬切 | ✅ | 2.5% smoothstep spatial-blend，视觉从"硬方块"变"柔融入" |
| 字占比 15-25% | ✅ | 16%，落在理想区间 |
| 视觉重心合理 | ✅ | 字中心 37.95%，几乎就是黄金分割 |
| 标题手写感 | ✅ | 霞鹜文楷 Bold |
| 整体观感「克制、现代、禅意」 | ✅ | 可直接发小红书 |

## 文件改动

| 文件 | 改动 |
|---|---|
| `shorts/ink_video_processor.py` | `build_color_correction_filter` 加 hqdn3d |
| `shorts/xhs_cover.py` | P1 feather、P3 alpha-blend、P4 char_center 锚点、P5 比例收紧、P6 字体优先级 + 提示 |
| `shorts/CHANGES_v5.md` | 本文 |
| **没动** | `webcam_ink_processor.py`, `ink_extraction.py`, `generate_calligraphy_thumbnail`, `phone_ink_processor.py` |

## 遗留的精修空间

1. **thumb 残余纸面颗粒**：thumb 路径用中位数合成单帧，hqdn3d 时域分量失效。
   如要完全无颗粒，可在 `generate_calligraphy_thumbnail` 里的 `vf_parts` 加
   一遍空间 hqdn3d（`hqdn3d=4:3:0:0`）。本轮未做（spec 明令不动 thumb 函数）。

2. **cover 字内部的纸面噪声透出**：`extract_calligraphy` 的 alpha 在 25-60
   之间的像素是真实笔迹的半透明部分，但也包括部分纸面噪声误判。`blend` 在
   这个区间把 cal_gray（带纸面颗粒）和 bg_gray_val 做混合，让这些噪点半
   透明地出现在 cover 的字纸块内。如要更干净，可提高 `alpha[alpha<X]=0`
   的阈值 X（当前 25）——但会伤真实笔画的飞白。没动。

3. **字体 fallback 视觉一致性**：如果用户系统没装霞鹜文楷，回退 Noto
   的 fallback 会让封面"商务感"回归。测试机都有霞鹜文楷；真实发布前
   确认目标环境字体。

## 结论

v5 达成设计目标：she.mp4 的 cover/thumb 从"拓片感"转向"小红书禅意封面"。
字占比从 26% 降到 16%、中心精准落在 38% 黄金分割、字周光晕消失、纸块
边缘从硬切变柔融、标题字体从 PPT 衬线变手写楷书。

P3 的 alpha-blend 实现与 spec 原方案差距最大：不引入显式纸块元素，
而是把 paste 从硬覆盖改为 smoothstep spatial blend——同样的视觉效果，
但实现上更贴近现有架构。

下一步交给运营（选题、节奏、模板固化）。
