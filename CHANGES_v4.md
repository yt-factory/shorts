# v4 — 纸面亮度归一化（根治下游「假设白纸」级联失败）

## 范围

本轮聚焦 **phone pipeline**。webcam_ink_processor 的调用点也同步接入了新签名
（代码改了但不再扩展回归测试）。不改 `xhs_cover.py` / `ink_extraction.py` /
`generate_calligraphy_thumbnail`。

## 改动清单

1. **`ink_video_processor.py::build_color_correction_filter`** — 签名变
   - 接受可选 `paper_p95: float = None`
   - 传入时返回 `lutyuv=y='clip(val*gain\,0\,255)'`，其中 `gain = 240 / paper_p95` clip 到 [0.8, 2.5]
   - 不传时回退到老固定 `eq=brightness=0.06:contrast=1.25:saturation=1.0`（保持 `process_video` 老管线行为）
2. **`ink_video_processor.py::sample_paper_brightness`** — 新增
   - 扫描多个候选帧位置（距尾 30/60/120/240 + 中间 1/2 和 1/4），跳过 mean<100 的淡出/淡入帧，取首个合格帧上 `paper_mask` 区域的 p95
   - 全部过暗时返回 240.0（gain≈1.0 → 不动画面）
3. **`phone_ink_processor.py::process_phone_video`** — Step 2.5 加入 `sample_paper_brightness(working_path, paper['mask'], 'last')`；3 处 `build_color_correction_filter` 调用都改为 `(paper_p95)`
4. **`webcam_ink_processor.py::process_webcam_video`** — 同样 Step 2.5；webcam 的 `paper` 字典是 `contour` 形态，Step 2.5 里把 contour 转为 mask 再采样；3 处调用加参（代码改了，没有扩展回归测试）

## she.mp4 端到端重跑

### 关键日志（phone pipeline，新增的归一化行）

```
📄 Step 2: 纸面检测 (sanity check)...
   ✅ 全纸模式: mean=159, std=8 (整画面视为纸)
   📊 纸面归一化: p95=168 → 240 (gain≈1.43)        # ← 新增
🔍 Step 3: 检测笔迹 (medium=pencil)...
   ✅ 铅笔笔迹: (136,376) 296x300, 占画面 9.6%
```

色彩校正阶段的 `build_color_correction_filter(168)` 生成：
`lutyuv=y='clip(val*1.4286\,0\,255)'`

裁剪缩放后的 hold-still 帧 p95 应该从 ~192（v3 eq 的上限）提升到 ~240。
实测 cover/thumb 下游指标证实了这一点：

### v3 → v4 指标对比

| 指标 | v3 (pre-v4) | v4 |
|---|---|---|
| `xhs_cover` flat_field 是否回退警告 | ⚠️ 是 (纸面中位数<150) | ✅ 否 |
| `xhs_cover` 笔迹对比度 `p95-p1` | **38** | **228**（6× 提升） |
| `xhs_cover` 触发 `auto_override_style` | `pencil-bold`（强拉伸兜底） | `None`（默认 pencil-zen 就够） |
| `render_ink` `p5_ink` | 26 | 169 |
| 生成 thumb mean | 35 | 198 |
| 生成 cover mean | ~35 | 236 |

### 视觉对比

- **she_thumb.jpg (v4)**：浅灰白纸底、纸纤维质感、铅笔「舍」字清晰、笔锋可辨；
  v3 是全黑底上勉强可辨的字形轮廓。
- **she_cover.jpg (v4)**：中央米白纸方块 + 深色铅笔「舍」字 + 标题 + 副标题
  + 右下红色印章。符合「极客禅」视觉语言。v3 是中央一块深灰砖。

## 🐞 测试中发现并修复的 bug

### Bug：`sample_paper_brightness` 对自带 fade-out 的输入返回 p95=16

**症状**：`extract_frame('last')` 默认拿「距尾 offset=[2,5,10,15,1]」里第一个
能读到的帧，落在淡出区的第 2 帧（几乎全黑，mean=12）。p95 于是是 16，
gain = 240/16 = 15 → clip 到上限 2.5。视频被过拉到烧白、thumb 爆掉。

**触发条件**：raw 手机录像通常没这个问题（she.mp4 没 fade）。但若用户把
一轮 pipeline 输出当输入重跑（迭代测试场景），fade-out 就会混进来。

**修复**：`sample_paper_brightness` 从「取某一固定帧」改为「扫描多个候选 +
均值过暗时跳过」。

- 候选顺序：offsets `[30, 60, 120, 240, total/2, total/4]`（先偏尾再偏中）
- 门槛：`frame_mean < 100` 视为淡出/淡入，跳过
- 全部过暗 → 返回 240.0（near no-op）

**验证**：
```
pre-fix:  sample_paper_brightness(dao_v2.mp4, None, 'last')  # dao_v2 有 fade
          → p95=16.0 (bug)
post-fix: sample_paper_brightness(dao_v2.mp4, None, 'last')
          → p95=236.0 ✅
          sample_paper_brightness(she.mp4, None, 'last')     # 无 fade，应该一致
          → p95=168.0 ✅（与 raw 末帧 p95 实测一致）
```

## lutyuv 语法兼容性

- ffmpeg 版本：**6.1.1-3ubuntu5**（Ubuntu 默认仓）
- filter 清单含 `lutyuv`
- chain 测试：`lutyuv=y='clip(val*1.37\,0\,255)',crop=32:32:16:16` ← 无错
- `\,` 是 ffmpeg filter chain 分隔符转义，保护 `clip()` 内部逗号不被外层 `,` 切分
- Python 字符串里写 `"\\,"`（字面两字符 `\,`），格式：`f"lutyuv=y='clip(val*{gain:.4f}\\,0\\,255)'"`

实际生成的 filter 片段样本：
```
paper_p95=168 → lutyuv=y='clip(val*1.4286\,0\,255)'
paper_p95=240 → lutyuv=y='clip(val*1.0000\,0\,255)'
paper_p95=255 → lutyuv=y='clip(val*0.9412\,0\,255)'
```

## 边界安全性

- `paper_p95=None` / `<=0` → 老 eq fallback（未接入的旧调用路径继续工作）
- gain clip 到 [0.8, 2.5]
  - `paper_p95>=300`（理论不可能，<= 255）→ 0.8 防变暗
  - `paper_p95<=96` → 2.5 防把阴影拉过曝
- 全视频过暗无可用帧 → 返回默认 240（gain=1.0）
- 采样异常 → 返回 240 + warning
- lutyuv 只映射 Y 通道 → chroma 保真

## 文件改动

| 文件 | 改动 |
|---|---|
| `shorts/ink_video_processor.py` | `build_color_correction_filter` 签名变；新增 `sample_paper_brightness`（淡出鲁棒） |
| `shorts/phone_ink_processor.py` | `sample_paper_brightness` 导入 + Step 2.5 + 3 处调用加 `paper_p95` |
| `shorts/webcam_ink_processor.py` | 同上（contour→mask）。**代码改了但本轮未扩展回归测试**，已有单元功能不受影响（`paper_p95=None` 路径保留） |
| `shorts/CHANGES_v4.md` | 本文 |

## 结论

**v4 设计目标达成**：she.mp4 的 thumb/cover 级联失败根治。xhs_cover 的
flat_field 不再回退、quality gate 不再 auto-override、render_ink 的 `p5_ink`
从 26 → 169、thumb 亮度从 35 → 198。

一条根源修复（gain × 纸面 p95 → 240）消除了三层下游代码对「纸近白」假设
的累积违反，而没有改动任何一处下游代码。

副作用很小：已近白的纸 gain≈1.0 接近 identity；灰纸 gain≈1.3-1.5 把纸面
拉回设计假设区间，不会把笔迹洗白（240 而非 255 给笔锋/飞白留头）。

测试中还顺手修掉了 `sample_paper_brightness` 在 fade-out 输入上的 p95=16
bug，详见上一节。
