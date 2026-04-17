# Phone 视频处理器 v1.0 — 新脚本

## 任务

新增 `phone_ink_processor.py` 支持「手机贴近纸」拍摄 setup。**不改动**
`webcam_ink_processor.py` / `ink_extraction.py` / `xhs_cover.py`（保留 webcam
作为备选管线）。

## 新建 / 复制 / 改造的函数

| 函数 | 来源 | 说明 |
|---|---|---|
| `rotate_video_if_needed` | **新增** | ffmpeg `transpose=1/2/1,1` 预处理，手机横拍时摆正 |
| `detect_paper_phone` | **新增** | 替代 webcam 的 Otsu-based `detect_paper_region`。三路径：full-frame / with-obstruction / center-fallback，基于 `mean` + `std` 的 sanity check |
| `_parse_char_region` | 从 webcam 复制 | `--char-region` 解析 |
| `detect_ink_phone` | 从 webcam **改造** | analysis_mask 直接用 paper['mask']、面积下限 500（vs 100）、呼吸空间 15%（vs 25%）、cluster_kernel 上调、面积上限 30%（vs 15%）。skin_mask / flat_field / `_brush_pass` / `_pencil_pass` 逻辑与 webcam 一致（pencil 的 bg_subtract ∪ 阈值并联） |
| `find_clean_last_frame_phone` | 从 webcam 复制 | 逻辑完全一致，相对排序 + 8 帧中位数合成 |
| `find_best_cover_frame_phone` | 从 webcam 复制 | 逻辑完全一致，全视频采样 + 末尾密采样 + skin_over_char 升序 |
| `process_phone_video` | 从 webcam **改造** | Step 0 加 rotate 预处理、Step 2 换 paper 检测、Step 3 换 ink 检测、裁剪约束基于 paper['mode'] 判断是否收紧。其余（Step 5-11、缩略图、封面）完全一致 |
| `main` CLI | 从 webcam **改造** | 加 `--rotate {none,cw,ccw,180}`，删 `--voice-only` 等一致的参数 |

## Makefile 改动

`org-yt-factory/Makefile` 加入三个 target（并保留所有 webcam 命令不动）：

```makefile
make phone IN=she.mp4 [MEDIUM=pencil] [ROTATE=cw] [TEXT=... | TF=file]
make phone-full IN=she.mp4 CHAR="舍" TITLE="..." [COVER_STYLE=pencil-bold] [...]
make phone-help
```

`phone-full` 复用现有的 `short-import` / `xhs_cover.py` / `short-export`，
只替换「处理脚本」为 `phone_ink_processor.py`。

## she.mp4 端到端测试

**源视频**：`/mnt/c/Users/david/Downloads/she.mp4`，720×1280（不是 spec 说的 4K，
实际上手机压缩到 720p 竖屏）、10.4s、30fps、paper mean=158、std=7.6。

### 关键日志

```
📹 Step 1: 720x1280, 10.4s, 30fps
📄 Step 2: 纸面检测 (sanity check)...
   ✅ 全纸模式: mean=159, std=8 (整画面视为纸)
🔍 Step 3: 检测笔迹 (medium=pencil)...
   ✅ 铅笔笔迹: (136,376) 296x300, 占画面 9.6%
      中心: (284,526), 轮廓数: 1, 介质: pencil
✂️  Step 4: 裁剪 (87,176) 394x700, 放大 2.74x, 目标字占比: ~75%
⏸️  Step 8: 合成池 8 帧 (末尾倒数 1-8), 肤色 0.0%-0.0%
🗣️  Step 9: TTS 27 字 → 8.0s, 语速 +0%, 3 句字幕
🎙️  Step 10: 响度 -25.27 → -14 LUFS
🖼️  封面搜索: 45 帧采样 (粗 16 + 末尾密 30), 45 候选, pool 8 帧 skin 0.0%
   封面源帧 180-284/313, char_ink 5867-6023 px（强信号）
✅ 输出: 1080x1920, 14.4s, 13.5MB
```

## 阈值调整说明（spec 预期 vs 实测）

Spec 写：
```
纸面灰度：180-210 (HDR 白平衡后偏中灰)
mean > 180 and std < 30 → full-frame
```

**实测** `she.mp4` 末帧：
```
mean = 158.8, std = 7.6
```

**调整为**：`mean > 140 and std < 25`。

**依据**：
- 实际 mean=159 远低于 spec 预期 180。手机自动白平衡在近距离拍摄时容易把光源做得偏暗。
- std=7.6 远低于 spec 的 30——纸面**非常**均匀，即便不是纯白。
- 关键信号不是「白」（mean 高），是「均匀」（std 低）——后者决定这帧是不是"纸+少量异物"。
- 放宽 std 阈值到 25 仍排除书写中的 active 帧（std=50-57）和 fallback 误伤。
- mean > 140 排除严重阴影场景（半帧变黑），同时覆盖各种色温下的纸面。

Case 2（with-obstruction）的 `mean > 120` 也一并放宽（spec 是 `> 165`），同样的依据。

## 生成的输出

| 文件 | 状态 | 备注 |
|---|---|---|
| `she.mp4` (Shorts 视频) | ✅ 成功 | 1080×1920、14.4s、字居中清晰、色彩校正把 159 灰度纸面提到 p95=192（近白）、TTS 自然合成 8s、字幕 3 句、淡入淡出正常 |
| `she_thumb.jpg` | ⚠️ 偏暗 | 字可辨但纸面被 `generate_calligraphy_thumbnail` 的 pencil-curves 压到 mean=35（几乎全黑）。**根因见下** |
| `she_cover.jpg` | ⚠️ 偏暗 | 中央裁切块也是暗灰，源于 xhs_cover 的 flat_field 对灰纸无能（中位数<150 触发回退），然后 pencil-bold 强拉伸在暗底上效果有限 |

## 已知问题（未修复，spec 禁止改动相关文件）

**灰纸 paper 在下游 thumb/cover 管线上的级联失败。**

设计假设：`generate_calligraphy_thumbnail`（pencil 分支）和 `xhs_cover.extract_calligraphy`
的 flat_field / pencil-zen 都假设输入纸面 ≥220 灰度。

she.mp4 的**色彩校正后**纸面 p95 只有 192（原始 158 + eq brightness=0.1 contrast=1.5）。
pencil curves `0/0 0.5/0.15 0.7/0.1 1/1` 在 x=192/255=0.75 处落到 y≈0.12，
把「近白」打回「暗灰」。

**实测**：
- 视频末帧 p95=192（肉眼接近白）
- 经 `generate_calligraphy_thumbnail` 后 p95=50（暗灰）
- xhs_cover 日志：`⚠️  封面平场校正异常（纸面中位数 <150），回退原始灰度`
- 对比度 `p95-p1 = 38`，触发 `auto_override_style='pencil-bold'`，但底色已是暗灰

**修复方向**（需要改动 spec 禁止改动的文件，请确认后单独开 PR）：
1. `generate_calligraphy_thumbnail`：在 pencil 分支前检测 `input.paper.p95`，
   ≥220 时用现有 curves，190-220 时改用更温和的 curves 或 eq，<190 时跳过整个 tone 阶段
2. `xhs_cover.extract_calligraphy`：flat_field 阈值 <150 时不简单回退，改用
   "直方图 stretch 到 p99→240"一类的强制归一化

目前的兜底：用户对这种视频可以手动指定 `COVER_STYLE=brush`（brush 不做强拉伸，
只做 0.9x microcontrast），**但在 she.mp4 这种铅笔视频上 brush 会完全看不见字**。
所以没有可用的 CLI workaround——需要代码改动。

**建议给用户的选项**：
- A) 接受当前暗底 thumb/cover（偏「夜晚冥想」风格，仍然可辨识）
- B) 授权修改 `ink_video_processor.py` 和 `xhs_cover.py`，实现 paper-brightness 自适应渲染

## Regression 检查（webcam 老管线）

**未测试**。Git diff 显示 `webcam_ink_processor.py` / `ink_extraction.py` /
`xhs_cover.py` / `ink_video_processor.py` **零改动**，理论上老 webcam pipeline
行为完全一致。如需显式回归，可：

```bash
make webcam-full IN=dao_v2.mp4 CHAR="道" TITLE="道生万物"
```

## 文件清单

| 文件 | 改动 |
|---|---|
| `shorts/phone_ink_processor.py` | **新增** ~730 行 |
| `Makefile` | 新增 `phone` / `phone-full` / `phone-help` 三个 target，保留所有 webcam 命令 |
| `shorts/CHANGES_PHONE.md` | **本文** |
| 其它 | 零改动 |

## 用户使用

```bash
# 最常见：自动判介质、不 rotate
make phone-full IN=she.mp4 CHAR="舍" TITLE="舍得之间" TF=/tmp/she.txt

# 显式铅笔
make phone-full IN=she.mp4 CHAR="舍" TITLE="..." TF=... MEDIUM=pencil

# 手机横拍
make phone-full IN=... ROTATE=cw

# 强拉伸封面（本次测试的实际用法）
make phone-full IN=she.mp4 CHAR="舍" ... MEDIUM=pencil COVER_STYLE=pencil-bold
```
