# v2 — 架构性修复：封面源帧搜索、质量门槛、自适应渲染

## 改动清单

### 1. `find_clean_last_frame_webcam` — 相对排序

**位置**：`webcam_ink_processor.py`

去掉绝对 `skin<2%` 硬门槛，改为对亮度合格帧按 `skin%` 升序取前 8。
这样即便所有帧都有手，至少合成出的视频静帧是「手位置最少的 8 帧中位数」，
不再直接 fallback 到单帧。

### 2. `find_best_cover_frame()` — 全视频搜最佳封面源帧

**新函数**，`webcam_ink_processor.py`

- 粗采样 `sample_fps=1.5` + **末尾 30 帧密采样**（书写完成的抬手瞬间常在末尾 <1s）
- **度量翻转**：最初用 `ink_area` 最大过滤，但书写中的手/臂阴影会在 bg_subtract
  产生假 ink 信号（she.mp4 实测：有手 17%+、ink=94k；无手 0%、ink=2.3k）。
  改为以 `skin_over_char_bbox` 为主排序（直接对应「手不挡字」），
  `char_ink ≥ 500 px` 作为「字在这里」的存在性校验。
- 取前 8 帧中位数合成。<3 帧返回 None，调用方 fallback。
- 需要 `ink_bbox` 参数——由 `detect_ink_webcam` 的结果直接传入。

**架构意义**：视频 bbox 用 `find_clean_last_frame_webcam`（观众看过程，单帧脏一点没事），
封面 thumb 用 `find_best_cover_frame`（静态图要求高）。两个目标彻底解耦。

### 3. xhs_cover 质量门槛 + 自动风格覆盖

**位置**：`xhs_cover.py::extract_calligraphy`

在 Step 4（裁出 `flat_cropped` 后）插入对比度检查：

- **度量修正**：最初用 `p95 - p5`，但稀疏淡笔迹在白纸上时 p5 仍接近纸白，
  contrast 虚低。改用 `p95 - p1`——p1 捕捉最暗 1% 像素，只要字存在就能反映最深处。
- `contrast < 25` → 抛 `ValueError`，拒绝生成空白封面。错误消息提示重录/用 `--char-region`。
- `contrast < 55` 且 `detected_medium == 'pencil'` → 返回 `auto_override_style='pencil-bold'`，
  强拉伸让淡笔迹更可见。
- 返回签名扩展为 `(Image, detected_medium, auto_override_style)`。

`generate_cover` 根据优先级解析风格：**用户 --cover-style > auto_override > medium 默认**。

### 4. `render_ink` 自适应 `target_dark`

**位置**：`xhs_cover.py::render_ink`

```python
if cover_style == 'pencil-bold':   target_dark = 25   # 始终强拉伸
elif cover_style == 'pencil-zen':  # 自适应
    if darkest < 100:  target_dark = 55   # 字够暗
    elif darkest < 160: target_dark = 40  # 中等
    else:              target_dark = 25   # 极淡 → 强拉伸
```

输入质量参差时输出视觉墨度保持一致。

---

## `she.mp4` 实测（新版：白纸背景）

源文件：`/mnt/c/Users/david/Downloads/she.mp4`，25.8s @ 30fps，4K，白纸（无印刷线）。

### 关键日志

```
📄 Step 2: 检测白纸...
   ✅ 白纸: (190,863) 3249x1271, 占画面 49.8%

🔍 Step 3: 检测笔迹 (medium=pencil)...
   ✅ 铅笔笔迹: (2172,1311) 313x284, 占画面 1.1%

⏸️  Step 8: 静止画面 (4s)...
      合成池: 8 帧 (末尾倒数 10-38), 肤色范围 8.2%-8.4%

🖼️  搜索最佳封面源帧（全视频）...
      全视频采样: 71 帧 (粗采样 49 + 末尾密采样 30; 总 193 帧, 25.8s @ 1.5 fps)
      有字候选 53 帧 (char_ink≥500), 取 skin_over_char 最低前 8
      封面合成池: 8 帧 (帧号 140-165/193, 字 bbox 内 skin 0.0%-0.0%, char_ink 803-1117 px)

   笔迹对比度: p95=255 - p1=183 = 72.0
   渲染风格: pencil-zen (detected=pencil, auto_override=None)
```

### 关键进展

1. **封面合成池全部是 0% 肤色遮挡**：找到了视频末尾写完抬手的 8 帧干净瞬间
   （帧号 140-165/193，对应 4.7-5.5s 末尾位置）。旧管线因为 skin<2% 绝对门槛
   完全错过这些帧。
2. **BBOX 精确锁定全字**：313×284，覆盖整个「舍」字。依赖 `_pencil_pass` 内的
   新改动 `bg_subtract ∪ threshold<200` 并联（单独的 bg_subtract 在纯白纸上
   只抓到 67/879 = 7.6% 的字素）。
3. **char_ink 合理**：8 帧每帧 800-1100 px 是真实字素（p95-p1=72 contrast）。
4. **质量门槛正确工作**：在首次测试中 `contrast=22` 触发 ValueError 阻止生成空封面；
   度量修正后 `contrast=72` 通过正常流程。

### 全视频搜索开销

| 阶段 | 用时 |
|------|------|
| find_clean_last_frame_webcam（末尾 90 帧 Pass 1+2） | ~2s |
| **find_best_cover_frame（71 帧全彩读、flat_field、bg_subtract、median）** | **~8s** |
| 整条 pipeline (25.8s 4K 视频) | 10-13 分钟（主力耗时是 ffmpeg 的 4K→1080p 处理，非搜索） |

全视频搜索增加约 8s（~1% 整体 pipeline 用时），可接受。

### Cover 视觉对比

**旧版（she.mp4 v1 有竖格线版本）**：手的剪影 + 竖格残影，「舍」字不可见。
**v2（新版白纸 + 新管线）**：淡铅笔笔迹残片可见，位置居中，没有手影。
但字仍然较淡——因为 thumb 生成阶段的 `eq=brightness=0.10:contrast=1.5` 对淡
铅笔过度提亮纸面，洗掉了字。render_ink 的 pencil-zen 动态范围拉伸能部分挽回。

### 剩余问题（未修复，影响有限）

**`generate_calligraphy_thumbnail` 的 eq 参数对毛笔最优、对淡铅笔伤字。**

- 毛笔：墨色 <50，`brightness=0.10 + contrast=1.5` 后仍有强反差
- 淡铅笔：字 150-200、纸 220+，相同 eq 后字被压近白色

尝试过 `brightness=0, contrast=1.8`，结果更差（把字进一步推向纸面白）。
根本原因：eq 对「稀疏+低对比」的淡铅笔样本做不了很多事——
input 动态范围本来就窄，任何正参数都会压缩它。

正确解决方向（后续单独 PR）：
- 对 pencil 的 thumb 用 `curves=all='0/0 0.7/0.1 1/1'` 之类的非线性映射
  （specifically 把 180 灰度推到 25-50）
- 或完全跳过 thumb 的 eq，把增强交给 xhs_cover 的 render_ink

目前的质量门槛（`auto_override_style='pencil-bold'`）是有效的兜底：
即便 thumb 洗白后 contrast 落到 ~25-55 区间，pencil-bold 的强拉伸仍能
让封面可见。

### 关于「she.mp4 是否需要重录」

**基于 v2 实测：不需要重录，但效果仍不理想。**

- 视频**有**干净瞬间（帧 140-165 全部 skin_over_char=0%），新管线找到了
- char_ink ~1000 px 证明字**真的**写在那里，不是空白
- 但字本身极淡（p1≈183 — 铅笔最暗处灰度 183/255 = 72% 亮度）
- 这种淡度在任何输出管线里都很难做出「视觉冲击力强」的封面

**建议给用户的反馈**：
- 当前视频可以生成封面，但字会偏淡。想要更浓的字：深色铅笔 + 更大压力。
- 录制行为已经正确（末尾有抬手静帧，新管线正确捕获）。
- 如果后续想改善淡铅笔的 thumb 链，是另一个工作项。

## 文件变更清单

| 文件 | 改动 |
|------|------|
| `webcam_ink_processor.py` | `find_clean_last_frame_webcam` 去绝对阈值；新增 `find_best_cover_frame`；`process_webcam_video` 的 thumb 路径改用新函数；`_pencil_pass` 加简单阈值并联 bg_subtract（修复白纸上 bg_subtract 欠灵敏） |
| `xhs_cover.py` | `extract_calligraphy` 加质量门槛 + `p95-p1` 度量；返回 `auto_override_style`；`generate_cover` 优先级解析 cover_style；`render_ink` 自适应 target_dark |
| `ink_video_processor.py` | （保留原 eq 参数，见剩余问题） |
| `Makefile` | （v1 已加 `COVER_STYLE`，v2 无改动） |
