# Pencil 识别与封面生成重构 — 改动说明

## 背景

铅笔在纸上写字的信噪比远低于毛笔（笔迹对纸面灰度差仅 30–60，而毛笔 > 150）。
旧的 brush/pencil 双管线存在四类问题：

1. `_brush_pass` 在 area>100 即判定成功 → 手指阴影能凑面积、骗走 auto 分支。
2. `adaptiveThreshold(C=15)` 对铅笔笔锋/飞白过严 → 笔画断裂。
3. xhs_cover 的 alpha 直接取自原始 cropped → 纸面渐变阴影被渲到封面上。
4. 去线逻辑的 `morphologyEx OPEN` 核长 `min(h,w)/30 ≈ 36px` 小于汉字最长笔画，
   在白纸场景上偷偷伤字。

## 改动

### 新增：`ink_extraction.py`（共享模块）

被 webcam_ink_processor 和 xhs_cover 共用的图像处理基础：

- **`flat_field_correct(gray)`** — 用大核灰度膨胀估算光照场，原图/光照场得到相对
  反射率。纸面归一化接近 255，笔迹保持相对暗度。之后任何全局阈值都可在统一基底上
  稳定工作。
- **`background_subtract_mask(gray)`** — 高斯降噪 + 大核灰度膨胀估计干净纸面 +
  absdiff + **Otsu 阈值（带 min_threshold=8 底板）**。底板很关键：空白帧上 diff
  全是近零噪声，Otsu 会在噪声带内瞎选、产生满屏假笔迹。
- **`classify_medium(gray, paper_mask)`** — 直方图预分类：看纸面内最暗 1% 像素的
  中位数。<55 → brush，≥200 → empty，其他 → pencil。paper_mask 会先 erode 几像素
  避开纸边阴影污染最暗统计。**替代了原先"先 brush 试、不行再 pencil"的脆弱
  attempt-fallback 循环**。
- **`remove_ruled_lines(mask)`** — 两层去线设计：
  - 第一层（廉价，<5ms）：列/行投影扫描，检测"窄(≤6px)、高(>40% 长度)、间距规律
    (std/mean<0.5)"的尖峰。
  - 第二层：检测到线 → 只在每根线 ±3px 的窄带上减；检测不到 → 原样返回。
  - fallback：投影不确定时用 `MORPH_OPEN` 但 `line_len = min(h,w)/4`
    （比汉字最长笔画安全地长）。
  - 横线方向默认开（汉字横画有提按、低误检风险），竖线方向默认 `'auto'`。
  - **白纸场景零成本、零误伤**。

### 修改：`webcam_ink_processor.py`

- `_brush_pass` 现在在 `flat_gray`（平场校正后）上做全局阈值 80；成功判定改为
  `area > 1000 AND 真墨色（contour 内中位数 < 60）`。
- `_pencil_pass` 改为 `flat_gray → background_subtract_mask + Otsu(带底板) →
  remove_ruled_lines('auto')`。保留了 cluster_kernel + `_block_score` aspect² 惩罚
  的聚类聚焦步骤。
- `detect_ink_webcam` 的 auto 模式改用 `classify_medium` 直接判定，废弃 attempt-
  fallback 循环。新增 `'empty'` 分类早退（画面近乎空白时报错而非伪检测）。
- 平场校正失败时自动回退到原始 gray（健康检查：若 flat_gray 纸面中位数 < 150
  视为异常）。
- `find_clean_last_frame_webcam` 改为**多帧中位数合成**：
  - 在末尾 90 帧内找肤色 <2% 的合格帧
  - 按肤色比例升序取最多 8 张进合成池
  - 逐像素 `np.median` 合成，SNR 改善约 √n
  - 候选帧不足 3 张时 fallback 回单帧模式

### 修改：`xhs_cover.py`

- `extract_calligraphy`：
  - Stage A（原样）找纸边 → Stage A.5 **平场校正** → Stage B 找字。
  - `flat_gray` 同时用于 bbox 检测 AND alpha 生成（关键！）
  - Stage B 用 `classify_medium` 替代 attempt-fallback；空白帧有显式警告。
  - `_pencil_extract` 在 `flat_gray` 上用**简单全局阈值 220**，不再是
    `adaptiveThreshold C=15`。（平场校正后纸面≈255，简单阈值更鲁棒。）
  - `_brush_success` 同步改为 area>500 + 中位数<60。
  - 返回签名 `→ (Image, detected_medium)`，介质显式向外传递。
- **Step 5 alpha 改从 `flat_cropped` 生成**（不是 `cropped`）。阴影被自动消除。
  RGB 通道仍用 `cropped` 原始灰度以保留笔迹自然浓淡。
- `render_ink(gray, cover_style)` 新增 `cover_style` 参数，废弃内部
  `darkest_raw > 60` 的脆弱推断：
  - `'brush'` — 毛笔默认：原样微调 ×0.9
  - `'pencil-zen'` — 铅笔默认：动态范围拉伸，最暗点映射到 **55**（禅意高级灰）
  - `'pencil-bold'` — 铅笔加浓：最暗点映射到 **25**（接近毛笔浓墨，A/B 备用）
- `generate_cover` 新增 `cover_style` 参数。不指定时默认 `brush→'brush'`、
  `pencil→'pencil-zen'`。
- CLI 新增 `--cover-style {brush, pencil-zen, pencil-bold}`。

## 保持不变

- 所有现有 CLI 参数和默认行为
- 毛笔视频的处理路径（brush 走全局阈值 80，和旧版一致）
- `ink_video_processor.py` 的 `generate_calligraphy_thumbnail`（输入是已处理的
  干净帧，input-quality 差异较小；旧 pencil fallback 仍可工作，必要时后续
  迭代再接入 `ink_extraction` 共享模块）
- 手动 `--char-region` 模式（跳过自动检测的行为）

## 验收

- **Step 0**：`_brush_success` 要求真墨色。阴影/折痕不再诱骗 auto。✓
- **Step 1**：`ink_extraction` 独立可 import。flat_field 在合成图上把纸面从
  渐变压到中位数 252；`remove_ruled_lines` 把 7 根合成竖线 + 一团笔画从 7500px
  降到 1650px（线清理、笔画保留）。✓
- **Step 2**：webcam 管线可成功载入，brush 路径回归无异常。pencil 管线对 flat_gray
  上的 background_subtract_mask 能稳定出笔迹 mask。
- **Step 3**：xhs_cover 上 `dao_v2_thumb.jpg`（毛笔）重新跑，cover 正常输出、
  分类为 brush。pencil 管线的 alpha 不再含阴影。

## 端到端测试调优（she.mp4 实测发现）

测试 4K she.mp4（铅笔在竖格练习本上）暴露了几个需要调整的细节：

1. **纸边 erode 尺寸要随分辨率**：原本固定 15px，对 4K 视频纸边的木纹/阴影残留不够。
   改为 `max(15, min(h,w)//60)`，iter=2（1080p → 36px 总收缩，4K → 72px）。
2. **3×3 MORPH_OPEN 会删铅笔笔锋**：原版用于去散点，但铅笔飞白/笔锋常只有 1-2px 宽。
   改为 `connectedComponentsWithStats` 过滤单像素分量（area<2），保留所有≥2px 的结构。
3. **ruled-line 检测阈值**：
   - 原本用全图高度作 extent → 纸面只占 74% 的 4K 视频永远过不了。改用 paper_mask 实际高度。
   - 门槛从覆盖 40%+ 高度降到 10%+（手部遮挡 + bg_subtract 只抓到部分线段，实测一条
     1596px 高的印刷线最终在 mask 里只剩 250-300px）。
4. **cluster 评分用 mask 密度**：原本 `cluster_area / aspect²` 选近方形最大 blob，
   但 144px dilate 把稀疏噪点拉成近方形大块，能打败真实字符。改为 bbox 内**原始 mask 像素数**
   / `aspect^1.5`，真实字符的 mask 密度明显高于杂散。
5. **cluster kernel 降到 20px**：原本 35-64px，在 4K 上有效合并半径 144px，
   会把邻近但无关的 ruled-line 残留拉到字符附近。20px(iter=2) → ~80px 半径，够把
   字内笔画连起来，不够吞入远端噪声。
6. **铅笔用 cluster bbox 作合成轮廓**：_pencil_pass 既然找对了 cluster，就直接
   把它作为单一合成轮廓返回，跳过下游的「个体 contour 聚类」。否则 area>100
   的过滤会把 20-80px 的单个笔画片段筛掉、只剩下最角落的一两个，导致最终 bbox
   缩回到字的一角。
7. **pencil 的个体 contour 最小面积降到 20**（brush 仍是 100）：铅笔笔画破碎成
   小片段是常态。

验证：she.mp4 最终检测框锁定在 (1909, 951) 198×204，正好框住「舍」字；
video/thumb/cover 全部生成，scale factor 3.27x 合理（此前误检 12×15 时 scale 54x）。

## 已知限制

- 对于**之前已生成的**铅笔缩略图（上游 webcam 旧管线可能把角色定位错了），
  xhs_cover 只能从该缩略图重新提取——如果缩略图本身没有可识别笔迹，封面也无
  法恢复。需用新 webcam 管线重新生成一遍原视频的 thumb。
- 铅笔对纸面反射率差 < 15% 的极淡字仍难以稳定检测——用户应使用深色铅笔、
  更大压力，或 `--char-region` 手动指定字符区域。
- 方格本双向都有线时（横+竖），两方向独立检测独立处理，但如果线本身和笔画
  粗细接近（均 ~3px）且间距小（<20px），投影检测会判"间距太近不像印刷"跳过，
  此时字形可能被 fallback 的 morphological opening 轻微影响。
- **录制要求**：书写结束后需留 1-2 秒「完成 + 手移开」的干净静帧，
  否则 `find_clean_last_frame_webcam` 找不到无手帧，thumbnail 会用
  fade-end（已黑）或最后一帧（仍有手）兜底，导致封面质量下降。
  she.mp4 是典型反面案例：视频结束时用户仍在写、手仍在字上，即使检测
  锁定了字符位置，thumbnail 里仍有手挡着。
