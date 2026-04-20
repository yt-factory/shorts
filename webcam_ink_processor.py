#!/usr/bin/env python3
"""
极客禅·墨 — Webcam 视频处理器 v1.0

Webcam 版本：视频已正确朝向，书本在左侧，白纸在灰色桌面上。
不需要旋转检测，只需裁剪白纸区域并居中墨迹。

处理流水线：
  ① 检测白纸区域
  ② 墨迹检测（在白纸区域内）
  ③ 裁剪居中（以字为中心，9:16 竖屏）
  ④ 放大（lanczos 高质量缩放）
  ⑤ 白平衡/亮度校正
  ⑥ 锐化
  ⑦ 淡入/淡出
  ⑧ 去除背景音
  ⑨ 拼接静止画面
  ⑩ 合并旁白/TTS（可选）
  ⑪ 叠加字幕（可选）
  ⑫ 生成缩略图（可选）

用法：
  python3 webcam_ink_processor.py input.mp4 -o output.mp4
  python3 webcam_ink_processor.py input.mp4 -o output.mp4 --text "旁白"
"""

import argparse
import sys
import os
import tempfile
import shutil

# 共享图像处理（平场校正、背景差分、介质分类、精确去线）
import ink_extraction as ie

# 复用手机版的共享工具函数
from ink_video_processor import (
    check_dependencies, get_video_info, get_audio_duration,
    run_ffmpeg, extract_frame, extract_last_frame,
    create_still_video, concat_videos,
    build_color_correction_filter, build_sharpen_filter,
    build_subtitle_filter, build_srt_subtitle_filter,
    normalize_audio, generate_tts, split_sentences, sample_paper_brightness,
    find_cjk_font, generate_thumbnail, generate_calligraphy_thumbnail, calculate_crop,
    DEFAULT_OUTPUT_WIDTH, DEFAULT_OUTPUT_HEIGHT, DEFAULT_FILL_RATIO,
    DEFAULT_HOLD_SECONDS, DEFAULT_FADE_IN, DEFAULT_FADE_OUT,
    DEFAULT_MIN_SCALE, TARGET_LUFS, SHORTS_MAX_DURATION, DEFAULT_FONT,
    DEFAULT_TTS_VOICE, DEFAULT_TTS_BUFFER, CHARS_PER_SECOND,
)


VERSION = "1.0"

# 品牌印章：末尾淡入、随整体淡出消失
STAMP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'files', 'seal', 'chan_seal.png')


# ============================================================
# Webcam 专用：白纸区域检测
# ============================================================

def detect_paper_region(video_path, debug=False):
    """
    检测画面中的白纸区域。

    Webcam 场景：白纸放在灰色桌面上。
    使用 Otsu 自适应阈值 + 形态学开运算 + 连通域分析：
    - Otsu 自动找最佳的"亮/暗"分界，比固定阈值更鲁棒
    - 开运算去除桌面条纹和小亮斑（kernel 25x25 足以消除常见条纹）
    - 取最大连通域作为纸张
    适合纸张占画面比例较小（<50%）的场景，比中位数扫描更鲁棒。
    返回白纸轮廓（用于生成精确遮罩）和外接矩形。

    Returns:
        dict: {'x', 'y', 'w', 'h', 'contour': np.ndarray}，检测失败返回 None
    """
    import cv2
    import numpy as np

    frame = extract_frame(video_path, 'last')
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Otsu 阈值找出"亮"区域（纸 + 浅色物体）
    _, white_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学开运算去除小噪点和细桌面条纹
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    # 连通域分析，最大的白色块即纸张
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    if n_labels < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    pw = int(stats[largest, cv2.CC_STAT_WIDTH])
    ph = int(stats[largest, cv2.CC_STAT_HEIGHT])

    if ph < h * 0.15 or pw < w * 0.15:
        return None

    # 缩进 2% 避免纸边
    shrink = int(min(pw, ph) * 0.02)
    x += shrink
    y += shrink
    pw -= 2 * shrink
    ph -= 2 * shrink

    if pw <= 0 or ph <= 0:
        return None

    # 矩形轮廓
    contour = np.array([
        [[x, y]], [[x + pw, y]], [[x + pw, y + ph]], [[x, y + ph]]
    ], dtype=np.int32)

    pct = pw * ph / (w * h) * 100
    print(f"   ✅ 白纸: ({x},{y}) {pw}x{ph}, 占画面 {pct:.1f}%")

    if debug:
        dbg = frame.copy()
        cv2.rectangle(dbg, (x, y), (x + pw, y + ph), (0, 255, 0), 3)
        dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_debug.png')
        cv2.imwrite(dp, dbg)
        print(f"      调试图: {dp}")

    return {'x': x, 'y': y, 'w': pw, 'h': ph, 'contour': contour}


# ============================================================
# Webcam 专用：墨迹检测（在白纸区域内）
# ============================================================

# 不同书写介质的灰度阈值：
#   brush  — 毛笔，墨色极深（灰度 < 50），threshold=80 可靠
#   pencil — 铅笔，笔迹较浅（灰度 ~100-180），需要更高阈值
MEDIUM_THRESHOLDS = {'brush': 80, 'pencil': 170}


def _parse_char_region(spec, frame_w, frame_h):
    """解析 --char-region 字符串。

    支持格式（数值可以是像素或 0-1 之间的小数）：
      'cx,cy'          → 以(cx,cy)为中心，自动用画面 1/4 作为 w/h
      'cx,cy,w,h'      → 完整矩形（中心点 + 尺寸）
      'x:y:w:h'        → 左上角 + 尺寸（绝对像素）
    """
    s = spec.strip()
    sep = ':' if ':' in s else ','
    parts = [p.strip() for p in s.split(sep)]
    if len(parts) not in (2, 4):
        raise ValueError(f"--char-region 需 2 或 4 个值 (cx,cy 或 cx,cy,w,h)，got: {spec}")

    def _resolve(v, axis):
        v = float(v)
        if 0 < v < 1:  # 小数视为画面比例
            return int(v * (frame_w if axis == 'x' else frame_h))
        return int(v)

    if len(parts) == 2:
        cx = _resolve(parts[0], 'x')
        cy = _resolve(parts[1], 'y')
        bw, bh = frame_w // 4, frame_h // 4
    else:
        if sep == ':':  # x:y:w:h 格式（左上角 + 尺寸）
            x = _resolve(parts[0], 'x')
            y = _resolve(parts[1], 'y')
            bw = _resolve(parts[2], 'x')
            bh = _resolve(parts[3], 'y')
            cx = x + bw // 2
            cy = y + bh // 2
        else:  # cx,cy,w,h（中心 + 尺寸）
            cx = _resolve(parts[0], 'x')
            cy = _resolve(parts[1], 'y')
            bw = _resolve(parts[2], 'x')
            bh = _resolve(parts[3], 'y')
    x = max(0, cx - bw // 2)
    y = max(0, cy - bh // 2)
    bw = min(bw, frame_w - x)
    bh = min(bh, frame_h - y)
    return {'x': x, 'y': y, 'w': bw, 'h': bh, 'cx': cx, 'cy': cy}


def detect_ink_webcam(video_path, paper=None, debug=False, medium='auto', char_region=None):
    """
    在白纸区域内检测笔迹（毛笔/铅笔）。

    核心区别（vs 手机版）：
    - 不需要旋转
    - 用白纸轮廓遮罩（而非白纸像素遮罩）精确隔离纸面，排除桌面纹理干扰
    - 更低的面积阈值（webcam 视角宽，字相对较小）

    Args:
        video_path: 视频路径
        paper: detect_paper_region 的返回值（含 contour），None 时回退到区域估算
        debug: 保存检测调试图
        medium: 书写介质 'auto' | 'brush' | 'pencil'
            - brush: 灰度阈值 80（默认，适合毛笔）
            - pencil: 灰度阈值 170（适合铅笔）
            - auto: 先按 brush 尝试，无笔迹则回退到 pencil

    Returns:
        (ink_dict, detected_medium)
            ink_dict: {'x', 'y', 'w', 'h', 'cx', 'cy'} 笔迹区域（全帧坐标）
            detected_medium: 'brush' | 'pencil'（auto 模式下实际使用的）
    """
    import cv2
    import numpy as np

    frame = extract_frame(video_path, 'last')
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 用户手动指定字符区域 — 跳过自动检测（适合极淡铅笔/lined paper）
    if char_region:
        ink = _parse_char_region(char_region, w, h)
        print(f"   ✅ 手动指定区域: ({ink['x']},{ink['y']}) {ink['w']}x{ink['h']}, 中心: ({ink['cx']},{ink['cy']})")
        # 手动模式仍按 medium 报告，brush 没指定时默认 pencil（手动模式通常用于淡笔迹）
        return ink, (medium if medium in ('brush', 'pencil') else 'pencil')

    # 构建分析遮罩：只在白纸区域内检测笔迹
    if paper and 'contour' in paper:
        # 用白纸轮廓生成精确遮罩，收缩避免纸边。收缩量需 scale-with-resolution：
        # 4K 视频纸边常有 30-50px 宽的阴影/木纹残留，固定 15px 不够。
        # 取 max(15, min(h,w)//60)：1080p→18, 4K→36（每次 erode 有 2× 效应，iter=2 → 72px 总收缩）
        erode_k = max(15, min(h, w) // 60)
        analysis_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(analysis_mask, [paper['contour']], -1, 255, -1)
        analysis_mask = cv2.erode(analysis_mask,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (erode_k, erode_k)),
                                  iterations=2)
        a_left, a_top = paper['x'], paper['y']
        a_right = paper['x'] + paper['w']
        a_bot = paper['y'] + paper['h']
    else:
        # 回退：排除左侧 20%（书本）+ 上下 5%
        analysis_mask = np.zeros(gray.shape, dtype=np.uint8)
        a_left = int(w * 0.20)
        a_right = int(w * 0.95)
        a_top = int(h * 0.05)
        a_bot = int(h * 0.95)
        analysis_mask[a_top:a_bot, a_left:a_right] = 255

    # 预计算肤色遮罩：手指阴影/指甲轮廓会被当作笔迹，用 YCrCb 色度剔除
    # （色度独立于灰度阈值，对 brush/pencil 都适用）
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb,
                            np.array([0, 133, 77], dtype=np.uint8),
                            np.array([255, 173, 127], dtype=np.uint8))
    skin_mask = cv2.dilate(skin_mask,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
                           iterations=2)
    not_skin = cv2.bitwise_not(skin_mask)

    # 平场校正：在分析之前把纸面光照渐变抹平，让 brush/pencil 都在
    # 统一的干净基底上跑阈值。flat_gray 中纸面 ≈ 255。
    try:
        flat_gray = ie.flat_field_correct(gray)
        if float(np.median(flat_gray[analysis_mask > 0])) < 150:
            # 异常：纸面中位数应接近 255，远离说明校正失败（极端光照）
            print("   ⚠️  平场校正结果异常，回退原始灰度")
            flat_gray = gray
    except Exception as _e:
        print(f"   ⚠️  平场校正失败 ({_e})，回退原始灰度")
        flat_gray = gray

    def _brush_pass():
        # 全局阈值：毛笔墨色极深 (<50)，在 flat_gray 上 threshold=80 仍然安全
        _, im = cv2.threshold(flat_gray, MEDIUM_THRESHOLDS['brush'], 255, cv2.THRESH_BINARY_INV)
        im = cv2.bitwise_and(im, analysis_mask)
        im = cv2.bitwise_and(im, not_skin)
        im = cv2.dilate(im, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        cs, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cs, im

    def _brush_success(cs):
        """毛笔检测成功的真墨色校验。
        单纯靠「大面积轮廓」会被手指阴影/桌面投影诱骗——它们能凑出上千像素，
        但灰度中位数落在 70-90 的阴影区。真正的毛笔墨迹中位数 < 60。
        同时要求 area > 1000（不再是 100），避免小噪点触发。
        在 flat_gray 上校验——确保是真墨色而非残留的光照效应。
        """
        for c in cs:
            if cv2.contourArea(c) <= 1000:
                continue
            mask = np.zeros(flat_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            vals = flat_gray[mask > 0]
            if vals.size and float(np.median(vals)) < 60:
                return True
        return False

    def _pencil_pass():
        # Pencil 管线：平场校正 → (bg_subtract ∪ 简单阈值) → 去线 → 聚类
        # 为什么两种方法并联：
        #   - bg_subtract 对 ruled-paper 场景鲁棒（局部对比），但纸面几乎纯白
        #     (~255) 时，局部膨胀 clean_paper 估计会被铅笔本身拉暗，diff 变小、
        #     Otsu 漏掉大部分字。she.mp4 (白纸) 实测：字真实暗像素 879，
        #     bg_subtract 只抓到 67。
        #   - 简单阈值 `flat_gray < 200` 直接抓任何比纸面明显暗的像素，
        #     在白纸上可靠；但在有印刷底纹/阴影的 paper 上会误抓。
        #   - 两者 OR 取并集：白纸取简单阈值的覆盖，ruled paper 取 bg_subtract 的精度。
        im_bg = ie.background_subtract_mask(flat_gray, blur_ksize=5, min_threshold=8)
        _, im_thresh = cv2.threshold(flat_gray, 200, 255, cv2.THRESH_BINARY_INV)
        im = cv2.bitwise_or(im_bg, im_thresh)
        im = cv2.bitwise_and(im, analysis_mask)
        im = cv2.bitwise_and(im, not_skin)
        # 先检测再减线：白纸图上零成本（两层去线设计）
        im = ie.remove_ruled_lines(im, paper_mask=analysis_mask,
                                   h_lines='auto', v_lines='auto')
        # ⚠️ 这里原本有 3x3 MORPH_OPEN 去散点——但铅笔笔锋/飞白往往就是 1-2px 宽，
        # 3x3 open 会把它们抹掉，导致字符失踪。改用只消除「孤立单像素」的连通域过滤。
        # 对 4K 视频，保留 ≥2 像素的连通分量即可去掉 CMOS 噪声的孤点，不伤铅笔线。
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(im, connectivity=8)
        if n_lbl > 1:
            # 面积 <2 的分量 = 孤立像素，移除（label 0 是背景，保持为 0）
            keep = np.zeros(n_lbl, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 2
            im = np.where(keep[labels], 255, 0).astype(np.uint8)
        # 关键：定位字符密集区。铅笔字像素分散，需要先用大核 close 把字
        # 笔画团成一坨，再选「形状最像方块 + 内部笔迹最多」的最大 blob，
        # 最后只保留落在其附近的原始笔画。
        # 降低 cluster_kernel：4K 视频字符 ~200x200，笔画间距 <50px，
        # 用 20-25px 的 kernel（iter=2 即 ~80-100px merge 范围）足够连接
        # 字内笔画，又不会把邻近的 ruled-line 残留碎片拉进来。
        cluster_kernel = max(20, min(h, w) // 100)
        merged = cv2.dilate(im, cv2.getStructuringElement(
            cv2.MORPH_RECT, (cluster_kernel, cluster_kernel)), iterations=2)
        big_cs, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if big_cs:
            # 评分：用 bbox 内的**原始 mask 像素数**，而非 cluster 面积。
            # cluster 面积被 144px 的 dilate 吹大，稀疏噪点也能凑出近方形
            # 大块；真实字符的 mask 密度应该明显高于杂散残留。
            def _block_score(c):
                bx0, by0, cw_, ch_ = cv2.boundingRect(c)
                aspect = max(cw_, ch_) / max(1, min(cw_, ch_))
                mask_in = int((im[by0:by0 + ch_, bx0:bx0 + cw_] > 0).sum())
                return mask_in / max(1, aspect ** 1.5)  # aspect^1.5 折衷
            best = max(big_cs, key=_block_score)
            bx, by, bw_, bh_ = cv2.boundingRect(best)
            pad = max(60, max(bw_, bh_) // 4)
            focus = np.zeros_like(im)
            focus[max(0, by - pad):by + bh_ + pad,
                  max(0, bx - pad):bx + bw_ + pad] = 255
            im = cv2.bitwise_and(im, focus)
            # 铅笔字笔画破碎，直接返回 cluster bbox 作为单一合成轮廓——
            # 让下游的 edge-artifact / spatial-cluster 阶段拿到正确的字符区域，
            # 而不是一堆 20-50px 的小片段被错误地重新聚类到字的一角。
            synth = np.array([
                [[bx, by]], [[bx + bw_, by]],
                [[bx + bw_, by + bh_]], [[bx, by + bh_]]
            ], dtype=np.int32)
            return [synth], im
        # 没有 big_cs 时退回原始连通分量
        cs, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cs, im

    # 按 medium 策略选择检测器
    # auto 模式用直方图预分类（一次判断、零分支竞争），替代旧的 attempt-fallback 循环。
    # 旧逻辑：brush 先跑，若 area>100 的任何轮廓都算成功 → 阴影诱骗
    # 新逻辑：先看纸面最暗 1% 像素的中位数，直接决定走哪条路
    if medium == 'auto':
        cls = ie.classify_medium(flat_gray, paper_mask=analysis_mask)
        if cls == 'empty':
            print("   ⚠️  画面近乎空白（纸面最暗区仍 ≥200），无法检测笔迹")
            cx = (a_left + a_right) // 2
            cy = (a_top + a_bot) // 2
            return ({'x': cx - w // 8, 'y': cy - h // 8, 'w': w // 4, 'h': h // 4,
                     'cx': cx, 'cy': cy}, 'pencil')
        detected_medium = cls
        print(f"   ℹ️  auto: 直方图分类 → {cls}")
    else:
        detected_medium = medium

    if detected_medium == 'brush':
        contours, ink_mask = _brush_pass()
        # brush 需要真墨色校验（防阴影诱骗）
        if not _brush_success(contours):
            # 验证失败：可能用户指定 brush 但实际是 pencil，尝试 pencil
            if medium == 'brush':
                print("   ⚠️  指定 brush 但未检出真墨色，仍按 brush 处理")
            else:  # classify_medium 给了 brush 但校验没过，极少见
                print("   ℹ️  brush 未通过真墨色校验，回退 pencil")
                contours, ink_mask = _pencil_pass()
                detected_medium = 'pencil'
    else:  # pencil
        contours, ink_mask = _pencil_pass()

    if not contours:
        print("   ⚠️  未检测到笔迹，使用白纸中心")
        cx = (a_left + a_right) // 2
        cy = (a_top + a_bot) // 2
        return {'x': cx - w // 8, 'y': cy - h // 8, 'w': w // 4, 'h': h // 4, 'cx': cx, 'cy': cy}, detected_medium

    # 面积过滤：最小 100px 适合毛笔，但铅笔笔画 3-5px 宽、破碎成 30-80px 的小段，
    # 100 门槛会把它们全部筛掉。pencil 用 20 作为底线（噪点已在 _pencil_pass 的
    # CC-size filter 里被消除，幸存下来的都是有结构的笔画片段）。
    mn = 20 if detected_medium == 'pencil' else 100
    mx = cv2.countNonZero(analysis_mask) * 0.15

    # 纸张边缘伪影过滤：贴近纸边 + 长宽比极端的轮廓通常是纸张边缘
    # 阴影/折痕，不是字。比如 hui.mp4 的左纸边一条 42x163 的暗条
    # （aspect 3.88）面积比真正的字笔画还大，会被误选为聚类锚点。
    edge_margin = 30  # 距纸边 30px 内算"贴边"
    aspect_max = 2.5  # 长宽比超过此阈值 + 贴边 → 视为伪影

    def _is_paper_edge_artifact(c):
        cx_, cy_, cw_, ch_ = cv2.boundingRect(c)
        touches_edge = (
            cx_ - a_left < edge_margin
            or a_right - (cx_ + cw_) < edge_margin
            or cy_ - a_top < edge_margin
            or a_bot - (cy_ + ch_) < edge_margin
        )
        aspect = max(cw_, ch_) / max(1, min(cw_, ch_))
        return touches_edge and aspect > aspect_max

    valid = [c for c in contours
             if mn < cv2.contourArea(c) < mx
             and not _is_paper_edge_artifact(c)]
    if not valid:
        # 全部被过滤时退回最大轮廓
        valid = [max(contours, key=cv2.contourArea)]

    # 空间聚类：以最大轮廓为锚点，只保留距离在 3 倍半径内的轮廓
    anchor = max(valid, key=cv2.contourArea)
    ax, ay, aw, ah = cv2.boundingRect(anchor)
    acx, acy = ax + aw // 2, ay + ah // 2
    radius = max(aw, ah) * 3
    clustered = [c for c in valid
                 if abs(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] // 2 - acx) < radius
                 and abs(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2 - acy) < radius]
    if clustered:
        valid = clustered

    pts = np.vstack(valid)
    x, y, bw, bh = cv2.boundingRect(pts)

    # 呼吸空间 25%（webcam 字小，多留一些）
    px, py = int(bw * 0.25), int(bh * 0.25)
    x = max(0, x - px)
    y = max(0, y - py)
    bw = min(w - x, bw + 2 * px)
    bh = min(h - y, bh + 2 * py)

    cx, cy = x + bw // 2, y + bh // 2

    pct = (bw * bh) / (w * h) * 100
    label = '墨迹' if detected_medium == 'brush' else '铅笔笔迹'
    print(f"   ✅ {label}: ({x},{y}) {bw}x{bh}, 占画面 {pct:.1f}%")
    print(f"      中心: ({cx},{cy}), 轮廓数: {len(valid)}, 介质: {detected_medium}")

    if debug:
        dbg = frame.copy()
        cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        cv2.circle(dbg, (cx, cy), 10, (0, 0, 255), -1)
        if paper and 'contour' in paper:
            cv2.drawContours(dbg, [paper['contour']], -1, (255, 0, 0), 2)
        dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ink_webcam_debug.png')
        cv2.imwrite(dp, dbg)
        print(f"      调试图: {dp}")

    return {'x': x, 'y': y, 'w': bw, 'h': bh, 'cx': cx, 'cy': cy}, detected_medium


# ============================================================
# Webcam 专用：找干净的最后一帧
# ============================================================

def find_clean_last_frame_webcam(video_path):
    """
    从视频末尾向前搜索「无手 + 亮度足够」的候选帧，取 5~10 帧的
    逐像素中位数合成作为最终干净帧。

    为什么要多帧中位数：
      铅笔信噪比远低于毛笔（笔迹相对纸面只有 30-60 灰度差），单帧
      CMOS 底噪会干扰后续阈值检测。多帧中位数对偶发动作（手影一闪、
      画面抖动、风扇气流）天然鲁棒，SNR 改善约 √n，是铅笔稳定检测
      的关键前置步骤。

    两层亮度兜底：
      - 已处理过的视频末尾有 fade-out（全黑帧），必须跳过
      - dao_v2.mp4 这类纸面本身偏暗的源，max≈190 也要能找到合格帧
      所以用「搜索范围内最大亮度的 85%」作自适应地板。

    候选帧偏好：肤色比例最低（手最干净）优先进入合成池。
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    SEARCH_FRAMES = 90
    TARGET_POOL = 8  # 合成目标帧数（5-10 之间折衷）

    # Pass 1: 收集每帧的亮度（轻量，仅 grayscale 均值）
    brightness_by_offset = {}
    for offset in range(1, min(SEARCH_FRAMES + 1, total)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - offset)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_by_offset[offset] = float(np.mean(gray))

    if not brightness_by_offset:
        cap.release()
        return extract_frame(video_path, 'last')

    max_bright = max(brightness_by_offset.values())
    brightness_floor = max_bright * 0.85  # 自适应阈值
    qualifying = [o for o, b in brightness_by_offset.items() if b >= brightness_floor]

    # Pass 2: 收集所有亮度合格帧的肤色比例（**去掉绝对阈值**，改用相对排序）
    # 旧逻辑：要求 skin<2%，但 she.mp4 这类全程手在画面的视频全部不合格、
    # 合成池永远为空、退回单帧 fallback——多帧中位数根本没机会跑。
    # 新逻辑：取 skin% 最低的 k 帧（无论绝对值），手在不同位置的部分
    # 会被中位数平均淡化，静态的字符保留。
    candidates: list[tuple[float, int, np.ndarray]] = []  # (skin, offset, frame)
    for offset in sorted(qualifying):
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - offset)
        ret, frame = cap.read()
        if not ret:
            continue
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb,
                                np.array([0, 133, 77], dtype=np.uint8),
                                np.array([255, 173, 127], dtype=np.uint8))
        skin_ratio = float(np.mean(skin_mask > 0))
        candidates.append((skin_ratio, offset, frame.copy()))

    cap.release()

    if not candidates:
        print("      ⚠️  未找到亮度合格帧（可能全部淡出），回退到最后一帧")
        return extract_frame(video_path, 'last')

    # 按肤色比例升序，取最干净的 N 帧（相对排序，不用绝对阈值）
    candidates.sort(key=lambda t: t[0])
    pool = candidates[:TARGET_POOL]

    if len(pool) < 3:
        skin, offset, frame = pool[0]
        print(f"      ⚠️  亮度合格帧仅 {len(pool)} 张，回退单帧模式（倒数第 {offset} 帧, 肤色 {skin*100:.2f}%）")
        return frame

    # 多帧中位数合成
    stack = np.stack([f for _, _, f in pool], axis=0)
    composite = np.median(stack, axis=0).astype(np.uint8)
    offsets = [o for _, o, _ in pool]
    skins = [s for s, _, _ in pool]
    print(f"      合成池: {len(pool)} 帧 (末尾倒数 {min(offsets)}-{max(offsets)}), "
          f"肤色范围 {min(skins)*100:.1f}%-{max(skins)*100:.1f}%")
    return composite


# ============================================================
# Webcam 专用：全视频搜索最佳封面源帧
# ============================================================

def find_best_cover_frame(video_path, paper=None, ink_bbox=None, sample_fps=1.5):
    """
    全视频搜索「字最完整 + 字上方的手遮挡最少」的 k 帧，逐像素中位数合成。

    与 find_clean_last_frame_webcam 的区别：
      - 后者只看末尾 90 帧，用于视频 bbox（观众看过程，单帧脏一点没事）
      - 本函数扫全视频，用于封面 thumb（静态图要求高）

    观察：书写者完成一个字后常有「短暂抬手检查」的瞬间，可能在末尾也可能在中段。
    仅末尾搜索完全错过这种黄金帧。

    关键指标（经过 she.mp4 实测修正）：
      - 不能用「纸面内 ink_area 最大」——书写中的手+臂在 bg_subtract 产生
        大量 shadow-induced 假 ink 信号（skin mask 遮掉的只是手本身，
        臂/袖的阴影扩展到周围），ink_area 反而随手的存在而增加。
      - 改用「字符 bbox 内的肤色占比最低」——直接对应「手最不挡字」，
        再要求字符 bbox 内 flat_gray 有足够暗像素（证明字真的写在那里）。

    算法：
      1. 按 sample_fps + 末尾 30 帧密采样混合
      2. 对每个采样帧：
         a) skin_over_char: ink_bbox 内肤色覆盖（手遮挡字的代理）
         b) char_ink: ink_bbox 内 flat_gray 的暗像素数（字的可见度）
      3. 筛选 char_ink ≥ max(char_ink) × 0.3 的帧（字基本可见）
      4. 按 skin_over_char 升序取前 5-8 帧
      5. BGR 重读这些帧，逐像素 np.median 合成
      6. <3 帧 → 返回 None

    Args:
        video_path: 视频文件路径
        paper: detect_paper_region 返回（选传）
        ink_bbox: {'x','y','w','h'}，字符在帧上的大致位置（用于定位「手挡字」区域）

    Returns:
        composite_bgr (np.ndarray) 或 None（候选不足）
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / max(fps, 1e-6)
    step = max(1, int(fps / sample_fps))
    # 粗采样整视频 + 密采样末尾 30 帧。末尾密采样很关键：
    # 书写者完成后常有 <1s 的抬手瞬间，粗采样步长 (fps/1.5 ≈ 6-20 帧)
    # 有很大概率直接跳过这个黄金窗口。
    coarse = list(range(0, total, step))
    tail_start = max(0, total - 30)
    dense_tail = list(range(tail_start, total))
    sample_indices = sorted(set(coarse) | set(dense_tail))
    print(f"      全视频采样: {len(sample_indices)} 帧 ("
          f"粗采样 {len(coarse)} + 末尾密采样 {len(dense_tail)}; "
          f"总 {total} 帧, {duration:.1f}s @ {sample_fps} fps)")

    # 纸面 mask（用于把计量框限定在纸内）
    if paper and 'contour' in paper:
        # 需要从任意一帧拿到 frame shape
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret0, f0 = cap.read()
        if not ret0:
            cap.release()
            return None
        h0, w0 = f0.shape[:2]
        paper_mask = np.zeros((h0, w0), dtype=np.uint8)
        cv2.drawContours(paper_mask, [paper['contour']], -1, 255, -1)
        paper_mask = cv2.erode(paper_mask,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                               iterations=1)
    else:
        paper_mask = None
        h0 = w0 = None

    skin_lo = np.array([0, 133, 77], dtype=np.uint8)
    skin_hi = np.array([255, 173, 127], dtype=np.uint8)

    # 解析 ink_bbox（字符区域），在其上做 skin occlusion 测量
    if ink_bbox:
        cbx, cby = int(ink_bbox['x']), int(ink_bbox['y'])
        cbw, cbh = int(ink_bbox['w']), int(ink_bbox['h'])
    else:
        # 无 bbox 时用画面中心 1/3 作兜底
        cbx = (w0 or 1920) // 3; cby = (h0 or 1080) // 3
        cbw = (w0 or 1920) // 3; cbh = (h0 or 1080) // 3

    stats: list[tuple[int, float, int]] = []  # (idx, skin_over_char, char_ink)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if paper_mask is None:
            h_, w_ = gray.shape
            paper_mask = np.ones((h_, w_), dtype=np.uint8) * 255
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, skin_lo, skin_hi)
        # Skin 占字符 bbox 的比例：手挡字的直接代理
        char_skin = skin[cby:cby + cbh, cbx:cbx + cbw]
        skin_over_char = float((char_skin > 0).mean()) if char_skin.size else 1.0
        # 字符 bbox 内 flat_gray 的可见暗像素（保证字真的在那里）
        flat = ie.flat_field_correct(gray)
        char_flat = flat[cby:cby + cbh, cbx:cbx + cbw]
        char_ink = int((char_flat < 200).sum()) if char_flat.size else 0
        stats.append((idx, skin_over_char, char_ink))

    if not stats:
        cap.release()
        return None

    # 核心洞察（she.mp4 实测）：书写中的手/臂阴影在字符 bbox 内产生 10-20k 的
    # 假 dark 信号；手移开后字真正的暗像素只有 1-3k。若用「char_ink 最大」筛选，
    # 永远选到"手最近"的帧。
    # 正确策略：先按 skin_over_char 升序选"手最不遮字"的前 N 帧，
    # 再过滤掉 char_ink < 500 的空白帧（书写开始前的帧）。
    MIN_CHAR_INK = 500  # 500 px dark 足以说明字开始写了
    meaningful = [s for s in stats if s[2] >= MIN_CHAR_INK]
    if not meaningful:
        cap.release()
        ink_max = max((s[2] for s in stats), default=0)
        print(f"      ⚠️  全视频字符 bbox 内暗像素 max={ink_max} <500，字未写")
        return None

    meaningful.sort(key=lambda s: s[1])  # skin_over_char 升序
    pool_stats = meaningful[:8]
    print(f"      有字候选 {len(meaningful)} 帧 (char_ink≥{MIN_CHAR_INK}), "
          f"取 skin_over_char 最低前 {len(pool_stats)}")
    if len(pool_stats) < 3:
        cap.release()
        print(f"      ⚠️  可用候选仅 {len(pool_stats)} 帧，不足以合成")
        return None

    # 全彩重读 pool 帧
    frames = []
    for idx, _, _ in pool_stats:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if len(frames) < 3:
        return None

    composite = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    idxs = [s[0] for s in pool_stats[:len(frames)]]
    skins = [s[1] for s in pool_stats[:len(frames)]]
    inks = [s[2] for s in pool_stats[:len(frames)]]
    print(f"      封面合成池: {len(frames)} 帧 "
          f"(帧号 {min(idxs)}-{max(idxs)}/{total}, "
          f"字 bbox 内 skin {min(skins)*100:.1f}%-{max(skins)*100:.1f}%, "
          f"char_ink {min(inks)}-{max(inks)} px)")
    return composite


# ============================================================
# 主处理流程
# ============================================================

def process_webcam_video(input_path, output_path, voiceover_path=None,
                         target_fill_ratio=DEFAULT_FILL_RATIO,
                         hold_seconds=DEFAULT_HOLD_SECONDS,
                         output_width=DEFAULT_OUTPUT_WIDTH,
                         output_height=DEFAULT_OUTPUT_HEIGHT,
                         enable_color_correct=True,
                         fade_in=DEFAULT_FADE_IN,
                         fade_out=DEFAULT_FADE_OUT,
                         subtitle_text=None,
                         tts_text=None,
                         tts_voice=DEFAULT_TTS_VOICE,
                         font_path=None,
                         generate_thumb=True,
                         debug=False,
                         medium='auto',
                         char_region=None):
    """Webcam 版主处理流程。"""

    print("=" * 60)
    print(f"极客禅·墨 Webcam 处理器 v{VERSION}")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    clean_frame_raw = None
    clean_frame_processed = None

    try:
        # Step 1: 视频信息
        print("\n📹 Step 1: 分析视频...")
        vi = get_video_info(input_path)
        print(f"   尺寸: {vi['width']}x{vi['height']}, 时长: {vi['duration']:.1f}s, 帧率: {vi['fps']:.0f}fps")

        # Step 2: 检测白纸区域
        print("\n📄 Step 2: 检测白纸...")
        paper = detect_paper_region(input_path, debug=debug)

        # Step 2.5: 采样纸面亮度 → 色彩校正归一化（v4：根治下游「假设白纸」级联失败）
        import cv2
        import numpy as np
        if paper and 'contour' in paper:
            _probe = extract_frame(input_path, 'last')
            _ph, _pw = _probe.shape[:2]
            _paper_mask = np.zeros((_ph, _pw), dtype=np.uint8)
            cv2.drawContours(_paper_mask, [paper['contour']], -1, 255, -1)
            paper_p95 = sample_paper_brightness(input_path, _paper_mask, 'last')
        else:
            paper_p95 = sample_paper_brightness(input_path, None, 'last')
        _gain_est = min(max(240.0 / max(paper_p95, 1.0), 0.8), 2.5)
        print(f"   📊 纸面归一化: p95={paper_p95:.0f} → 240 (gain≈{_gain_est:.2f})")

        # Step 3: 笔迹检测（支持毛笔/铅笔/手动指定区域）
        if char_region:
            print(f"\n🔍 Step 3: 用户指定字符区域 ({char_region})...")
        else:
            print(f"\n🔍 Step 3: 检测笔迹 (medium={medium})...")
        ink, detected_medium = detect_ink_webcam(input_path, paper, debug=debug,
                                                 medium=medium, char_region=char_region)

        # Step 4: 计算裁剪参数
        # 约束裁剪框在白纸区域内
        if paper:
            crop_x_min = paper['x']
            crop_x_max = paper['x'] + paper['w']
            crop_y_min = paper['y']
            crop_y_max = paper['y'] + paper['h']
        else:
            crop_x_min = int(vi['width'] * 0.15)
            crop_x_max = vi['width']
            crop_y_min = 0
            crop_y_max = vi['height']

        crop = calculate_crop(
            ink, vi['width'], vi['height'],
            target_fill_ratio, output_width, output_height,
            y_min=crop_y_min, y_max=crop_y_max,
        )
        # 约束 x 方向不超出白纸（位置 + 宽度都钳位）
        if crop['x'] < crop_x_min:
            crop['x'] = crop_x_min
        if crop['x'] + crop['w'] > crop_x_max:
            crop['x'] = max(crop_x_min, crop_x_max - crop['w'])
        # 裁剪宽度不超过纸面宽度（保持 9:16 宽高比避免缩放失真）
        available_w = crop_x_max - crop['x']
        if crop['w'] > available_w:
            crop['w'] = available_w - (available_w % 2)  # ffmpeg 偶数
            target_aspect = output_width / output_height
            crop['h'] = int(crop['w'] / target_aspect)
            crop['h'] -= crop['h'] % 2
            crop['y'] = max(crop_y_min, min(ink['cy'] - crop['h'] // 2, crop_y_max - crop['h']))
            crop['scale_factor'] = output_width / crop['w']

        sf = crop['scale_factor']

        print(f"\n✂️  Step 4: 裁剪参数")
        print(f"   区域: ({crop['x']},{crop['y']}) {crop['w']}x{crop['h']}")
        print(f"   放大: {sf:.2f}x, 目标字占比: ~{target_fill_ratio * 100:.0f}%")

        # Step 5-7: 构建滤镜链
        print(f"\n🎬 Step 5-7: 视频处理...")
        vf_parts = []

        # 裁剪
        vf_parts.append(f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}")
        print(f"   ✓ 裁剪居中")

        # 缩放
        vf_parts.append(f"scale={output_width}:{output_height}:flags=lanczos")
        print(f"   ✓ 缩放 {output_width}x{output_height}")

        # 白平衡
        if enable_color_correct:
            vf_parts.append(build_color_correction_filter(paper_p95))
            print("   ✓ 白平衡校正")

        # 锐化
        sharp = build_sharpen_filter(sf)
        vf_parts.append(sharp)
        print(f"   ✓ 锐化 ({sharp.split('=')[0]})")

        # 淡入
        if fade_in > 0:
            vf_parts.append(f"fade=t=in:st=0:d={fade_in}")
            print(f"   ✓ 淡入 {fade_in}s")

        # 一次性处理
        cropped_video = os.path.join(tmpdir, 'processed.mp4')
        run_ffmpeg([
            'ffmpeg', '-y', '-i', input_path,
            '-vf', ','.join(vf_parts),
            '-an',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            cropped_video
        ], "主视频处理")
        print("   ✅ 视频处理完成")

        # Step 8: 拼接静止画面
        current_video = cropped_video

        if hold_seconds > 0:
            print(f"\n⏸️  Step 8: 静止画面 ({hold_seconds}s)...")
            print("   检测无手帧...")

            import cv2
            clean_frame = find_clean_last_frame_webcam(input_path)

            clean_frame_raw = os.path.join(tmpdir, 'clean_raw.png')
            cv2.imwrite(clean_frame_raw, clean_frame)

            # 对帧应用相同的裁剪+缩放+色彩校正
            vf_still = []
            vf_still.append(f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}")
            vf_still.append(f"scale={output_width}:{output_height}:flags=lanczos")
            if enable_color_correct:
                vf_still.append(build_color_correction_filter(paper_p95))
            vf_still.append(build_sharpen_filter(sf))

            clean_frame_processed = os.path.join(tmpdir, 'clean_processed.png')
            run_ffmpeg([
                'ffmpeg', '-y', '-i', clean_frame_raw,
                '-vf', ','.join(vf_still),
                clean_frame_processed
            ], "处理静止帧")

            hold_video = os.path.join(tmpdir, 'hold.mp4')
            create_still_video(clean_frame_processed, hold_video, hold_seconds, vi['fps'])

            merged = os.path.join(tmpdir, 'merged.mp4')
            concat_videos([cropped_video, hold_video], merged)
            current_video = merged
            print("   ✅ 静止画面完成")

        # Step 9: TTS
        tts_srt_path = None

        if tts_text and not voiceover_path:
            print(f"\n🗣️  Step 9: TTS 语音合成...")
            cur_dur = get_video_info(current_video)['duration']
            # 旁白与视频「同时」播放（混音于 t=0），不是接在视频后。
            # 所以预算应为视频长度本身，让旁白以自然语速覆盖书写过程；
            # 上限 SHORTS_MAX_DURATION-1 防止旁白延长后超 60s。
            # （此前误用 SHORTS_MAX_DURATION-cur_dur，把旁白挤进剩余时间，
            #  例如 hui.mp4 视频 45s → 预算 15s → 121 字被压成 14.9s 极速念完，
            #  字才写到 1/3 旁白就结束了。）
            tts_budget = min(cur_dur, SHORTS_MAX_DURATION - 1)
            print(f"   视频: {cur_dur:.0f}s, 旁白预算: {tts_budget:.0f}s")
            tts_audio_path, tts_srt_path = generate_tts(
                tts_text, tts_budget, tmpdir, voice=tts_voice,
            )
            voiceover_path = tts_audio_path

        # Step 10: 合并旁白
        if voiceover_path and os.path.exists(voiceover_path):
            print(f"\n🎙️  Step 10: 合并旁白...")

            normalized_audio = os.path.join(tmpdir, 'normalized.m4a')
            normalize_audio(voiceover_path, normalized_audio)

            vid_dur = get_video_info(current_video)['duration']
            vo_dur = get_audio_duration(normalized_audio)
            print(f"   视频: {vid_dur:.1f}s, 旁白: {vo_dur:.1f}s")

            if vo_dur > vid_dur:
                extra = vo_dur - vid_dur + 1.0
                print(f"   延长静止画面 {extra:.1f}s...")
                last_frame = os.path.join(tmpdir, 'extend_frame.png')
                extract_last_frame(current_video, last_frame)
                extra_vid = os.path.join(tmpdir, 'extra.mp4')
                create_still_video(last_frame, extra_vid, extra, vi['fps'])
                extended = os.path.join(tmpdir, 'extended.mp4')
                concat_videos([current_video, extra_vid], extended)
                current_video = extended

            final_with_audio = os.path.join(tmpdir, 'with_audio.mp4')
            run_ffmpeg([
                'ffmpeg', '-y',
                '-i', current_video,
                '-i', normalized_audio,
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                '-map', '0:v:0', '-map', '1:a:0',
                final_with_audio
            ], "合并旁白")
            current_video = final_with_audio
            print("   ✅ 旁白合并完成")
        else:
            print(f"\n📝 Step 10: 无旁白，输出静音视频")

        # 保存无字幕视频路径，用于缩略图（只展示字，不含字幕文字）
        thumb_source_video = current_video

        # Step 11: 淡出 + 字幕 + 印章
        stamp_enabled = os.path.exists(STAMP_PATH)

        cur_info = None
        fade_start = None
        if fade_out > 0:
            print(f"\n🌑 Step 11: 淡出 ({fade_out}s)...")
            cur_info = get_video_info(current_video)
            fade_start = cur_info['duration'] - fade_out
            print(f"   淡出起始: {fade_start:.1f}s (总时长 {cur_info['duration']:.1f}s)")

        subtitle_filter = None
        if tts_srt_path and os.path.exists(tts_srt_path):
            print(f"\n📝 叠加同步字幕（SRT）...")
            subtitle_filter = build_srt_subtitle_filter(tts_srt_path, font_path, output_width, output_height)
        elif subtitle_text:
            print(f"\n📝 叠加字幕: \"{subtitle_text}\"")
            subtitle_filter = build_subtitle_filter(subtitle_text, output_width, output_height, font_path)

        has_effects = (fade_out > 0) or (subtitle_filter is not None)

        if stamp_enabled:
            # 印章叠加参数
            stamp_width_ratio = 0.12      # 画面宽度的 12%
            stamp_margin_ratio = 0.05     # 距边 5%
            stamp_opacity = 0.75          # 75% 不透明度
            stamp_fade_in = 0.5           # 淡入时长
            stamp_hold = 2.0              # 停留时长
            stamp_lead = stamp_fade_in + stamp_hold  # 印章在淡出前多久出现 = 2.5s

            if cur_info is None:
                cur_info = get_video_info(current_video)
            total_dur = cur_info['duration']

            # 印章出现时间：淡出开始前 2.5s；无淡出时为结尾前 3.5s
            if fade_out > 0:
                stamp_start = max(0.0, fade_start - stamp_lead)
            else:
                stamp_start = max(0.0, total_dur - stamp_lead - 1.0)

            stamp_w = int(output_width * stamp_width_ratio)
            stamp_margin_x = int(output_width * stamp_margin_ratio)
            stamp_margin_y = int(output_height * stamp_margin_ratio)
            overlay_x = output_width - stamp_w - stamp_margin_x
            overlay_y_expr = f"{output_height}-overlay_h-{stamp_margin_y}"

            # 印章预处理：缩放 + 透明度 + 淡入（alpha=1 只改透明通道）
            stamp_filters = (
                f"[1:v]scale={stamp_w}:-1,"
                f"format=rgba,"
                f"colorchannelmixer=aa={stamp_opacity},"
                f"fade=t=in:st={stamp_start:.2f}:d={stamp_fade_in}:alpha=1"
                f"[stamp]"
            )

            # overlay 时间窗口：stamp_start 之后才叠印章
            overlay_filter = (
                f"[0:v][stamp]overlay="
                f"x={overlay_x}:y={overlay_y_expr}:"
                f"enable='gte(t,{stamp_start:.2f})'"
                f"[stamped]"
            )

            # 在印章之后叠加淡出和字幕（整体一起淡到黑）
            post_chain_parts = []
            if fade_out > 0:
                post_chain_parts.append(f"fade=t=out:st={fade_start:.2f}:d={fade_out}")
            if subtitle_filter is not None:
                post_chain_parts.append(subtitle_filter)

            if post_chain_parts:
                post_chain = f"[stamped]{','.join(post_chain_parts)}[out]"
            else:
                post_chain = "[stamped]copy[out]"

            filter_complex = f"{stamp_filters};{overlay_filter};{post_chain}"

            post_video = os.path.join(tmpdir, 'post_processed.mp4')
            run_ffmpeg([
                'ffmpeg', '-y',
                '-i', current_video,
                '-loop', '1', '-i', STAMP_PATH,
                '-filter_complex', filter_complex,
                '-map', '[out]',
                '-map', '0:a?',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'copy',
                '-pix_fmt', 'yuv420p',
                '-t', f'{total_dur:.3f}',
                post_video
            ], "淡出+字幕+印章")
            current_video = post_video
            print(f"   🔴 印章叠加: {stamp_start:.1f}s 淡入, 停留至淡出")
            print("   ✅ 后期处理完成")
        elif has_effects:
            print("ℹ️  未找到印章文件，跳过")
            post_filters = []
            if fade_out > 0:
                post_filters.append(f"fade=t=out:st={fade_start:.2f}:d={fade_out}")
            if subtitle_filter is not None:
                post_filters.append(subtitle_filter)

            post_video = os.path.join(tmpdir, 'post_processed.mp4')
            run_ffmpeg([
                'ffmpeg', '-y', '-i', current_video,
                '-vf', ','.join(post_filters),
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'copy',
                '-pix_fmt', 'yuv420p',
                post_video
            ], "淡出+字幕")
            current_video = post_video
            print("   ✅ 后期处理完成")
        else:
            print("ℹ️  未找到印章文件，跳过")

        # 输出
        shutil.copy2(current_video, output_path)

        # 缩略图
        # 架构变化：封面源帧用 find_best_cover_frame（全视频搜「字最完整+手最少」
        # 的合成池），而非 find_clean_last_frame_webcam（只看末尾 90 帧、偏向 hold
        # still 的场景）。两个目标解耦：视频过程看到的最后一帧可以脏，但封面是
        # 静态图、对干净度要求高得多。
        if generate_thumb:
            thumb_path = os.path.splitext(output_path)[0] + '_thumb.jpg'
            print(f"\n🖼️  生成缩略图...")
            print("   搜索最佳封面源帧（全视频）...")
            import cv2
            cover_src = find_best_cover_frame(input_path, paper=paper, ink_bbox=ink)
            if cover_src is None:
                print("   ⚠️  全视频未找到干净封面帧，回退到 hold-still 的末尾合成帧")
                cover_processed = clean_frame_processed
            else:
                # 对该帧跑与主视频一致的裁剪+缩放+色彩+锐化管线
                cover_raw_path = os.path.join(tmpdir, 'cover_raw.png')
                cv2.imwrite(cover_raw_path, cover_src)
                cover_processed = os.path.join(tmpdir, 'cover_processed.png')
                vf_cover = [f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}",
                            f"scale={output_width}:{output_height}:flags=lanczos"]
                if enable_color_correct:
                    vf_cover.append(build_color_correction_filter(paper_p95))
                vf_cover.append(build_sharpen_filter(sf))
                run_ffmpeg([
                    'ffmpeg', '-y', '-i', cover_raw_path,
                    '-vf', ','.join(vf_cover),
                    cover_processed
                ], "处理封面源帧")

            if cover_processed and os.path.exists(cover_processed):
                ok = generate_calligraphy_thumbnail(cover_processed, thumb_path,
                                                     output_width, output_height,
                                                     medium=detected_medium)
            else:
                ok = False
            if not ok:
                generate_thumbnail(thumb_source_video, thumb_path)
            print(f"   ✅ {thumb_path}")

        # 报告
        fi = get_video_info(output_path)
        sz = os.path.getsize(output_path) / 1024 / 1024

        print(f"\n{'=' * 60}")
        print(f"✅ 处理完成!")
        print(f"{'=' * 60}")
        print(f"   文件: {output_path}")
        print(f"   尺寸: {fi['width']}x{fi['height']}")
        print(f"   时长: {fi['duration']:.1f}s")
        print(f"   大小: {sz:.1f}MB")

        steps = ["✂️裁剪", f"🔎放大{sf:.1f}x"]
        if enable_color_correct:
            steps.append("🎨色彩")
        steps.append("🔍锐化")
        if fade_in > 0:
            steps.append(f"▶️淡入{fade_in}s")
        if fade_out > 0:
            steps.append(f"⏹淡出{fade_out}s")
        steps.append("🔇去音")
        if hold_seconds > 0:
            steps.append(f"⏸停留{hold_seconds}s")
        if voiceover_path:
            steps.append("🎙️旁白")
        print(f"   处理: {' → '.join(steps)}")
        print(f"   {'📱 Shorts可用 (≤60s)' if fi['duration'] <= SHORTS_MAX_DURATION else '⚠️ 超60s'}")
        print(f"{'=' * 60}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=f'极客禅·墨 Webcam 处理器 v{VERSION}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 webcam_ink_processor.py input.mp4 -o output.mp4
  python3 webcam_ink_processor.py input.mp4 -o output.mp4 --text "旁白文本"
  python3 webcam_ink_processor.py input.mp4 -o output.mp4 --fill 0.7
        """
    )

    parser.add_argument('input', help='原始视频文件路径')
    parser.add_argument('--output', '-o', default='output.mp4', help='输出文件路径')
    parser.add_argument('--voiceover', '-v', help='旁白音频文件')

    parser.add_argument('--fill', '-f', type=float, default=DEFAULT_FILL_RATIO,
                        help=f'字占画面比例 (默认: {DEFAULT_FILL_RATIO})')
    parser.add_argument('--hold', type=float, default=DEFAULT_HOLD_SECONDS,
                        help=f'停留秒数 (默认: {DEFAULT_HOLD_SECONDS})')
    parser.add_argument('--width', type=int, default=DEFAULT_OUTPUT_WIDTH,
                        help=f'输出宽度 (默认: {DEFAULT_OUTPUT_WIDTH})')
    parser.add_argument('--height', type=int, default=DEFAULT_OUTPUT_HEIGHT,
                        help=f'输出高度 (默认: {DEFAULT_OUTPUT_HEIGHT})')

    parser.add_argument('--fade-in', type=float, default=DEFAULT_FADE_IN,
                        help=f'淡入秒 (默认: {DEFAULT_FADE_IN})')
    parser.add_argument('--fade-out', type=float, default=DEFAULT_FADE_OUT,
                        help=f'淡出秒 (默认: {DEFAULT_FADE_OUT})')
    parser.add_argument('--no-color-correct', action='store_true', help='跳过白平衡')
    parser.add_argument('--no-thumbnail', action='store_true', help='不生成缩略图')
    parser.add_argument('--debug', action='store_true', help='保存检测调试图 (paper_debug.png, ink_webcam_debug.png)')
    parser.add_argument('--medium', choices=['auto', 'brush', 'pencil'], default='auto',
                        help='书写介质：brush=毛笔(墨色深), pencil=铅笔(笔迹浅), auto=自动 (默认: auto)')
    parser.add_argument('--char-region',
                        help='手动指定字符区域，跳过自动检测（适合极淡铅笔/lined paper）。'
                             '格式: "cx,cy" 或 "cx,cy,w,h" 或 "x:y:w:h"。'
                             '数值<1视为画面比例，否则为像素。例: "0.55,0.5" 或 "1900,1100,800,800"')

    parser.add_argument('--subtitle', help='静态字幕文字')
    parser.add_argument('--font', default=DEFAULT_FONT, help=f'字体路径 (默认: {DEFAULT_FONT})')

    tts_group = parser.add_mutually_exclusive_group()
    tts_group.add_argument('--text', help='TTS 旁白文本')
    tts_group.add_argument('--text-file', help='TTS 旁白文本文件')
    parser.add_argument('--voice', default=DEFAULT_TTS_VOICE,
                        help=f'TTS 语音 (默认: {DEFAULT_TTS_VOICE})')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 找不到: {args.input}")
        sys.exit(1)
    if args.voiceover and not os.path.exists(args.voiceover):
        print(f"❌ 找不到: {args.voiceover}")
        sys.exit(1)

    tts_text = None
    if args.text:
        tts_text = args.text
    elif args.text_file:
        if not os.path.exists(args.text_file):
            print(f"❌ 找不到: {args.text_file}")
            sys.exit(1)
        with open(args.text_file, 'r', encoding='utf-8') as f:
            tts_text = f.read().strip()

    if tts_text and args.voiceover:
        print("⚠️  同时提供了 --text 和 --voiceover，使用 voiceover（忽略 TTS）")
        tts_text = None

    check_dependencies()

    process_webcam_video(
        input_path=args.input,
        output_path=args.output,
        voiceover_path=args.voiceover,
        target_fill_ratio=args.fill,
        hold_seconds=args.hold,
        output_width=args.width,
        output_height=args.height,
        enable_color_correct=not args.no_color_correct,
        fade_in=args.fade_in,
        fade_out=args.fade_out,
        subtitle_text=args.subtitle,
        tts_text=tts_text,
        tts_voice=args.voice,
        font_path=args.font,
        generate_thumb=not args.no_thumbnail,
        debug=args.debug,
        medium=args.medium,
        char_region=args.char_region,
    )


if __name__ == '__main__':
    main()
