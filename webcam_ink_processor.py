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
import asyncio
import subprocess
import sys
import os
import json
import tempfile
import shutil

# 复用手机版的共享工具函数
from ink_video_processor import (
    check_dependencies, get_video_info, get_audio_duration,
    run_ffmpeg, extract_frame, extract_last_frame,
    create_still_video, concat_videos,
    build_color_correction_filter, build_sharpen_filter,
    build_subtitle_filter, build_srt_subtitle_filter,
    normalize_audio, generate_tts, split_sentences,
    find_cjk_font, generate_thumbnail, generate_calligraphy_thumbnail, calculate_crop,
    DEFAULT_OUTPUT_WIDTH, DEFAULT_OUTPUT_HEIGHT, DEFAULT_FILL_RATIO,
    DEFAULT_HOLD_SECONDS, DEFAULT_FADE_IN, DEFAULT_FADE_OUT,
    DEFAULT_MIN_SCALE, TARGET_LUFS, SHORTS_MAX_DURATION, DEFAULT_FONT,
    DEFAULT_TTS_VOICE, DEFAULT_TTS_BUFFER, CHARS_PER_SECOND,
)


VERSION = "1.0"


# ============================================================
# Webcam 专用：白纸区域检测
# ============================================================

def detect_paper_region(video_path):
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

    # 调试图
    dbg = frame.copy()
    cv2.rectangle(dbg, (x, y), (x + pw, y + ph), (0, 255, 0), 3)
    dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_debug.png')
    cv2.imwrite(dp, dbg)

    pct = pw * ph / (w * h) * 100
    print(f"   ✅ 白纸: ({x},{y}) {pw}x{ph}, 占画面 {pct:.1f}%")
    print(f"      调试图: {dp}")

    return {'x': x, 'y': y, 'w': pw, 'h': ph, 'contour': contour}


# ============================================================
# Webcam 专用：墨迹检测（在白纸区域内）
# ============================================================

def detect_ink_webcam(video_path, paper=None):
    """
    在白纸区域内检测墨迹。

    核心区别（vs 手机版）：
    - 不需要旋转
    - 用白纸轮廓遮罩（而非白纸像素遮罩）精确隔离纸面，排除桌面纹理干扰
    - 更低的面积阈值（webcam 视角宽，字相对较小）

    Args:
        video_path: 视频路径
        paper: detect_paper_region 的返回值（含 contour），None 时回退到区域估算

    Returns:
        dict: {'x', 'y', 'w', 'h', 'cx', 'cy'} 墨迹区域（全帧坐标）
    """
    import cv2
    import numpy as np

    frame = extract_frame(video_path, 'last')
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 构建分析遮罩：只在白纸区域内检测墨迹
    if paper and 'contour' in paper:
        # 用白纸轮廓生成精确遮罩，收缩 15px 避免纸边
        analysis_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(analysis_mask, [paper['contour']], -1, 255, -1)
        analysis_mask = cv2.erode(analysis_mask,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                                  iterations=1)
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

    # 检测深色像素（墨迹）
    _, ink_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    ink_mask = cv2.bitwise_and(ink_mask, analysis_mask)

    # 排除手部：手指阴影/指甲轮廓会被当作墨迹，用肤色检测剔除
    # 肤色区域膨胀 25px 覆盖手指边缘的暗轮廓
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb,
                            np.array([0, 133, 77], dtype=np.uint8),
                            np.array([255, 173, 127], dtype=np.uint8))
    skin_mask = cv2.dilate(skin_mask,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
                           iterations=2)
    ink_mask = cv2.bitwise_and(ink_mask, cv2.bitwise_not(skin_mask))

    # 轻度膨胀连接笔画（webcam 字小，不能太激进）
    ink_mask = cv2.dilate(ink_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("   ⚠️  未检测到墨迹，使用白纸中心")
        cx = (a_left + a_right) // 2
        cy = (a_top + a_bot) // 2
        return {'x': cx - w // 8, 'y': cy - h // 8, 'w': w // 4, 'h': h // 4, 'cx': cx, 'cy': cy}

    # 面积过滤：最小 100px（webcam 字很小），最大 15% 分析区域
    mn = 100
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

    # 调试图
    dbg = frame.copy()
    cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
    cv2.circle(dbg, (cx, cy), 10, (0, 0, 255), -1)
    if paper and 'contour' in paper:
        cv2.drawContours(dbg, [paper['contour']], -1, (255, 0, 0), 2)
    dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ink_webcam_debug.png')
    cv2.imwrite(dp, dbg)

    pct = (bw * bh) / (w * h) * 100
    print(f"   ✅ 墨迹: ({x},{y}) {bw}x{bh}, 占画面 {pct:.1f}%")
    print(f"      中心: ({cx},{cy}), 轮廓数: {len(valid)}")
    print(f"      调试图: {dp}")

    return {'x': x, 'y': y, 'w': bw, 'h': bh, 'cx': cx, 'cy': cy}


# ============================================================
# Webcam 专用：找干净的最后一帧
# ============================================================

def find_clean_last_frame_webcam(video_path):
    """
    从视频末尾向前搜索，找到 (a) 无手 且 (b) 非淡出（亮度足够）的帧。

    淡出检查很关键：若输入是已经处理过、末尾带 fade-out 的视频
    （比如重复加工旧 output），最后 ~1s 的帧都是淡到全黑的，
    肤色比例都是 0% 但内容也是黑的——必须跳过这些帧。

    采用「自适应亮度阈值」：先扫一遍记录每帧亮度，取搜索范围内的
    最大亮度作为参考，凡是低于 85% 最大亮度的判定为淡出区跳过。
    这样既能处理 xi.mp4 (max≈247) 也能处理 dao_v2.mp4 (max≈184)
    这种纸面本身偏暗的源。

    搜索范围 90 帧（约 3s）覆盖 fade + 部分静止画面区。
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    SEARCH_FRAMES = 90

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

    # Pass 2: 在合格帧中找肤色最少的（手最干净的）
    best_frame = None
    best_skin = 1.0
    best_offset = None
    for offset in sorted(qualifying):
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - offset)
        ret, frame = cap.read()
        if not ret:
            continue
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        skin_ratio = float(np.mean(skin_mask > 0))

        if skin_ratio < best_skin:
            best_skin = skin_ratio
            best_frame = frame.copy()
            best_offset = offset

        if skin_ratio < 0.01:
            break  # 完全无手，提前结束

    cap.release()
    if best_frame is not None:
        print(f"      找到干净帧: 倒数第 {best_offset} 帧, 肤色 {best_skin * 100:.2f}%, "
              f"亮度 {brightness_by_offset[best_offset]:.0f} (max {max_bright:.0f})")
        return best_frame

    print("      ⚠️  未找到合格帧，回退到最后一帧")
    return extract_frame(video_path, 'last')


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
                         generate_thumb=True):
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
        paper = detect_paper_region(input_path)

        # Step 3: 墨迹检测
        print("\n🔍 Step 3: 检测墨迹...")
        ink = detect_ink_webcam(input_path, paper)

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
        # 裁剪宽度不超过纸面宽度
        available_w = crop_x_max - crop['x']
        if crop['w'] > available_w:
            crop['w'] = available_w - (available_w % 2)  # ffmpeg 偶数

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
            vf_parts.append(build_color_correction_filter())
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
                vf_still.append(build_color_correction_filter())
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

        # Step 11: 淡出 + 字幕
        post_filters = []

        if fade_out > 0:
            print(f"\n🌑 Step 11: 淡出 ({fade_out}s)...")
            cur_info = get_video_info(current_video)
            fade_start = cur_info['duration'] - fade_out
            post_filters.append(f"fade=t=out:st={fade_start:.2f}:d={fade_out}")
            print(f"   淡出起始: {fade_start:.1f}s (总时长 {cur_info['duration']:.1f}s)")

        if tts_srt_path and os.path.exists(tts_srt_path):
            print(f"\n📝 叠加同步字幕（SRT）...")
            post_filters.append(build_srt_subtitle_filter(tts_srt_path, font_path, output_width, output_height))
        elif subtitle_text:
            print(f"\n📝 叠加字幕: \"{subtitle_text}\"")
            post_filters.append(build_subtitle_filter(subtitle_text, output_width, output_height, font_path))

        if post_filters:
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

        # 输出
        shutil.copy2(current_video, output_path)

        # 缩略图
        if generate_thumb:
            thumb_path = os.path.splitext(output_path)[0] + '_thumb.jpg'
            print(f"\n🖼️  生成缩略图...")
            if clean_frame_processed and os.path.exists(clean_frame_processed):
                ok = generate_calligraphy_thumbnail(clean_frame_processed, thumb_path,
                                                     output_width, output_height)
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
    )


if __name__ == '__main__':
    main()
