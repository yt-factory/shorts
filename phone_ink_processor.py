#!/usr/bin/env python3
"""
极客禅·墨 — Phone 视频处理器 v1.0

手机贴近纸拍摄：画面基本就是纸（~99% 覆盖）、字信号强（对比度 80+ 灰度级）、
无旋转 metadata（像素已按字正朝上排列）。

vs webcam_ink_processor.py 的核心差异：
  - 不做 Otsu 大块查找：画面就是纸，sanity check 式检测
  - 支持 --rotate 参数（手机横拍时可能需要 cw/ccw/180）
  - 对强信号场景优化：更大 fill_ratio、面积阈值上调、呼吸空间下调
  - 假设视频 setup 相对理想，管线更简单

处理流水线：
  ① (可选) rotate 预处理                     -- 新
  ② 纸面 sanity check（而非 Otsu 大块查找）  -- 新
  ③ 笔迹检测（强信号，面积 >500 即可）       -- 调整
  ④ 裁剪居中、放大（字占比 0.75 ~ 0.8）
  ⑤-⑫ 同 webcam 版本（色彩、锐化、淡入、静止、TTS、字幕、封面）
"""

import argparse
import sys
import os
import tempfile
import shutil

# 共享图像处理（平场校正、背景差分、介质分类、精确去线）
import ink_extraction as ie

# 复用公共工具函数
from ink_video_processor import (
    check_dependencies, get_video_info, get_audio_duration,
    run_ffmpeg, extract_frame, extract_last_frame,
    create_still_video, concat_videos,
    build_color_correction_filter, build_sharpen_filter,
    build_subtitle_filter, build_srt_subtitle_filter,
    normalize_audio, generate_tts, sample_paper_brightness,
    generate_thumbnail, generate_calligraphy_thumbnail, calculate_crop,
    DEFAULT_OUTPUT_WIDTH, DEFAULT_OUTPUT_HEIGHT,
    DEFAULT_HOLD_SECONDS, DEFAULT_FADE_IN, DEFAULT_FADE_OUT,
    TARGET_LUFS, SHORTS_MAX_DURATION, DEFAULT_FONT,
    DEFAULT_TTS_VOICE,
)


VERSION = "1.0"

# 品牌印章：末尾淡入、随整体淡出消失
STAMP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'files', 'seal', 'chan_seal.png')

# Phone 场景覆盖面积 > 90%（几乎就是纸），字信号强，填得更满
# webcam 默认 0.60 — phone 因为 bbox 精度高、字信号强，可以到 0.75
PHONE_FILL_RATIO = 0.75

# Phone 场景字大，不同于 webcam 的 100px
PHONE_MIN_INK_AREA = 500

# 呼吸空间 15%（vs webcam 25%）— 字已占画面较大部分，多余 padding 会过度缩字
PHONE_BREATHING = 0.15


# ============================================================
# Phone 专用：旋转预处理（替代 webcam 没有的步骤）
# ============================================================

def rotate_video_if_needed(video_path: str, rotate: str, tmpdir: str) -> str:
    """
    手机横拍、倒拍或拍反了时用此步骤把画面摆正。

    vs webcam: webcam 朝向固定，不需要此步骤。

    Args:
        video_path: 原始视频路径
        rotate: 'none' | 'cw' | 'ccw' | '180'
        tmpdir: 临时目录（存放 rotated.mp4）

    Returns:
        预处理后的视频路径（none 时即原路径）
    """
    if rotate == 'none':
        return video_path

    # ffmpeg transpose 参考：https://ffmpeg.org/ffmpeg-filters.html#transpose-1
    #   transpose=1 → 顺时针 90°
    #   transpose=2 → 逆时针 90°
    #   两次 transpose=1 → 180°（比 hflip,vflip 保守但等价）
    vf = {
        'cw': 'transpose=1',
        'ccw': 'transpose=2',
        '180': 'transpose=1,transpose=1',
    }[rotate]

    out = os.path.join(tmpdir, 'rotated.mp4')
    run_ffmpeg([
        'ffmpeg', '-y', '-i', video_path,
        '-vf', vf,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-an',
        out
    ], f"视频 rotate ({rotate})")
    print(f"   ✅ 已 rotate: {rotate}")
    return out


# ============================================================
# Phone 专用：纸面 sanity check（替代 webcam 的 Otsu 大块查找）
# ============================================================

def detect_paper_phone(video_path: str, debug: bool = False) -> dict:
    """
    手机场景：画面基本就是纸。此处做 sanity check + 异物剥离，
    **不找纸边**（通常不存在纸边）、**不用 Otsu**（会把纸内明暗切开）。

    vs webcam 的 detect_paper_region:
        webcam 场景：纸只占画面 ~50%、外周是桌面，必须用 Otsu + 连通域找最大亮块。
        phone 场景：纸占画面 ~99%，判断「整体是纸」即可。

    阈值基于 she.mp4 实测（paper mean≈158, std≈7.6）校准，**比初始设计的
    180/30 更宽松**——实际手机自动白平衡可能把白纸压到 150-170 灰度。
    关键判别信号是 std 很小（<25），说明画面只有一个主导灰度（纸面）。

    Returns:
        dict: {
            'mode': 'full-frame' | 'with-obstruction' | 'center-fallback',
            'mask': np.ndarray (H, W) uint8 0/255,  # 被认为是纸的像素
            'confidence': 'high' | 'medium' | 'low',
            'stats': {'mean', 'std'},
        }
    """
    import cv2
    import numpy as np

    frame = extract_frame(video_path, 'last')
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean = float(np.mean(gray))
    std = float(np.std(gray))

    # Case 1: 整体就是纸（最常见路径）
    # 宽松阈值：mean > 140（覆盖 she.mp4 的 158），std < 25（纸面均匀）
    if mean > 140 and std < 25:
        mask = np.full((h, w), 255, dtype=np.uint8)
        print(f"   ✅ 全纸模式: mean={mean:.0f}, std={std:.0f} (整画面视为纸)")
        return {
            'mode': 'full-frame',
            'mask': mask,
            'confidence': 'high',
            'stats': {'mean': mean, 'std': std},
        }

    # Case 2: 大体是纸但有异物（手/笔尖/阴影残留）
    # 用「高于 mean-std 的像素」近似纸面，然后 MORPH_CLOSE 把字/笔画的暗洞回填
    if mean > 120:
        paper_pixels = (gray > mean - std).astype(np.uint8) * 255
        # 31×31 闭运算：字的笔画（1-10px 宽）会被回填到纸面
        paper_pixels = cv2.morphologyEx(
            paper_pixels, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        )
        coverage = float(np.mean(paper_pixels > 0))
        if coverage > 0.80:
            print(f"   ⚠️  有异物模式: mean={mean:.0f}, std={std:.0f}, 纸面覆盖 {coverage*100:.0f}%")
            if debug:
                dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phone_paper_debug.png')
                cv2.imwrite(dp, paper_pixels)
                print(f"      调试图: {dp}")
            return {
                'mode': 'with-obstruction',
                'mask': paper_pixels,
                'confidence': 'medium',
                'stats': {'mean': mean, 'std': std},
            }

    # Case 3: 画面异常（拍摄失败或 setup 不对）
    cx, cy = w // 2, h // 2
    dw, dh = int(w * 0.7), int(h * 0.7)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[cy - dh // 2:cy + dh // 2, cx - dw // 2:cx + dw // 2] = 255
    print(f"   ⚠️  画面异常: mean={mean:.0f}, std={std:.0f}")
    print(f"      建议检查：是否对准纸？是否有大片阴影？")
    print(f"      退回中心 70% 区域")
    return {
        'mode': 'center-fallback',
        'mask': mask,
        'confidence': 'low',
        'stats': {'mean': mean, 'std': std},
    }


# ============================================================
# Phone 专用：笔迹检测（简化自 webcam 版）
# ============================================================

# 介质阈值（与 webcam 保持一致）
MEDIUM_THRESHOLDS = {'brush': 80, 'pencil': 170}


def _parse_char_region(spec: str, frame_w: int, frame_h: int) -> dict:
    """解析 --char-region 字符串（与 webcam 版一致）。

    支持格式：
      'cx,cy'          → 中心 + 画面 1/4 自动尺寸
      'cx,cy,w,h'      → 完整矩形（中心 + 尺寸）
      'x:y:w:h'        → 左上角 + 尺寸
    数值 <1 视为画面比例，≥1 视为像素。
    """
    s = spec.strip()
    sep = ':' if ':' in s else ','
    parts = [p.strip() for p in s.split(sep)]
    if len(parts) not in (2, 4):
        raise ValueError(f"--char-region 需 2 或 4 个值，got: {spec}")

    def _resolve(v, axis):
        v = float(v)
        if 0 < v < 1:
            return int(v * (frame_w if axis == 'x' else frame_h))
        return int(v)

    if len(parts) == 2:
        cx = _resolve(parts[0], 'x')
        cy = _resolve(parts[1], 'y')
        bw, bh = frame_w // 4, frame_h // 4
    else:
        if sep == ':':
            x = _resolve(parts[0], 'x')
            y = _resolve(parts[1], 'y')
            bw = _resolve(parts[2], 'x')
            bh = _resolve(parts[3], 'y')
            cx = x + bw // 2
            cy = y + bh // 2
        else:
            cx = _resolve(parts[0], 'x')
            cy = _resolve(parts[1], 'y')
            bw = _resolve(parts[2], 'x')
            bh = _resolve(parts[3], 'y')
    x = max(0, cx - bw // 2)
    y = max(0, cy - bh // 2)
    bw = min(bw, frame_w - x)
    bh = min(bh, frame_h - y)
    return {'x': x, 'y': y, 'w': bw, 'h': bh, 'cx': cx, 'cy': cy}


def detect_ink_phone(video_path, paper, debug=False, medium='auto', char_region=None):
    """
    手机场景的笔迹检测（简化自 webcam 版）。

    vs webcam 的 detect_ink_webcam:
      - analysis_mask 直接用 paper['mask']（多数情况就是整图），不再需要精细 erode
      - 面积阈值提高到 PHONE_MIN_INK_AREA（500 vs webcam 100）—— 字大、信号强
      - 呼吸空间 PHONE_BREATHING（15% vs webcam 25%）—— 字已占画面较大份额
      - skin_mask 仍然保留：手指/手背阴影虽不常出现但偶有

    Returns:
        (ink_dict, detected_medium)
    """
    import cv2
    import numpy as np

    frame = extract_frame(video_path, 'last')
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 手动指定字符区域 — 跳过自动检测
    if char_region:
        ink = _parse_char_region(char_region, w, h)
        print(f"   ✅ 手动指定区域: ({ink['x']},{ink['y']}) {ink['w']}x{ink['h']}")
        return ink, (medium if medium in ('brush', 'pencil') else 'pencil')

    # analysis_mask：手机场景直接用 paper mask（多数是整图）
    # 不做 erode（webcam 是为了避开纸边阴影；phone 没有纸边）
    analysis_mask = paper['mask']
    a_left, a_top = 0, 0
    a_right, a_bot = w, h
    # 对 with-obstruction / center-fallback 模式，用 mask 的外接矩形收紧搜索范围
    if paper['mode'] != 'full-frame':
        ys, xs = np.where(analysis_mask > 0)
        if len(xs) > 0:
            a_left, a_right = int(xs.min()), int(xs.max())
            a_top, a_bot = int(ys.min()), int(ys.max())

    # 肤色遮罩（手指阴影排除）
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb,
                            np.array([0, 133, 77], dtype=np.uint8),
                            np.array([255, 173, 127], dtype=np.uint8))
    skin_mask = cv2.dilate(skin_mask,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
                           iterations=2)
    not_skin = cv2.bitwise_not(skin_mask)

    # 平场校正（在强信号场景 redundant 但一致性更好）
    try:
        flat_gray = ie.flat_field_correct(gray)
        if float(np.median(flat_gray[analysis_mask > 0])) < 130:
            print("   ⚠️  平场校正结果异常，回退原始灰度")
            flat_gray = gray
    except Exception as _e:
        print(f"   ⚠️  平场校正失败 ({_e})，回退原始灰度")
        flat_gray = gray

    def _brush_pass():
        _, im = cv2.threshold(flat_gray, MEDIUM_THRESHOLDS['brush'], 255, cv2.THRESH_BINARY_INV)
        im = cv2.bitwise_and(im, analysis_mask)
        im = cv2.bitwise_and(im, not_skin)
        im = cv2.dilate(im, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        cs, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cs, im

    def _brush_success(cs):
        """毛笔真墨色校验——阴影不应通过（中位数 <60）。"""
        for c in cs:
            if cv2.contourArea(c) <= PHONE_MIN_INK_AREA * 2:
                continue
            mask = np.zeros(flat_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            vals = flat_gray[mask > 0]
            if vals.size and float(np.median(vals)) < 60:
                return True
        return False

    def _pencil_pass():
        # 与 webcam 同样的并联策略：bg_subtract ∪ 简单阈值
        im_bg = ie.background_subtract_mask(flat_gray, blur_ksize=5, min_threshold=8)
        _, im_thresh = cv2.threshold(flat_gray, 200, 255, cv2.THRESH_BINARY_INV)
        im = cv2.bitwise_or(im_bg, im_thresh)
        im = cv2.bitwise_and(im, analysis_mask)
        im = cv2.bitwise_and(im, not_skin)
        im = ie.remove_ruled_lines(im, paper_mask=analysis_mask,
                                   h_lines='auto', v_lines='auto')
        # 单像素噪点过滤
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(im, connectivity=8)
        if n_lbl > 1:
            keep = np.zeros(n_lbl, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 2
            im = np.where(keep[labels], 255, 0).astype(np.uint8)
        # 密集区聚类——phone 场景字大，cluster_kernel 可以略大
        cluster_kernel = max(25, min(h, w) // 80)
        merged = cv2.dilate(im, cv2.getStructuringElement(
            cv2.MORPH_RECT, (cluster_kernel, cluster_kernel)), iterations=2)
        big_cs, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if big_cs:
            def _block_score(c):
                bx0, by0, cw_, ch_ = cv2.boundingRect(c)
                aspect = max(cw_, ch_) / max(1, min(cw_, ch_))
                mask_in = int((im[by0:by0 + ch_, bx0:bx0 + cw_] > 0).sum())
                return mask_in / max(1, aspect ** 1.5)
            best = max(big_cs, key=_block_score)
            bx, by, bw_, bh_ = cv2.boundingRect(best)
            pad = max(60, max(bw_, bh_) // 4)
            focus = np.zeros_like(im)
            focus[max(0, by - pad):by + bh_ + pad,
                  max(0, bx - pad):bx + bw_ + pad] = 255
            im = cv2.bitwise_and(im, focus)
            synth = np.array([
                [[bx, by]], [[bx + bw_, by]],
                [[bx + bw_, by + bh_]], [[bx, by + bh_]]
            ], dtype=np.int32)
            return [synth], im
        cs, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cs, im

    # 介质分类
    if medium == 'auto':
        cls = ie.classify_medium(flat_gray, paper_mask=analysis_mask)
        if cls == 'empty':
            print("   ⚠️  画面近乎空白，无法检测笔迹")
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
        if not _brush_success(contours):
            if medium == 'brush':
                print("   ⚠️  指定 brush 但未检出真墨色，仍按 brush 处理")
            else:
                print("   ℹ️  brush 未通过真墨色校验，回退 pencil")
                contours, ink_mask = _pencil_pass()
                detected_medium = 'pencil'
    else:
        contours, ink_mask = _pencil_pass()

    if not contours:
        print("   ⚠️  未检测到笔迹，使用纸面中心")
        cx = (a_left + a_right) // 2
        cy = (a_top + a_bot) // 2
        return ({'x': cx - w // 8, 'y': cy - h // 8, 'w': w // 4, 'h': h // 4,
                 'cx': cx, 'cy': cy}, detected_medium)

    # 面积过滤（phone 版下限 500——字大、信号强）
    mn = max(20, PHONE_MIN_INK_AREA // 5) if detected_medium == 'pencil' else PHONE_MIN_INK_AREA
    mx = cv2.countNonZero(analysis_mask) * 0.30  # phone 字可占画面 15-25%，上限 30%

    # 纸面边缘伪影过滤（phone 场景通常无纸边，这段几乎不生效，保留防御）
    edge_margin = 30
    aspect_max = 2.5

    def _is_edge_artifact(c):
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
             and not _is_edge_artifact(c)]
    if not valid:
        valid = [max(contours, key=cv2.contourArea)]

    # 空间聚类（以最大轮廓为锚点）
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

    # 呼吸空间 15%（phone 专用，vs webcam 25%）
    px, py = int(bw * PHONE_BREATHING), int(bh * PHONE_BREATHING)
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
        dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ink_phone_debug.png')
        cv2.imwrite(dp, dbg)
        print(f"      调试图: {dp}")

    return {'x': x, 'y': y, 'w': bw, 'h': bh, 'cx': cx, 'cy': cy}, detected_medium


# ============================================================
# Phone 专用：找干净的最后一帧（逻辑复制自 webcam 版）
# ============================================================

def find_clean_last_frame_phone(video_path):
    """
    多帧中位数合成干净视频尾帧。逻辑与 webcam 版完全一致，
    仅名字区分避免混淆。

    为什么沿用：
      - 多帧合成对偶发抖动/手影一闪天然鲁棒
      - 相对排序（skin% 升序取前 8），不用绝对阈值
      - 手机场景手可能常在画面，但合成仍能把静态字符保留
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    SEARCH_FRAMES = 90
    TARGET_POOL = 8

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
    brightness_floor = max_bright * 0.85
    qualifying = [o for o, b in brightness_by_offset.items() if b >= brightness_floor]

    candidates: list = []
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
        print("      ⚠️  未找到亮度合格帧，回退最后一帧")
        return extract_frame(video_path, 'last')

    candidates.sort(key=lambda t: t[0])
    pool = candidates[:TARGET_POOL]

    if len(pool) < 3:
        skin, offset, frame = pool[0]
        print(f"      ⚠️  亮度合格帧仅 {len(pool)} 张，回退单帧 (倒数 {offset}, 肤色 {skin*100:.2f}%)")
        return frame

    stack = np.stack([f for _, _, f in pool], axis=0)
    composite = np.median(stack, axis=0).astype(np.uint8)
    offsets = [o for _, o, _ in pool]
    skins = [s for s, _, _ in pool]
    print(f"      合成池: {len(pool)} 帧 (末尾倒数 {min(offsets)}-{max(offsets)}), "
          f"肤色范围 {min(skins)*100:.1f}%-{max(skins)*100:.1f}%")
    return composite


# ============================================================
# Phone 专用：全视频搜索最佳封面源帧（逻辑复制自 webcam 版）
# ============================================================

def find_best_cover_frame_phone(video_path, paper=None, ink_bbox=None, sample_fps=1.5):
    """
    全视频扫描「字最完整 + 手最不挡字」的 k 帧，合成封面源。

    逻辑与 webcam 版 find_best_cover_frame 一致。手机场景的 skin% 数值
    通常比 webcam 更高（手机离手近），但因为用相对排序不受影响。

    Args:
        video_path: 视频路径
        paper: detect_paper_phone 的返回值（可选，用于限定 mask）
        ink_bbox: {'x','y','w','h'} 字符位置，用于定位「手挡字」指标
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / max(fps, 1e-6)
    step = max(1, int(fps / sample_fps))
    coarse = list(range(0, total, step))
    tail_start = max(0, total - 30)
    dense_tail = list(range(tail_start, total))
    sample_indices = sorted(set(coarse) | set(dense_tail))
    print(f"      全视频采样: {len(sample_indices)} 帧 "
          f"(粗采样 {len(coarse)} + 末尾密采样 {len(dense_tail)}; "
          f"总 {total} 帧, {duration:.1f}s @ {sample_fps} fps)")

    paper_mask = None
    if paper and paper.get('mode') != 'full-frame':
        # 只有 with-obstruction / center-fallback 模式才真正限定 mask
        paper_mask = paper.get('mask')

    skin_lo = np.array([0, 133, 77], dtype=np.uint8)
    skin_hi = np.array([255, 173, 127], dtype=np.uint8)

    if ink_bbox:
        cbx, cby = int(ink_bbox['x']), int(ink_bbox['y'])
        cbw, cbh = int(ink_bbox['w']), int(ink_bbox['h'])
    else:
        cbx = cby = cbw = cbh = None

    stats: list = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_, w_ = gray.shape
        if cbx is None:
            cbx = w_ // 3; cby = h_ // 3
            cbw = w_ // 3; cbh = h_ // 3
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, skin_lo, skin_hi)
        char_skin = skin[cby:cby + cbh, cbx:cbx + cbw]
        skin_over_char = float((char_skin > 0).mean()) if char_skin.size else 1.0
        flat = ie.flat_field_correct(gray)
        char_flat = flat[cby:cby + cbh, cbx:cbx + cbw]
        char_ink = int((char_flat < 200).sum()) if char_flat.size else 0
        stats.append((idx, skin_over_char, char_ink))

    if not stats:
        cap.release()
        return None

    MIN_CHAR_INK = 500
    meaningful = [s for s in stats if s[2] >= MIN_CHAR_INK]
    if not meaningful:
        cap.release()
        ink_max = max((s[2] for s in stats), default=0)
        print(f"      ⚠️  全视频字符 bbox 内暗像素 max={ink_max} <500，字未写")
        return None

    meaningful.sort(key=lambda s: s[1])
    pool_stats = meaningful[:8]
    print(f"      有字候选 {len(meaningful)} 帧 (char_ink≥{MIN_CHAR_INK}), "
          f"取 skin_over_char 最低前 {len(pool_stats)}")
    if len(pool_stats) < 3:
        cap.release()
        print(f"      ⚠️  可用候选仅 {len(pool_stats)} 帧，不足以合成")
        return None

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

def process_phone_video(input_path, output_path, voiceover_path=None,
                        target_fill_ratio=PHONE_FILL_RATIO,
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
                        char_region=None,
                        rotate='none'):
    """Phone 版主处理流程。结构沿用 webcam，仅替换 paper/ink 检测 + 加 rotate 步骤。"""

    print("=" * 60)
    print(f"极客禅·墨 Phone 处理器 v{VERSION}")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    clean_frame_processed = None

    try:
        # Step 0: rotate 预处理（phone 专有）
        if rotate != 'none':
            print(f"\n🔄 Step 0: 视频 rotate ({rotate})...")
        working_path = rotate_video_if_needed(input_path, rotate, tmpdir)

        # Step 1: 视频信息
        print("\n📹 Step 1: 分析视频...")
        vi = get_video_info(working_path)
        print(f"   尺寸: {vi['width']}x{vi['height']}, 时长: {vi['duration']:.1f}s, 帧率: {vi['fps']:.0f}fps")

        # Step 2: 纸面 sanity check（phone 专有）
        print("\n📄 Step 2: 纸面检测 (sanity check)...")
        paper = detect_paper_phone(working_path, debug=debug)

        # Step 2.5: 采样纸面亮度 → 色彩校正归一化（v4：根治灰纸 thumb/cover 级联失败）
        paper_p95 = sample_paper_brightness(working_path, paper['mask'], 'last')
        _gain_est = min(max(240.0 / max(paper_p95, 1.0), 0.8), 2.5)
        print(f"   📊 纸面归一化: p95={paper_p95:.0f} → 240 (gain≈{_gain_est:.2f})")

        # Step 3: 笔迹检测
        if char_region:
            print(f"\n🔍 Step 3: 用户指定字符区域 ({char_region})...")
        else:
            print(f"\n🔍 Step 3: 检测笔迹 (medium={medium})...")
        ink, detected_medium = detect_ink_phone(working_path, paper, debug=debug,
                                                medium=medium, char_region=char_region)

        # Step 4: 计算裁剪参数
        # Phone 场景：paper mask 通常是整图，crop 可以自由定位；
        # 仅当 obstruction/fallback 模式时才用 mask 外接矩形收紧
        if paper['mode'] == 'full-frame':
            crop_x_min, crop_x_max = 0, vi['width']
            crop_y_min, crop_y_max = 0, vi['height']
        else:
            import numpy as np
            ys, xs = np.where(paper['mask'] > 0)
            if len(xs) > 0:
                crop_x_min = int(xs.min())
                crop_x_max = int(xs.max())
                crop_y_min = int(ys.min())
                crop_y_max = int(ys.max())
            else:
                crop_x_min, crop_x_max = 0, vi['width']
                crop_y_min, crop_y_max = 0, vi['height']

        crop = calculate_crop(
            ink, vi['width'], vi['height'],
            target_fill_ratio, output_width, output_height,
            y_min=crop_y_min, y_max=crop_y_max,
        )
        if crop['x'] < crop_x_min:
            crop['x'] = crop_x_min
        if crop['x'] + crop['w'] > crop_x_max:
            crop['x'] = max(crop_x_min, crop_x_max - crop['w'])
        available_w = crop_x_max - crop['x']
        if crop['w'] > available_w:
            crop['w'] = available_w - (available_w % 2)
            target_aspect = output_width / output_height
            crop['h'] = int(crop['w'] / target_aspect)
            crop['h'] -= crop['h'] % 2
            crop['y'] = max(crop_y_min, min(ink['cy'] - crop['h'] // 2, crop_y_max - crop['h']))
            crop['scale_factor'] = output_width / crop['w']

        sf = crop['scale_factor']
        print(f"\n✂️  Step 4: 裁剪参数")
        print(f"   区域: ({crop['x']},{crop['y']}) {crop['w']}x{crop['h']}")
        print(f"   放大: {sf:.2f}x, 目标字占比: ~{target_fill_ratio * 100:.0f}%")

        # Step 5-7: 视频处理
        print(f"\n🎬 Step 5-7: 视频处理...")
        vf_parts = [
            f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}",
            f"scale={output_width}:{output_height}:flags=lanczos",
        ]
        print(f"   ✓ 裁剪+缩放 {output_width}x{output_height}")
        if enable_color_correct:
            vf_parts.append(build_color_correction_filter(paper_p95))
            print("   ✓ 白平衡校正")
        sharp = build_sharpen_filter(sf)
        vf_parts.append(sharp)
        print(f"   ✓ 锐化 ({sharp.split('=')[0]})")
        if fade_in > 0:
            vf_parts.append(f"fade=t=in:st=0:d={fade_in}")
            print(f"   ✓ 淡入 {fade_in}s")

        cropped_video = os.path.join(tmpdir, 'processed.mp4')
        run_ffmpeg([
            'ffmpeg', '-y', '-i', working_path,
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
            clean_frame = find_clean_last_frame_phone(working_path)
            clean_frame_raw = os.path.join(tmpdir, 'clean_raw.png')
            cv2.imwrite(clean_frame_raw, clean_frame)

            vf_still = [
                f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}",
                f"scale={output_width}:{output_height}:flags=lanczos",
            ]
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

        shutil.copy2(current_video, output_path)

        # 缩略图
        if generate_thumb:
            thumb_path = os.path.splitext(output_path)[0] + '_thumb.jpg'
            print(f"\n🖼️  生成缩略图...")
            print("   搜索最佳封面源帧（全视频）...")
            import cv2
            cover_src = find_best_cover_frame_phone(working_path, paper=paper, ink_bbox=ink)
            if cover_src is None:
                print("   ⚠️  全视频未找到干净封面帧，回退到 hold-still 的末尾合成帧")
                cover_processed = clean_frame_processed
            else:
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
                # 铅笔封面用原始合成帧（未经 pencil-curves 处理）：
                # generate_calligraphy_thumbnail 的 pencil-curves 会把铅笔灰度
                # 从 100-180 压到 30-80（接近毛笔），导致 cover 失去铅笔感。
                # 保存一份 crop+scale+color+sharpen 但未 pencil-curves 的帧，
                # 供 xhs_cover_pencil.py 使用。
                cover_frame_path = os.path.splitext(output_path)[0] + '_cover_frame.png'
                shutil.copy2(cover_processed, cover_frame_path)
                print(f"   ✅ {cover_frame_path} (pencil cover 源帧)")
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

        steps = []
        if rotate != 'none':
            steps.append(f"🔄rotate({rotate})")
        steps.extend(["✂️裁剪", f"🔎放大{sf:.1f}x"])
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
        description=f'极客禅·墨 Phone 处理器 v{VERSION}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 phone_ink_processor.py input.mp4 -o output.mp4
  python3 phone_ink_processor.py input.mp4 -o output.mp4 --medium pencil
  python3 phone_ink_processor.py input.mp4 -o output.mp4 --rotate cw
  python3 phone_ink_processor.py input.mp4 -o output.mp4 --text "旁白文本"
        """
    )

    parser.add_argument('input', help='原始视频文件路径')
    parser.add_argument('--output', '-o', default='output.mp4', help='输出文件路径')
    parser.add_argument('--voiceover', '-v', help='旁白音频文件')

    parser.add_argument('--fill', '-f', type=float, default=PHONE_FILL_RATIO,
                        help=f'字占画面比例 (默认: {PHONE_FILL_RATIO})')
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
    parser.add_argument('--debug', action='store_true',
                        help='保存检测调试图 (phone_paper_debug.png, ink_phone_debug.png)')

    parser.add_argument('--medium', choices=['auto', 'brush', 'pencil'], default='auto',
                        help='书写介质 (默认: auto)')
    parser.add_argument('--char-region',
                        help='手动指定字符区域，跳过自动检测。格式同 webcam 版。')
    parser.add_argument('--rotate', choices=['none', 'cw', 'ccw', '180'], default='none',
                        help='视频 rotate 预处理 (默认: none，即不 rotate)')

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

    process_phone_video(
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
        rotate=args.rotate,
    )


if __name__ == '__main__':
    main()
