#!/usr/bin/env python3
"""
极客禅·墨 — 铅笔专用小红书封面生成器

方向 A 极简留白（日式素描本扉页风）：
- 纯米白底，无画布纹理噪声
- 字直接漂浮在米白底上，是唯一的视觉主体
- 石墨灰调保留铅笔质感（不强制拉到纯黑）
- 极致留白

vs xhs_cover.py（毛笔版）的设计差异：

  毛笔版 xhs_cover.py                 铅笔版 xhs_cover_pencil.py
  ───────────────────────────────      ───────────────────────────────
  高斯噪声纸纹理（GaussianBlur）       纯色背景，无纹理
  render_ink 多层渲染                   统一 ink_color + alpha 直接合成
  ink_weight + 动态范围拉伸             不需要——alpha 已包含浓淡
  CLAHE 增强铅笔弱信号                  不需要——直接从 flat 灰度翻转
  暖黑色调偏移 (R+8 G+2 B-3)           统一石墨灰 (45,42,40)

为什么铅笔版不用 render_ink：
  render_ink 的核心逻辑是「把原始灰度通过 ink_weight 分离出笔迹，再做动态范围
  拉伸 + 暖色偏移」。这套对毛笔很好（墨色浓郁需要保留），但对铅笔是负面的——
  会把石墨的自然灰调变成仿墨黑，把真实笔触的力度变化压成平面色块。

  铅笔版用统一 ink_color (45,42,40) 做 RGB、alpha 做浓淡。alpha 已经包含了所有
  笔触信息（重按=高 alpha=深色、轻触=低 alpha=淡灰），纸面噪声被 alpha=0 天然
  过滤掉。三层复杂度归零。

用法：
  python3 xhs_cover_pencil.py --thumb she_thumb.jpg --char "舍" \\
      --title "舍得之间" -o she_cover.jpg
"""

import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

import ink_extraction as ie

# 从毛笔版复用：字体加载链、共享常量、印章路径
from xhs_cover import (
    load_title_font,
    load_subtitle_font,
    BG_COLOR,
    STAMP_IMAGE,
    COVER_WIDTH,
    COVER_HEIGHT,
)

VERSION = "1.0"

# ============================================================
# 铅笔版色彩系统
# ============================================================

TITLE_COLOR = (60, 58, 55)        # 暖深灰标题（和笔画色呼应，不用纯黑）
SUBTITLE_COLOR = (140, 135, 130)  # 中灰副标题（更轻，层次分明）
DOT_COLOR = (180, 175, 170)       # 淡灰圆点

# ============================================================
# 布局常量
# ============================================================

CHAR_CENTER_Y_RATIO = 0.37        # 字中心：画面高度 37%（黄金分割偏上）
CHAR_GAP_RATIO = 0.03             # 字底边到圆点的呼吸间距
TOP_MIN_RATIO = 0.10              # 字顶边不超出此位置

MAX_W_SINGLE = 0.28               # 单字占画布宽度 28%
MAX_W_MULTI = 0.45                # 多字占画布宽度 45%
MAX_H = 0.28                      # 字高度上限 28%


# ============================================================
# 铅笔字提取
# ============================================================

def extract_pencil_calligraphy(thumb_path: str, char: str = "") -> Image.Image:
    """
    从 thumb 提取铅笔字 → RGB（纸面替换为米白，笔画保留原灰度）。

    两阶段流水线：
      Stage 1 — 自适应阈值定位字 bbox（对光照渐变鲁棒）
      Stage 2 — sigmoid 背景替换 + 轻微压暗（纸面→米白，笔画原封不动）

    不做任何"渲染"——不用 alpha 表浓淡、不用 ink_color、不做动态范围拉伸。
    笔画深浅完全由原始灰度值表现，石墨反光区自然显示为"笔画内部稍浅"。
    """
    import numpy as np
    import cv2

    img = Image.open(thumb_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    gray = np.mean(arr, axis=2)
    h, w = gray.shape
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    # ══════════════════════════════════════════════
    # Stage 1: 自适应阈值定位字
    # ══════════════════════════════════════════════
    # 自适应阈值对光照渐变天然鲁棒：只检测"局部比邻域暗 C 级以上"的像素。
    # 纸面纹理（局部均匀 ±10）→ 不触发；笔画（局部暗 60-200 级）→ 触发。
    # 不需要 flat_field_correct 作为前置。
    block = max(31, ((min(h, w) // 20) | 1))  # ~5% 图像尺寸，必须奇数
    ink_detect = cv2.adaptiveThreshold(
        gray_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block,
        C=12  # 比邻域暗 12 灰度级才算笔迹（raw frame 纸面干净 std<5，可以低）
    )

    # 去印刷格线
    ink_detect = ie.remove_ruled_lines(ink_detect, h_lines='auto', v_lines='auto')

    # 去散点噪声（开运算去单像素 + 连通域面积过滤去散斑）
    ink_detect = cv2.morphologyEx(
        ink_detect, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    # 连通域过滤：面积 + 空间距离双重策略
    # 面积 <200 → 一定是噪点，删
    # 面积 200-800 且远离字主体 → 噪点，删
    # 面积 200-800 但靠近字主体 → 字的细笔画碎片（如"口"），保留
    # 面积 >800 → 无条件保留
    n_labels, labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(
        ink_detect, connectivity=8)
    if n_labels > 1:
        areas = cc_stats[1:, cv2.CC_STAT_AREA]
        # 找最大连通域作为锚点
        anchor_idx = int(np.argmax(areas)) + 1  # +1 因为 label 0 是背景
        ax_cc = int(cc_stats[anchor_idx, cv2.CC_STAT_LEFT] + cc_stats[anchor_idx, cv2.CC_STAT_WIDTH] // 2)
        ay_cc = int(cc_stats[anchor_idx, cv2.CC_STAT_TOP] + cc_stats[anchor_idx, cv2.CC_STAT_HEIGHT] // 2)
        anchor_span = max(cc_stats[anchor_idx, cv2.CC_STAT_WIDTH],
                          cc_stats[anchor_idx, cv2.CC_STAT_HEIGHT])
        dist_radius = anchor_span * 2

        keep = np.zeros(n_labels, dtype=bool)
        for i in range(1, n_labels):
            area = int(cc_stats[i, cv2.CC_STAT_AREA])
            if area < 200:
                continue  # 太小，一定是噪点
            if area >= 800:
                keep[i] = True
                continue  # 大碎片，无条件保留
            # 中等碎片 (200-800)：检查距离锚点的距离
            cx_i = int(cc_centroids[i, 0])
            cy_i = int(cc_centroids[i, 1])
            if abs(cx_i - ax_cc) < dist_radius and abs(cy_i - ay_cc) < dist_radius:
                keep[i] = True  # 靠近字主体，保留
        ink_detect = np.where(keep[labels], 255, 0).astype(np.uint8)

    # 轻度膨胀连接笔画碎片
    ink_detect = cv2.dilate(
        ink_detect,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )

    contours, _ = cv2.findContours(
        ink_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("   ⚠️  未检测到笔迹")
        margin = min(w, h) // 6
        return img.crop((margin, margin, w - margin, h - margin)).convert('RGB')

    # 过滤碎片（面积阈值按帧尺寸缩放）
    min_area = max(100, w * h // 20000)
    strokes = [c for c in contours if cv2.contourArea(c) > min_area]
    if not strokes:
        strokes = [max(contours, key=cv2.contourArea)]

    # 空间聚类：以最大笔画为锚点
    anchor = max(strokes, key=cv2.contourArea)
    ax, ay, aw, ah = cv2.boundingRect(anchor)
    acx, acy = ax + aw // 2, ay + ah // 2
    radius = max(aw, ah) * (3 if len(char) > 1 else 2)

    clustered = [
        c for c in strokes
        if (abs(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] // 2 - acx) < radius
            and abs(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2 - acy) < radius)
    ]
    if not clustered:
        clustered = [anchor]

    pts = np.vstack(clustered)
    x_min, y_min, bw, bh = cv2.boundingRect(pts)

    # 呼吸空间 20%
    pad_x = int(bw * 0.20)
    pad_y = int(bh * 0.20)
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_min + bw + pad_x)
    y2 = min(h, y_min + bh + pad_y)

    # ══════════════════════════════════════════════
    # Stage 2: sigmoid 背景替换（最简方案）
    # ══════════════════════════════════════════════
    # 不做任何"渲染"——只把灰色纸面换成米白底色，笔画原封不动。
    # sigmoid 过渡让纸面→米白的替换边界柔和，不留切痕。
    # ×0.8 轻微压暗让铅笔灰调在米白底上更清晰（可调：0.85 更淡 / 0.75 更深）。
    cropped = gray[y1:y2, x1:x2].astype(np.float64)
    ch_h, ch_w = cropped.shape

    paper_val = float(np.percentile(cropped, 95))
    bg_r, bg_g, bg_b = float(BG_COLOR[0]), float(BG_COLOR[1]), float(BG_COLOR[2])
    print(f"   局部: paper_p95={paper_val:.0f}, bg=({bg_r:.0f},{bg_g:.0f},{bg_b:.0f})")

    # sigmoid 权重：纸面（亮）→ 0（用米白替换），笔画（暗）→ 1（保留原灰度）
    center = paper_val - 15
    width = 8.0
    weight = 1.0 / (1.0 + np.exp((cropped - center) / width))

    # 适度增强对比度：笔画灰度压暗 35%（raw frame 铅笔灰调偏淡，需足够压暗）
    darkened = cropped * 0.80

    # 每通道分别混合，确保纸面区域精确匹配 BG_COLOR（避免灰度近似导致色差方块）
    result_r = darkened * weight + bg_r * (1.0 - weight)
    result_g = darkened * weight + bg_g * (1.0 - weight)
    result_b = darkened * weight + bg_b * (1.0 - weight)

    # 灰度 snap：sigmoid 后接近 BG 的像素直接设为精确 BG（消除纸面纹理方块）
    # 阈值 45：纸面纹理最暗颗粒经 sigmoid+压暗后落在 195-210 区间
    # 铅笔笔画最浅 ~180 经 *0.8 → ~144，远低于 bg-45=195，安全
    result_mean = (result_r + result_g + result_b) / 3.0
    bg_mean = (bg_r + bg_g + bg_b) / 3.0
    near_bg = result_mean > (bg_mean - 45)
    result_r[near_bg] = bg_r
    result_g[near_bg] = bg_g
    result_b[near_bg] = bg_b

    # 去除 snap 后残余的孤立灰点（纸面纤维/灰尘）
    non_bg = (result_mean <= (bg_mean - 45)).astype(np.uint8) * 255
    cleaned = cv2.morphologyEx(non_bg, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    removed = (non_bg > 0) & (cleaned == 0)
    result_r[removed] = bg_r
    result_g[removed] = bg_g
    result_b[removed] = bg_b

    # bbox 边缘羽化：消除 sigmoid 处理区域和画布之间的灰度差"方块"边界
    feather_px = max(5, int(min(ch_h, ch_w) * 0.08))
    edge_blend = np.ones((ch_h, ch_w), dtype=np.float64)
    ramp = np.linspace(0, 1, feather_px)
    edge_blend[:feather_px, :] *= ramp[:, None]
    edge_blend[-feather_px:, :] *= ramp[::-1, None]
    edge_blend[:, :feather_px] *= ramp[None, :]
    edge_blend[:, -feather_px:] *= ramp[None, ::-1]
    result_r = result_r * edge_blend + bg_r * (1.0 - edge_blend)
    result_g = result_g * edge_blend + bg_g * (1.0 - edge_blend)
    result_b = result_b * edge_blend + bg_b * (1.0 - edge_blend)

    # 返回纯 RGB——不用 alpha，笔画浓淡直接由灰度值表现
    rgb = np.stack([
        np.clip(result_r, 0, 255).astype(np.uint8),
        np.clip(result_g, 0, 255).astype(np.uint8),
        np.clip(result_b, 0, 255).astype(np.uint8),
    ], axis=-1)
    print(f"   墨迹尺寸: {ch_w}x{ch_h}")
    return Image.fromarray(rgb, 'RGB')


# ============================================================
# 封面生成
# ============================================================

def generate_pencil_cover(
    thumb_path: str,
    char: str = "",
    title: str = "",
    subtitle: str = "极客禅 · 墨",
    output_path: str = "cover.jpg",
    cover_width: int = COVER_WIDTH,
    cover_height: int = COVER_HEIGHT,
    enable_stamp: bool = True,
) -> None:
    """
    生成铅笔风格小红书封面。

    vs xhs_cover.generate_cover 的核心区别：
    1. 无画布纹理噪声——纯米白底
    2. 纯 RGB 粘贴（sigmoid 背景替换，无 alpha / 无 render_ink）
    3. 字占比 25-30% 宽度（极致留白）
    4. 无"画中画"纸块边框——字漂浮在底上
    """
    import numpy as np

    # ── 提取字 ──
    print(f"   提取铅笔字...")
    calligraphy = extract_pencil_calligraphy(thumb_path, char)

    # ── tight crop：找到非米白像素的边界 ──
    cal_arr = np.array(calligraphy)
    cal_gray = np.mean(cal_arr[:, :, :3].astype(np.float64), axis=2)
    bg_thresh = float(np.mean(BG_COLOR[:3])) - 5  # 比米白暗 5 以上视为有笔迹
    rows = np.any(cal_gray < bg_thresh, axis=1)
    cols = np.any(cal_gray < bg_thresh, axis=0)
    if rows.any() and cols.any():
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        calligraphy = calligraphy.crop((x1, y1, x2 + 1, y2 + 1))

    cal_w, cal_h = calligraphy.size

    # ── 判断单字/多字 ──
    if len(char) > 1:
        is_multi = True
    elif len(char) == 1:
        is_multi = False
    else:
        aspect = cal_w / max(cal_h, 1)
        is_multi = aspect > 2.0 or aspect < 0.3
    mode = "多字" if is_multi else "单字"
    print(f"   墨迹裁切后: {cal_w}x{cal_h} (char='{char}' → {mode}模式)")

    # ── 缩放字 ──
    max_w = int(cover_width * (MAX_W_MULTI if is_multi else MAX_W_SINGLE))
    max_h = int(cover_height * MAX_H)

    scale = min(max_w / max(cal_w, 1), max_h / max(cal_h, 1))
    target_w = max(1, int(cal_w * scale))
    target_h = max(1, int(cal_h * scale))
    calligraphy = calligraphy.resize((target_w, target_h), Image.LANCZOS)

    pct = target_w / cover_width * 100
    print(f"   缩放后: {target_w}x{target_h} (scale={scale:.2f})")
    print(f"   字占画布宽度: {pct:.0f}%")

    # ── 创建纯米白画布（无纹理——关键区别！） ──
    canvas = Image.new('RGB', (cover_width, cover_height), BG_COLOR)

    # ── 布局定位 ──
    char_center_y = int(cover_height * CHAR_CENTER_Y_RATIO)
    paste_x = (cover_width - target_w) // 2
    paste_y = char_center_y - target_h // 2

    # 保护：字顶不超出 10%
    top_min = int(cover_height * TOP_MIN_RATIO)
    if paste_y < top_min:
        paste_y = top_min

    # ── 粘贴字（纯 RGB 直接覆盖，无 alpha） ──
    canvas.paste(calligraphy, (paste_x, paste_y))

    char_bottom = paste_y + target_h
    print(f"   粘贴位置: ({paste_x}, {paste_y}), 字底: y={char_bottom}")

    # ── 圆点分隔符 ──
    draw = ImageDraw.Draw(canvas)
    dot_y = char_bottom + int(cover_height * CHAR_GAP_RATIO)
    dot_r = 3
    dot_cx = cover_width // 2
    draw.ellipse(
        [dot_cx - dot_r, dot_y - dot_r, dot_cx + dot_r, dot_y + dot_r],
        fill=DOT_COLOR
    )

    # ── 标题 ──
    title_y = dot_y + int(cover_height * 0.03)
    if title:
        title_size = int(cover_height * 0.035)  # ~58px
        title_font = load_title_font(title_size)
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]

        # 标题过长自动缩小
        max_title_w = int(cover_width * 0.90)
        while tw > max_title_w and title_size > 28:
            title_size -= 2
            title_font = load_title_font(title_size)
            bbox = draw.textbbox((0, 0), title, font=title_font)
            tw = bbox[2] - bbox[0]

        tx = (cover_width - tw) // 2
        draw.text((tx, title_y), title, fill=TITLE_COLOR, font=title_font)

    # ── 副标题 ──
    if subtitle:
        sub_y = title_y + int(cover_height * 0.04)
        sub_size = int(cover_height * 0.02)  # ~33px
        sub_font = load_subtitle_font(sub_size)
        bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw = bbox[2] - bbox[0]
        sx = (cover_width - sw) // 2
        draw.text((sx, sub_y), subtitle, fill=SUBTITLE_COLOR, font=sub_font)

    # ── 品牌印章（右下角，和 xhs_cover.py 完全一致） ──
    if enable_stamp and os.path.exists(STAMP_IMAGE):
        stamp_src = Image.open(STAMP_IMAGE).convert('RGBA')
        # 宽度约 11%
        stamp_w = int(cover_width * 0.11)
        ratio = stamp_w / stamp_src.width
        stamp_h = int(stamp_src.height * ratio)
        stamp_resized = stamp_src.resize((stamp_w, stamp_h), Image.LANCZOS)

        # 整体透明度 80%（和毛笔版一致）
        r, g, b, a = stamp_resized.split()
        a = a.point(lambda x: int(x * 0.80))
        stamp_resized.putalpha(a)

        # 右下角定位
        margin_x = int(cover_width * 0.05)
        margin_y = int(cover_height * 0.04)
        stx = cover_width - margin_x - stamp_resized.width
        sty = cover_height - margin_y - stamp_resized.height

        stamp_layer = Image.new('RGBA', (cover_width, cover_height), (0, 0, 0, 0))
        stamp_layer.paste(stamp_resized, (stx, sty))
        canvas = Image.alpha_composite(
            canvas.convert('RGBA'), stamp_layer
        ).convert('RGB')

    # ── 保存 ──
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path, 'JPEG', quality=95)
    print(f"   ✅ {output_path} ({cover_width}x{cover_height})")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=f'极客禅·墨 铅笔封面生成器 v{VERSION}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张封面
  python3 xhs_cover_pencil.py --thumb she_thumb.jpg --char "舍" \\
      --title "舍得之间" -o she_cover.jpg

  # 无印章
  python3 xhs_cover_pencil.py --thumb she_thumb.jpg --char "舍" \\
      --no-stamp -o clean.jpg
        """
    )

    parser.add_argument('--thumb', required=True, help='书法缩略图路径')
    parser.add_argument('--char', default='', help='书法字内容')
    parser.add_argument('--title', default='', help='标题文字')
    parser.add_argument('--subtitle', default='极客禅 · 墨', help='副标题')
    parser.add_argument('--output', '-o', default='cover.jpg', help='输出路径')
    parser.add_argument('--width', type=int, default=COVER_WIDTH)
    parser.add_argument('--height', type=int, default=COVER_HEIGHT)
    parser.add_argument('--no-stamp', action='store_true', help='不加印章')

    args = parser.parse_args()

    if not os.path.exists(args.thumb):
        print(f"❌ 找不到: {args.thumb}")
        sys.exit(1)

    generate_pencil_cover(
        thumb_path=args.thumb,
        char=args.char,
        title=args.title,
        subtitle=args.subtitle,
        output_path=args.output,
        cover_width=args.width,
        cover_height=args.height,
        enable_stamp=not args.no_stamp,
    )


if __name__ == '__main__':
    main()
