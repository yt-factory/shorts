#!/usr/bin/env python3
"""
极客禅·墨 — 小红书封面生成器

从书法缩略图 + 标题文字生成小红书风格封面图。
极简禅意美学：米白底、浓墨字、大量留白、衬线标题。

用法：
  # 单张
  python3 xhs_cover.py --thumb tea_thumb.jpg --char "茶" \\
      --title "赵州禅师只说三个字" -o tea_cover.jpg

  # 批量
  python3 xhs_cover.py --batch covers.json -o output_dir/

依赖：
  pip install Pillow
"""

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 常量
# ============================================================

VERSION = "4.0"

# 小红书推荐封面比例 3:4
COVER_WIDTH = 1242
COVER_HEIGHT = 1660

# 色彩
BG_COLOR = (245, 240, 235)       # #F5F0EB 米白底色
DOT_COLOR = (200, 192, 184)      # #C8C0B8 暖灰圆点
TITLE_COLOR = (45, 42, 38)       # 深棕黑
SUBTITLE_COLOR = (155, 148, 140) # 暖灰色
STAMP_COLOR = (180, 60, 50)      # 暗红印章（传统印泥色）

# 布局：以圆点为锚点，字往上长，文字往下长
DIVIDER_Y_RATIO = 0.52           # 圆点（视觉中心锚点）
CHAR_GAP_RATIO = 0.03            # 字底边到圆点的呼吸间距
TOP_MIN_RATIO = 0.10             # 字顶边不超出此位置

# 书法字占画面宽度比例
CHAR_FILL_WIDTH = 0.55

# 字体
SERIF_FONT = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
SANS_FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
SERIF_BOLD = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'
# TTC index: 0=JP, 1=KR, 2=SC, 3=TC, 4=HK
CJK_SC_INDEX = 2
# WSL Windows 字体路径（备选）
WIN_SIMKAI = '/mnt/c/Windows/Fonts/simkai.ttf'


# ============================================================
# 字体加载
# ============================================================

def _load_font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    """加载字体，支持 TTC 索引"""
    try:
        return ImageFont.truetype(path, size=size, index=index)
    except (OSError, IOError):
        return None


def load_title_font(size: int) -> ImageFont.FreeTypeFont:
    """加载标题字体（衬线体优先）"""
    for path in [SERIF_BOLD, SERIF_FONT]:
        f = _load_font(path, size, CJK_SC_INDEX)
        if f:
            return f
    f = _load_font(WIN_SIMKAI, size)
    if f:
        return f
    return ImageFont.load_default()


def load_subtitle_font(size: int) -> ImageFont.FreeTypeFont:
    """加载副标题字体（无衬线体优先）"""
    f = _load_font(SANS_FONT, size, CJK_SC_INDEX)
    if f:
        return f
    f = _load_font(SERIF_FONT, size, CJK_SC_INDEX)
    if f:
        return f
    return ImageFont.load_default()


# ============================================================
# 书法字提取
# ============================================================

def extract_calligraphy(thumb_path: str, char: str = "") -> Image.Image:
    """
    从缩略图中提取纯净的书法字。

    流水线：
    1. 边缘清洗：四边各 8% 区域中的灰色像素推白（去桌面/纸边阴影）
    2. 背景清洗：Otsu 阈值，背景推白、墨迹增黑（保留自然笔触浓淡）
    3. 墨迹定位：在清洗后图上找字的紧边界框
    4. 裁出 + 呼吸空间
    5. 转 RGBA：白色→透明，墨迹→保留原始灰度（笔锋浓淡自然）+ 边缘羽化
    """
    import numpy as np

    import cv2

    img = Image.open(thumb_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    gray = np.mean(arr, axis=2)
    h, w = gray.shape

    # ══════════════════════════════════════════════
    # 两阶段策略：先"找纸"清桌面，再"找字"定位墨迹
    # ══════════════════════════════════════════════

    # ── Stage A: 白纸检测 (OPEN + 填充) ──
    # OPEN 去除桌面条纹，填充恢复墨迹留下的孔洞
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    # OPEN 参数：kernel=11, iter=2 → 总侵蚀 22px
    # 足以消除桌面条纹（5-15px 宽），但保留字笔画周围的白纸（30px+）
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    for paper_thresh in [210, 200, 190]:
        _, pmask = cv2.threshold(gray_u8, paper_thresh, 255, cv2.THRESH_BINARY)
        pmask = cv2.morphologyEx(pmask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        contours_p, _ = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_p:
            continue
        biggest_p = max(contours_p, key=cv2.contourArea)
        area_ratio = cv2.contourArea(biggest_p) / (w * h)
        if not (0.15 < area_ratio < 0.90):
            continue
        paper_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(paper_mask, [biggest_p], -1, 255, -1)
        outside_pixels = gray[paper_mask == 0]
        # 判断外部是桌面还是纸+墨迹：桌面有大量灰色像素(80-200)，纸+墨几乎没有
        gray_zone = ((outside_pixels > 80) & (outside_pixels < 200)).mean() if outside_pixels.size > 0 else 0
        if gray_zone > 0.20:
            paper_mask = cv2.erode(paper_mask,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)),
                                   iterations=1)
            gray[paper_mask == 0] = 255.0
            gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
            break  # 只在确认桌面纹理时 break

    # ── Stage B: 找字 — 直接定位墨迹轮廓 ──
    _, ink_bin = cv2.threshold(gray_u8, 80, 255, cv2.THRESH_BINARY_INV)
    ink_bin = cv2.dilate(ink_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    contours, _ = cv2.findContours(ink_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        margin = min(w, h) // 6
        return img.crop((margin, margin, w - margin, h - margin)).convert('RGBA')

    # ── Step 2: 过滤 — 只保留"笔画级"轮廓 ──
    # 排除：桌面条纹（极扁，宽高比 > 8）、微小噪点（面积 < 50px）
    min_area = 50
    strokes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        # 排除桌面条纹：高宽高比 + 横跨画面大部分宽/高
        if aspect > 5 and (cw > w * 0.6 or ch > h * 0.6):
            continue
        strokes.append(c)

    if not strokes:
        margin = min(w, h) // 6
        return img.crop((margin, margin, w - margin, h - margin)).convert('RGBA')

    # ── Step 3: 空间聚类 — 找到主字群 ──
    # 以最大笔画为锚点，保留距离在 3× 半径内的笔画
    anchor = max(strokes, key=cv2.contourArea)
    ax, ay, aw, ah = cv2.boundingRect(anchor)
    acx, acy = ax + aw // 2, ay + ah // 2
    # 聚类半径：单字用紧半径避免桌面碎片，多字用宽半径连接多个字
    if len(char) > 1:
        radius = max(aw, ah) * 3
    else:
        radius = max(aw, ah) * 2.0

    clustered = []
    for c in strokes:
        bx, by, bw_c, bh_c = cv2.boundingRect(c)
        mcx = bx + bw_c // 2
        mcy = by + bh_c // 2
        if abs(mcx - acx) < radius and abs(mcy - acy) < radius:
            clustered.append(c)

    if not clustered:
        clustered = [anchor]

    # 合并边界框
    pts = np.vstack(clustered)
    x_min, y_min, bw, bh = cv2.boundingRect(pts)
    x_max = x_min + bw
    y_max = y_min + bh

    # ── Step 4: 裁出 + 12% 呼吸空间 + 边缘保护 ──
    pad_x = int(bw * 0.12)
    pad_y = int(bh * 0.12)
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    # 裁出原始灰度（Stage A 已清理桌面，无需二次清理）
    cropped = gray[y1:y2, x1:x2].copy()
    ch_h, ch_w = cropped.shape

    # ── Step 5: 转 RGBA ──
    alpha = np.clip(255.0 - cropped, 0, 255)
    alpha[alpha < 25] = 0

    # 边缘羽化
    feather = max(3, int(min(ch_w, ch_h) * 0.04))
    fade = np.ones((ch_h, ch_w), dtype=np.float32)
    ramp = np.linspace(0, 1, feather)
    fade[:feather, :] *= ramp[:, None]
    fade[ch_h - feather:, :] *= ramp[::-1, None]
    fade[:, :feather] *= ramp[None, :]
    fade[:, ch_w - feather:] *= ramp[None, ::-1]
    alpha *= fade
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    ink_gray = np.clip(cropped, 0, 255).astype(np.uint8)
    result = np.zeros((ch_h, ch_w, 4), dtype=np.uint8)
    result[:, :, 0] = ink_gray
    result[:, :, 1] = ink_gray
    result[:, :, 2] = ink_gray
    result[:, :, 3] = alpha

    return Image.fromarray(result, 'RGBA')


# ============================================================
# 墨迹渲染（简化版：信任输入，只做背景替换）
# ============================================================

def render_ink(gray: 'np.ndarray') -> 'np.ndarray':
    """
    简化渲染：背景替换为米白底色 + 可选轻微对比度增强。

    原则：信任输入。脚本是排版工具，不是图像处理工具。
    只做两件事：
    1. 灰度 > 200 → 替换为米白底色，阈值边界 3px 高斯羽化
    2. 如果墨迹不够黑（最暗 > 40），轻微线性加深 ×0.9
    """
    import numpy as np
    import cv2

    gray = gray.astype(np.float64)
    h, w = gray.shape
    bg_val = np.mean(BG_COLOR[:3])  # ~240

    # ── 1. 背景替换 + 羽化 ──
    # 生成 0-1 墨迹权重：灰度 ≤ 197 → 1.0（纯墨迹），≥ 203 → 0.0（纯背景）
    # 中间 6 级灰度（197-203）做线性过渡，再高斯模糊 3px 消除硬边
    ink_weight = np.clip((203.0 - gray) / 6.0, 0.0, 1.0)
    ink_weight_u8 = (ink_weight * 255).astype(np.uint8)
    ink_weight_smooth = cv2.GaussianBlur(ink_weight_u8, (7, 7), 1.5).astype(np.float64) / 255.0

    # 墨迹像素保持原值，背景像素替换为底色
    result = gray * ink_weight_smooth + bg_val * (1.0 - ink_weight_smooth)

    # ── 2. 轻微对比度增强（可选）──
    darkest = float(gray[ink_weight > 0.5].min()) if (ink_weight > 0.5).any() else 255
    if darkest > 40:
        # 墨色偏淡，轻微加深（仅对墨迹区域）
        enhanced = result * 0.9
        result = enhanced * ink_weight_smooth + result * (1.0 - ink_weight_smooth)

    result = np.clip(result, 0, 255)

    # RGB：暖黑色调 — 越黑的像素暖色偏移越大，白色区域不受影响
    darkness = 1.0 - (result / 255.0)
    r = np.clip(result + 8 * darkness, 0, 255).astype(np.uint8)
    g = np.clip(result + 2 * darkness, 0, 255).astype(np.uint8)
    b = np.clip(result - 3 * darkness, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


# ============================================================
# 封面生成
# ============================================================

def generate_cover(thumb_path: str,
                   char: str = "",
                   title: str = "",
                   subtitle: str = "极客禅·墨",
                   output_path: str = "cover.jpg",
                   cover_width: int = COVER_WIDTH,
                   cover_height: int = COVER_HEIGHT,
                   enable_texture: bool = True,
                   enable_stamp: bool = True):
    """生成小红书/Shorts 封面。"""
    import numpy as np

    # --- 提取书法字 ---
    print(f"   提取书法字...")
    calligraphy = extract_calligraphy(thumb_path, char)

    # Tight-crop
    alpha_arr = np.array(calligraphy)[:, :, 3]
    rows_vis = np.any(alpha_arr > 30, axis=1)
    cols_vis = np.any(alpha_arr > 30, axis=0)
    if rows_vis.any() and cols_vis.any():
        ty1, ty2 = np.where(rows_vis)[0][[0, -1]]
        tx1, tx2 = np.where(cols_vis)[0][[0, -1]]
        calligraphy = calligraphy.crop((tx1, ty1, tx2 + 1, ty2 + 1))

    cw, ch = calligraphy.size
    aspect = cw / max(ch, 1)
    # 单字 vs 多字：优先用 --char 长度判断，否则用宽高比
    if len(char) > 1:
        is_multi = True
    elif len(char) == 1:
        is_multi = False
    else:
        is_multi = aspect > 2.0 or aspect < 0.3
    mode = "多字" if is_multi else "单字"
    print(f"   墨迹尺寸: {cw}x{ch} (宽高比={aspect:.2f}, char='{char}' → {mode}模式)")

    # --- 锚点布局 ---
    divider_y = int(cover_height * DIVIDER_Y_RATIO)   # 52%
    char_gap = int(cover_height * CHAR_GAP_RATIO)      # 3%
    top_min = int(cover_height * TOP_MIN_RATIO)         # 10%
    char_bottom = divider_y - char_gap  # 字底边固定位置

    # 统一目标尺寸
    if is_multi:
        max_w = int(cover_width * 0.70)
        max_h = int(cover_height * 0.40)
    else:
        max_w = int(cover_width * 0.48)
        max_h = int(cover_height * 0.35)

    scale_w = max_w / cw
    scale_h = max_h / ch
    scale = min(scale_w, scale_h)
    target_w = int(cw * scale)
    target_h = int(ch * scale)

    # 如果字太高（顶边超出 top_min），缩小
    paste_y = char_bottom - target_h
    if paste_y < top_min:
        target_h = char_bottom - top_min
        scale = target_h / ch
        target_w = int(cw * scale)
        target_h = int(ch * scale)
        paste_y = char_bottom - target_h

    paste_x = (cover_width - target_w) // 2

    print(f"   缩放后: {target_w}x{target_h} (scale={scale:.2f})")
    print(f"   粘贴位置: ({paste_x}, {paste_y})")
    print(f"   字占画布宽度: {target_w / cover_width * 100:.0f}%")

    # --- 灰度合成 ---
    # 关键：不用 alpha 混合（会稀释墨色），而是直接保留原始灰度值
    # 只在 alpha 极低的边缘羽化区做过渡
    cal_arr = np.array(calligraphy)
    cal_alpha = cal_arr[:, :, 3].astype(np.float64) / 255.0
    cal_gray = np.mean(cal_arr[:, :, :3].astype(np.float64), axis=2)
    bg_gray_val = np.mean(BG_COLOR[:3])
    # alpha > 0.3 → 使用原始灰度（保留墨色）
    # alpha < 0.05 → 纯背景
    # 中间 → 平滑过渡
    blend = np.clip((cal_alpha - 0.05) / 0.25, 0.0, 1.0)
    composite_gray = cal_gray * blend + bg_gray_val * (1.0 - blend)

    gray_img = Image.fromarray(composite_gray.astype(np.uint8), 'L')
    gray_resized = gray_img.resize((target_w, target_h), Image.LANCZOS)

    # --- 画布 + 纸张纹理 ---
    canvas_gray = np.full((cover_height, cover_width), bg_gray_val, dtype=np.float64)

    # 升级1：纸张纹理（柔和高斯噪声模拟纸纤维起伏）
    if enable_texture:
        import cv2
        noise = np.random.normal(loc=bg_gray_val, scale=3,
                                 size=(cover_height, cover_width)).astype(np.uint8)
        noise = cv2.GaussianBlur(noise, (0, 0), 40)  # sigma=40 → 柔和起伏
        canvas_gray = canvas_gray * 0.94 + noise.astype(np.float64) * 0.06

    # 贴入墨迹
    gray_arr = np.array(gray_resized, dtype=np.float64)
    py1, py2 = max(0, paste_y), min(cover_height, paste_y + target_h)
    px1, px2 = max(0, paste_x), min(cover_width, paste_x + target_w)
    sy1, sy2 = py1 - paste_y, py1 - paste_y + (py2 - py1)
    sx1, sx2 = px1 - paste_x, px1 - paste_x + (px2 - px1)
    canvas_gray[py1:py2, px1:px2] = gray_arr[sy1:sy2, sx1:sx2]

    canvas_rgb = render_ink(canvas_gray)
    canvas = Image.fromarray(canvas_rgb, 'RGB')
    draw = ImageDraw.Draw(canvas)

    # --- 升级2：圆点分隔符（替代直线）---
    dot_r = 4
    draw.ellipse(
        [cover_width // 2 - dot_r, divider_y - dot_r,
         cover_width // 2 + dot_r, divider_y + dot_r],
        fill=DOT_COLOR
    )

    # --- 升级4：标题（加大字号）---
    title_y = divider_y + int(cover_height * 0.06)
    if title:
        base_size = int(cover_width * 0.058)  # ~72px（之前 0.045 ≈ 56px）
        if len(title) > 12:
            base_size = int(base_size * 0.85)
        if len(title) > 18:
            base_size = int(base_size * 0.85)

        title_font = load_title_font(base_size)
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        tx = (cover_width - tw) // 2
        draw.text((tx, title_y), title, fill=TITLE_COLOR, font=title_font)

    # --- 升级3：副标题（简化为"极客禅·墨"）---
    sub_y = title_y + int(cover_height * 0.08)
    if subtitle:
        sub_size = int(cover_width * 0.028)
        sub_font = load_subtitle_font(sub_size)
        bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw = bbox[2] - bbox[0]
        sx = (cover_width - sw) // 2
        draw.text((sx, sub_y), subtitle, fill=SUBTITLE_COLOR, font=sub_font)

    # --- 升级5：品牌印章（右下角"禅"字红印）---
    if enable_stamp:
        stamp_size = int(cover_width * 0.04)
        stamp_margin = int(cover_width * 0.05)
        stamp_x = cover_width - stamp_margin - stamp_size
        stamp_y = cover_height - stamp_margin - stamp_size

        # 在 RGBA 临时图层上绘制印章（75% 透明度）
        stamp_layer = Image.new('RGBA', (cover_width, cover_height), (0, 0, 0, 0))
        stamp_draw = ImageDraw.Draw(stamp_layer)
        stamp_alpha = 190  # 75%

        # 方框
        stamp_draw.rectangle(
            [stamp_x, stamp_y, stamp_x + stamp_size, stamp_y + stamp_size],
            outline=(*STAMP_COLOR, stamp_alpha), width=2
        )

        # 框内"禅"字
        stamp_font_size = int(stamp_size * 0.65)
        stamp_font = _load_font(SERIF_FONT, stamp_font_size, CJK_SC_INDEX)
        if not stamp_font:
            stamp_font = _load_font(SANS_FONT, stamp_font_size, CJK_SC_INDEX)
        if stamp_font:
            bbox = stamp_draw.textbbox((0, 0), "禅", font=stamp_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = stamp_x + (stamp_size - tw) // 2
            ty = stamp_y + (stamp_size - th) // 2 - bbox[1]
            stamp_draw.text((tx, ty), "禅",
                            fill=(*STAMP_COLOR, stamp_alpha), font=stamp_font)

        canvas = Image.alpha_composite(canvas.convert('RGBA'), stamp_layer).convert('RGB')

    # --- 保存 ---
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path, 'JPEG', quality=95)
    print(f"   ✅ {output_path} ({cover_width}x{cover_height})")


# ============================================================
# 批量模式
# ============================================================

def batch_generate(config_path: str, output_dir: str,
                   cover_width: int = COVER_WIDTH,
                   cover_height: int = COVER_HEIGHT):
    """
    从 JSON 配置文件批量生成封面。

    JSON 格式:
    [
      {"thumb": "tea_thumb.jpg", "char": "茶", "title": "赵州禅师只说三个字"},
      {"thumb": "kong_thumb.jpg", "char": "空", "title": "空不是没有"}
    ]
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    for i, item in enumerate(items):
        thumb = item['thumb']
        # 相对路径基于配置文件所在目录
        if not os.path.isabs(thumb):
            thumb = os.path.join(config_dir, thumb)

        char = item.get('char', '')
        title = item.get('title', '')
        subtitle = item.get('subtitle', '极客禅·墨')
        out_name = item.get('output', f"{char or f'cover_{i}'}_cover.jpg")
        out_path = os.path.join(output_dir, out_name)

        print(f"\n[{i+1}/{len(items)}] {char} — {title}")
        generate_cover(
            thumb_path=thumb, char=char, title=title,
            subtitle=subtitle, output_path=out_path,
            cover_width=cover_width, cover_height=cover_height,
        )

    print(f"\n✅ 批量完成: {len(items)} 张封面 → {output_dir}/")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=f'极客禅·墨 小红书封面生成器 v{VERSION}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张封面
  python3 xhs_cover.py --thumb tea_thumb.jpg --char "茶" \\
      --title "赵州禅师只说三个字" -o tea_cover.jpg

  # 批量生成
  python3 xhs_cover.py --batch covers.json -o output_dir/

  # 无印章/无纹理
  python3 xhs_cover.py --thumb tea_thumb.jpg --char "茶" --no-stamp --no-texture -o clean.jpg
        """
    )

    parser.add_argument('--thumb', help='书法缩略图路径')
    parser.add_argument('--char', default='', help='书法字内容')
    parser.add_argument('--title', default='', help='标题文字')
    parser.add_argument('--subtitle', default='极客禅·墨', help='副标题 (默认: 极客禅·墨)')
    parser.add_argument('--batch', help='批量配置 JSON 文件路径')
    parser.add_argument('--output', '-o', default='cover.jpg', help='输出路径')
    parser.add_argument('--width', type=int, default=COVER_WIDTH)
    parser.add_argument('--height', type=int, default=COVER_HEIGHT)
    parser.add_argument('--no-stamp', action='store_true', help='不加印章')
    parser.add_argument('--no-texture', action='store_true', help='不加纸张纹理')
    args = parser.parse_args()

    if args.batch:
        if not os.path.exists(args.batch):
            print(f"❌ 找不到: {args.batch}")
            sys.exit(1)
        batch_generate(args.batch, args.output, args.width, args.height)
    elif args.thumb:
        if not os.path.exists(args.thumb):
            print(f"❌ 找不到: {args.thumb}")
            sys.exit(1)
        generate_cover(
            thumb_path=args.thumb, char=args.char,
            title=args.title, subtitle=args.subtitle,
            output_path=args.output,
            cover_width=args.width, cover_height=args.height,
            enable_texture=not args.no_texture,
            enable_stamp=not args.no_stamp,
        )
    else:
        parser.print_help()
        print("\n❌ 请提供 --thumb（单张模式）或 --batch（批量模式）")
        sys.exit(1)


if __name__ == '__main__':
    main()
