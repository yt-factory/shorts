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

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ============================================================
# 常量
# ============================================================

VERSION = "1.0"

# 小红书推荐封面比例 3:4
COVER_WIDTH = 1242
COVER_HEIGHT = 1660

# 米白底色（模拟宣纸质感，不是纯白）
BG_COLOR = (245, 240, 235)       # #F5F0EB
INK_COLOR = (30, 28, 26)         # 近纯黑，微暖
DIVIDER_COLOR = (200, 195, 188)  # 淡灰暖色分隔线
TITLE_COLOR = (45, 42, 38)       # 深棕黑
SUBTITLE_COLOR = (155, 148, 140) # 暖灰色

# 布局比例（从上到下）
TOP_MARGIN_RATIO = 0.13          # 顶部留白
CHAR_ZONE_RATIO = 0.45           # 书法字区域
DIVIDER_Y_RATIO = 0.62           # 分隔线位置
TITLE_Y_RATIO = 0.67             # 标题起始位置
SUBTITLE_Y_RATIO = 0.75          # 副标题位置
BOTTOM_MARGIN_RATIO = 0.10       # 底部留白

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

def extract_calligraphy(thumb_path: str) -> Image.Image:
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

    img = Image.open(thumb_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    gray = np.mean(arr, axis=2).copy()
    h, w = gray.shape

    # ── Step 1: 边缘清洗 ──
    # 四边各 8% 区域：非深色像素（灰度>100）全部推白
    # 目的：消除纸张边缘、桌面阴影，但保留真正的墨迹笔画
    edge = int(min(w, h) * 0.08)
    ink_thresh_edge = 100  # 只有 <100 的像素才算墨迹
    for region in [
        gray[:edge, :],           # 上
        gray[h - edge:, :],       # 下
        gray[:, :edge],           # 左
        gray[:, w - edge:],       # 右
    ]:
        region[region > ink_thresh_edge] = 255.0

    # ── Step 2: 背景清洗 ──
    # Otsu 自适应阈值
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(gray_u8.ravel(), bins=256, range=(0, 256))
    total = gray_u8.size
    sum_total = float(np.sum(np.arange(256) * hist))
    sum_bg, weight_bg = 0.0, 0
    max_var, threshold = 0.0, 128
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    threshold = max(100, min(180, threshold))

    # 背景→纯白，墨迹→增强对比但保留浓淡层次
    cleaned = gray.copy()
    bg_mask = cleaned > threshold
    cleaned[bg_mask] = 255.0
    # 墨迹区域：拉伸对比度，映射 [0, threshold] → [0, threshold*0.7]
    ink_pixels = ~bg_mask
    if ink_pixels.any():
        cleaned[ink_pixels] = cleaned[ink_pixels] * 0.7

    # ── Step 3: 墨迹定位 ──
    # 只检测真正的深色墨迹（< 80），忽略灰色纸张纹理
    ink_detect = cleaned < 80
    rows_with_ink = np.any(ink_detect, axis=1)
    cols_with_ink = np.any(ink_detect, axis=0)

    if not rows_with_ink.any():
        margin = min(w, h) // 6
        return img.crop((margin, margin, w - margin, h - margin)).convert('RGBA')

    y_min, y_max = np.where(rows_with_ink)[0][[0, -1]]
    x_min, x_max = np.where(cols_with_ink)[0][[0, -1]]

    # ── Step 4: 裁出 + 12% 呼吸空间 ──
    ink_w = x_max - x_min
    ink_h = y_max - y_min
    pad_x = int(ink_w * 0.12)
    pad_y = int(ink_h * 0.12)
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    cropped = cleaned[y1:y2, x1:x2]
    ch_h, ch_w = cropped.shape

    # ── Step 5: 转 RGBA ──
    # alpha：白色→透明，墨迹→不透明（保留自然浓淡）
    # 使用 cleaned 灰度值：越黑→alpha 越大
    alpha = np.clip(255.0 - cropped, 0, 255)
    alpha[alpha < 25] = 0  # 去微弱噪点

    # 边缘羽化：4 边各 4% 渐变到透明
    feather = max(3, int(min(ch_w, ch_h) * 0.04))
    fade = np.ones((ch_h, ch_w), dtype=np.float32)
    ramp = np.linspace(0, 1, feather)
    fade[:feather, :] *= ramp[:, None]
    fade[ch_h - feather:, :] *= ramp[::-1, None]
    fade[:, :feather] *= ramp[None, :]
    fade[:, ch_w - feather:] *= ramp[None, ::-1]
    alpha *= fade

    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    # RGB：使用清洗后的灰度值作为墨迹颜色（保留笔锋浓淡层次）
    ink_gray = np.clip(cropped, 0, 255).astype(np.uint8)
    result = np.zeros((ch_h, ch_w, 4), dtype=np.uint8)
    result[:, :, 0] = ink_gray
    result[:, :, 1] = ink_gray
    result[:, :, 2] = ink_gray
    result[:, :, 3] = alpha

    return Image.fromarray(result, 'RGBA')


# ============================================================
# 封面生成
# ============================================================

def generate_cover(thumb_path: str,
                   char: str = "",
                   title: str = "",
                   subtitle: str = "极客禅·墨",
                   output_path: str = "cover.jpg",
                   cover_width: int = COVER_WIDTH,
                   cover_height: int = COVER_HEIGHT):
    """
    生成一张小红书风格封面。

    Args:
        thumb_path: 书法缩略图路径
        char: 书法字内容（用于文件命名/日志，不叠加到图上）
        title: 标题文字
        subtitle: 副标题文字
        output_path: 输出路径
    """

    # --- 画布 ---
    canvas = Image.new('RGB', (cover_width, cover_height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # --- 提取书法字 ---
    print(f"   提取书法字...")
    calligraphy = extract_calligraphy(thumb_path)

    # Tight-crop：去掉透明像素的边距，只保留实际墨迹
    import numpy as np
    alpha_arr = np.array(calligraphy)[:, :, 3]
    rows_vis = np.any(alpha_arr > 30, axis=1)
    cols_vis = np.any(alpha_arr > 30, axis=0)
    if rows_vis.any() and cols_vis.any():
        ty1, ty2 = np.where(rows_vis)[0][[0, -1]]
        tx1, tx2 = np.where(cols_vis)[0][[0, -1]]
        calligraphy = calligraphy.crop((tx1, ty1, tx2 + 1, ty2 + 1))

    cw, ch = calligraphy.size
    print(f"   墨迹尺寸: {cw}x{ch}")

    # 强制缩放：宽度约束 vs 高度约束，取较小 scale
    area_top = int(cover_height * 0.12)
    area_bot = int(cover_height * 0.62)
    area_h = area_bot - area_top

    scale_w = (cover_width * CHAR_FILL_WIDTH) / cw
    scale_h = (area_h * 0.85) / ch  # 字最大占区域高度 85%
    scale = min(scale_w, scale_h)

    target_w = int(cw * scale)
    target_h = int(ch * scale)

    calligraphy_resized = calligraphy.resize(
        (target_w, target_h), Image.LANCZOS
    )

    # 强制居中于字区域 (y: 12%~62%)
    area_center_y = (area_top + area_bot) // 2
    paste_x = (cover_width - target_w) // 2
    paste_y = area_center_y - target_h // 2

    print(f"   缩放后: {target_w}x{target_h} (scale={scale:.2f})")
    print(f"   粘贴位置: ({paste_x}, {paste_y})")
    print(f"   字占画布宽度: {target_w / cover_width * 100:.0f}%")

    # 先画一层米白底作为书法字的背景（覆盖透明区域）
    bg_layer = Image.new('RGBA', (cover_width, cover_height), (*BG_COLOR, 255))
    bg_layer.paste(calligraphy_resized, (paste_x, paste_y), calligraphy_resized)
    canvas = Image.alpha_composite(
        canvas.convert('RGBA'), bg_layer
    ).convert('RGB')
    draw = ImageDraw.Draw(canvas)

    # --- 分隔线 ---
    divider_y = int(cover_height * DIVIDER_Y_RATIO)
    line_w = int(cover_width * 0.12)
    line_x = (cover_width - line_w) // 2
    draw.line(
        [(line_x, divider_y), (line_x + line_w, divider_y)],
        fill=DIVIDER_COLOR, width=2
    )

    # --- 标题 ---
    if title:
        # 动态字号：标题越长字越小
        base_size = int(cover_width * 0.045)  # ~56px
        if len(title) > 12:
            base_size = int(base_size * 0.85)
        if len(title) > 18:
            base_size = int(base_size * 0.85)

        title_font = load_title_font(base_size)
        title_y = int(cover_height * TITLE_Y_RATIO)

        # 居中
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        tx = (cover_width - tw) // 2
        draw.text((tx, title_y), title, fill=TITLE_COLOR, font=title_font)

    # --- 副标题 ---
    if subtitle:
        sub_size = int(cover_width * 0.028)  # ~35px
        sub_font = load_subtitle_font(sub_size)
        sub_y = int(cover_height * SUBTITLE_Y_RATIO)

        bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw = bbox[2] - bbox[0]
        sx = (cover_width - sw) // 2
        draw.text((sx, sub_y), subtitle, fill=SUBTITLE_COLOR, font=sub_font)

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

  # 含副标题
  python3 xhs_cover.py --thumb xi_thumb.jpg --char "息" \\
      --title "它是意识和无意识之间的桥" \\
      --subtitle "极客禅·墨｜程序员写书法" -o xi_cover.jpg

  # 批量生成
  python3 xhs_cover.py --batch covers.json -o output_dir/
        """
    )

    # 单张模式
    parser.add_argument('--thumb', help='书法缩略图路径')
    parser.add_argument('--char', default='', help='书法字内容（用于日志）')
    parser.add_argument('--title', default='', help='标题文字')
    parser.add_argument('--subtitle', default='极客禅·墨',
                        help='副标题文字 (默认: 极客禅·墨)')

    # 批量模式
    parser.add_argument('--batch', help='批量配置 JSON 文件路径')

    # 输出
    parser.add_argument('--output', '-o', default='cover.jpg',
                        help='输出路径（单张: 文件路径，批量: 目录路径）')

    # 尺寸
    parser.add_argument('--width', type=int, default=COVER_WIDTH,
                        help=f'封面宽度 (默认: {COVER_WIDTH})')
    parser.add_argument('--height', type=int, default=COVER_HEIGHT,
                        help=f'封面高度 (默认: {COVER_HEIGHT})')

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
        )
    else:
        parser.print_help()
        print("\n❌ 请提供 --thumb（单张模式）或 --batch（批量模式）")
        sys.exit(1)


if __name__ == '__main__':
    main()
