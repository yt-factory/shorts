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

# 共享图像处理工具（平场校正、介质分类、去线等）
import ink_extraction as ie

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
STAMP_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', 'seal', 'chan_seal.png')

# 布局：v5 改为以「字中心」为锚点落在黄金分割点
# 旧版用 DIVIDER_Y_RATIO=0.52 作为字底边 + 圆点的固定位置，
# 字中心会随 target_h 漂移（方字中心高、宽字中心低）——视觉重心不稳。
# 新版定字中心 = 38%（视觉黄金分割），圆点 / 标题位置从字底边动态派生。
CHAR_CENTER_Y_RATIO = 0.38       # 字的视觉中心落在画面高度 38% 处（黄金分割）
CHAR_GAP_RATIO = 0.03            # 字底边到圆点的呼吸间距
TOP_MIN_RATIO = 0.10             # 字顶边不超出此位置（保护稀有超高字）

# 字体
SERIF_FONT = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
SANS_FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
SERIF_BOLD = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'
# TTC index: 0=JP, 1=KR, 2=SC, 3=TC, 4=HK
CJK_SC_INDEX = 2
# WSL Windows 字体路径（备选）
WIN_SIMKAI = '/mnt/c/Windows/Fonts/simkai.ttf'

# v5 审美精修：标题字体优先级
# 旧版只用 NotoSerifCJK，字形方正均匀，"PPT 标题"感强，和手写书法气质不搭。
# 新版优先霞鹜文楷 Bold（接近手写楷书），再降级到思源宋体 Heavy，最后落回 Noto。
# 每项是 (路径, TTC 子字体索引)；非 TTC 字体 index 填 0 即可。
TITLE_FONT_CANDIDATES = [
    ('/usr/share/fonts/truetype/lxgw-wenkai/LXGWWenKai-Bold.ttf', 0),
    ('/usr/share/fonts/opentype/source-han-serif/SourceHanSerifSC-Heavy.otf', 0),
    (SERIF_BOLD, CJK_SC_INDEX),
    (SERIF_FONT, CJK_SC_INDEX),
    (WIN_SIMKAI, 0),
]
_title_font_hint_shown = False


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
    """加载标题字体。按 TITLE_FONT_CANDIDATES 优先级尝试，首次未命中霞鹜文楷时
    打印一行提示（安装命令），此后静默避免日志噪音。"""
    global _title_font_hint_shown
    for i, (path, index) in enumerate(TITLE_FONT_CANDIDATES):
        if os.path.exists(path):
            f = _load_font(path, size, index)
            if f:
                # 命中非首选（非霞鹜文楷）时一次性提示升级方案
                if i > 0 and not _title_font_hint_shown:
                    print("   💡 标题字体提示：未找到霞鹜文楷，回退到 "
                          f"{os.path.basename(path)}。建议 sudo apt install fonts-lxgw-wenkai 获得手写感更强的封面。")
                    _title_font_hint_shown = True
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

def extract_calligraphy(thumb_path: str, char: str = "",
                        medium: str = 'auto') -> "tuple[Image.Image, str, str | None]":
    """
    从缩略图中提取纯净的书法字。

    流水线（v5 重构）：
    1. Stage A 白纸检测：原始灰度上找纸边轮廓（用于滤掉纸外噪声）
    2. **平场校正**：把纸面光照渐变抹平，flat_gray 中纸面 ≈ 255。
       同时供给 bbox 检测 AND alpha 生成——这是 alpha 不再带阴影的关键。
    3. Stage B 介质分类：看 flat_gray 上最暗 1% 像素的中位数，直接决定 brush/pencil
    4. 找字：brush 全局阈值 / pencil 全局阈值 + 去线
    5. 裁出 + 呼吸空间
    6. 转 RGBA：**alpha = 255 - flat_cropped**（不是原始 cropped），
       自动消除纸张渐变阴影

    Args:
        medium: 'auto' | 'brush' | 'pencil'
            - brush: 毛笔，墨色深
            - pencil: 铅笔，笔迹浅
            - auto: 直方图分类（最暗 1% 中位数 < 55 → brush，否则 pencil）

    Returns:
        (RGBA Image, detected_medium)
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

    # ── Stage A: 白纸检测 ──
    # 在原始灰度上找纸边，把非纸区域推白（防止桌面纹理被后续误判为笔迹）
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    paper_mask_global: "np.ndarray | None" = None
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
            paper_mask_global = paper_mask
            break  # 只在确认桌面纹理时 break

    # ── Stage A.5: 平场校正 ──
    # 核心改进：flat_gray 同时用于 bbox 检测 AND alpha 生成。
    # 旧实现 alpha = 255 - cropped 带纸张渐变阴影；
    # flat_gray 中纸面 ≈ 255 → alpha = 255 - flat_cropped 自动消除阴影。
    try:
        flat_gray_full = ie.flat_field_correct(gray_u8)
        check_mask = paper_mask_global if paper_mask_global is not None else (np.ones_like(gray_u8) * 255)
        if float(np.median(flat_gray_full[check_mask > 0])) < 150:
            print("   ⚠️  封面平场校正异常（纸面中位数 <150），回退原始灰度")
            flat_gray_full = gray_u8.copy()
    except Exception as _e:
        print(f"   ⚠️  封面平场校正失败 ({_e})，回退原始灰度")
        flat_gray_full = gray_u8.copy()

    # ── Stage B: 找字 — 在 flat_gray 上做简单全局阈值 ──
    # brush: 全局阈值 80（毛笔墨色深）
    # pencil: 平场校正后纸面 ≈ 255，直接用全局阈值 220 抓笔迹
    #         不再用 adaptiveThreshold C=15（那个对铅笔笔锋/飞白过严）
    def _brush_extract():
        _, _ib = cv2.threshold(flat_gray_full, 80, 255, cv2.THRESH_BINARY_INV)
        _ib = cv2.dilate(_ib, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        _cs, _ = cv2.findContours(_ib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return _cs, _ib

    def _pencil_extract():
        _, _ib = cv2.threshold(flat_gray_full, 220, 255, cv2.THRESH_BINARY_INV)
        # 印刷格线去除：先检测再减，白纸零成本
        _ib = ie.remove_ruled_lines(_ib, paper_mask=paper_mask_global,
                                    h_lines='auto', v_lines='auto')
        _ib = cv2.morphologyEx(_ib, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _ib = cv2.dilate(_ib, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        _cs, _ = cv2.findContours(_ib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return _cs, _ib

    def _brush_success(cs):
        """毛笔成功校验：area>500 且该轮廓在 flat_gray 上的中位数 < 60（真墨色）。"""
        for c in cs:
            if cv2.contourArea(c) <= 500:
                continue
            m = np.zeros(flat_gray_full.shape, dtype=np.uint8)
            cv2.drawContours(m, [c], -1, 255, -1)
            vals = flat_gray_full[m > 0]
            if vals.size and float(np.median(vals)) < 60:
                return True
        return False

    # 介质判定：auto 用直方图预分类（替代旧 attempt-fallback）
    if medium == 'auto':
        cls = ie.classify_medium(flat_gray_full, paper_mask=paper_mask_global)
        if cls == 'empty':
            print("   ⚠️  输入图近乎空白，无法提取笔迹")
            cls = 'pencil'
        detected_medium = cls
        print(f"   ℹ️  封面 auto: 直方图分类 → {cls}")
    else:
        detected_medium = medium

    if detected_medium == 'brush':
        contours, ink_bin = _brush_extract()
        if not _brush_success(contours) and medium != 'brush':
            print("   ℹ️  brush 未过真墨色校验，回退 pencil")
            contours, ink_bin = _pencil_extract()
            detected_medium = 'pencil'
    else:
        contours, ink_bin = _pencil_extract()

    if not contours:
        margin = min(w, h) // 6
        return (img.crop((margin, margin, w - margin, h - margin)).convert('RGBA'),
                detected_medium, None)

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
        return (img.crop((margin, margin, w - margin, h - margin)).convert('RGBA'),
                detected_medium, None)

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

    # 裁出两份灰度：
    #   cropped      — 原始灰度，用作 RGB 通道（保留笔迹自然浓淡/颗粒感）
    #   flat_cropped — 平场校正后，纸面 ≈ 255；用于生成 alpha 消除纸面阴影
    cropped = gray[y1:y2, x1:x2].copy()
    flat_cropped = flat_gray_full[y1:y2, x1:x2].copy()
    ch_h, ch_w = cropped.shape

    # ── Step 4.5: 质量门槛 ──
    # 旧度量 p95-p5 对「稀疏淡笔迹在白纸上」不灵——p5 仍然接近纸白，contrast 看起来低。
    # 改用 p95 - p1：p1 捕捉最暗 1% 像素，只要字真实存在就能反映其最深处。
    # 经验阈值：p95-p1 < 25 视为基本空白；< 55 视为偏弱，切 pencil-bold。
    p95 = float(np.percentile(flat_cropped, 95))
    p1 = float(np.percentile(flat_cropped, 1))
    contrast = p95 - p1
    auto_override_style: "str | None" = None
    print(f"   笔迹对比度: p95={p95:.0f} - p1={p1:.0f} = {contrast:.1f}")
    if contrast < 25:
        raise ValueError(
            f"缩略图字迹对比度过低 (p95-p1={contrast:.1f})，无法生成可识别封面。\n"
            "建议：(1) 重录视频并用深色铅笔/更大压力书写；"
            "(2) 在书写完成后留 1-2 秒无手静帧；"
            "(3) 用 --char-region 手动指定字符区域。")
    if contrast < 55 and detected_medium == 'pencil':
        print("   ⚠️  笔迹偏弱，自动切到 pencil-bold 渲染（target_dark=25）")
        auto_override_style = 'pencil-bold'

    # ── Step 5: 转 RGBA ──
    # 铅笔的 flat_cropped 笔画常在 230-245 附近，alpha=255-flat 只有 10-25（几乎透明）。
    # 用 CLAHE 做局部对比度增强：8x8 小窗口内做直方图均衡，能把 230-245 的弱信号
    # 拉伸到 60-200，alpha 就能进入可见区间。clipLimit=3.0 是经验值——
    # 太高放大噪点，太低对弱信号帮助不够。
    # 毛笔不跑 CLAHE（输入已经有强反差，CLAHE 会放大纸面纤维噪声）。
    flat_for_alpha = flat_cropped.copy()
    if detected_medium == 'pencil':
        flat_u8 = np.clip(flat_for_alpha, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        flat_for_alpha = clahe.apply(flat_u8)
        p1_before = float(np.percentile(flat_cropped, 1))
        p1_after = float(np.percentile(flat_for_alpha, 1))
        print(f"   CLAHE alpha 增强: flat_cropped p1={p1_before:.0f} → "
              f"flat_for_alpha p1={p1_after:.0f}")
    # alpha 基于 flat_gray：纸面阴影已被平场校正抹平，alpha 真实反映「笔迹浓度」
    alpha = np.clip(255.0 - flat_for_alpha.astype(np.float32), 0, 255)
    alpha[alpha < 25] = 0

    # v5 审美精修：去字周「白色光晕」
    # 旧实现 4% 线性 ramp 让 alpha 在 12-20 px 内线性过渡，在 composite 阶段
    # 这条 ramp 被当成"字的一部分"渲染，形成字外 1-2% 的白色光晕（类似 PS 外发光）。
    # 极客禅目标是"字直接写在纸上"的干净感，不要这种光晕。
    # 新实现：feather 1%（原 4%）+ smoothstep（原 linear）——过渡区域更窄、曲线更陡，
    # 消除 ramp 尾部的"字边缘轻 alpha 带"。
    feather = max(2, int(min(ch_w, ch_h) * 0.01))
    fade = np.ones((ch_h, ch_w), dtype=np.float32)
    ramp = np.linspace(0, 1, feather, dtype=np.float32)
    # smoothstep: y = x² (3 - 2x)，两端导数为 0，中段陡——视觉过渡柔但窄
    ramp_ss = ramp * ramp * (3.0 - 2.0 * ramp)
    fade[:feather, :] *= ramp_ss[:, None]
    fade[ch_h - feather:, :] *= ramp_ss[::-1, None]
    fade[:, :feather] *= ramp_ss[None, :]
    fade[:, ch_w - feather:] *= ramp_ss[None, ::-1]
    alpha *= fade
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    # RGB 仍用原始灰度保留笔迹浓淡纹理（flat_gray 会让颗粒感失真）
    ink_gray = np.clip(cropped, 0, 255).astype(np.uint8)
    result = np.zeros((ch_h, ch_w, 4), dtype=np.uint8)
    result[:, :, 0] = ink_gray
    result[:, :, 1] = ink_gray
    result[:, :, 2] = ink_gray
    result[:, :, 3] = alpha

    return Image.fromarray(result, 'RGBA'), detected_medium, auto_override_style


# ============================================================
# 墨迹渲染（简化版：信任输入，只做背景替换）
# ============================================================

def render_ink(gray: 'np.ndarray', cover_style: str = 'brush') -> 'np.ndarray':
    """
    渲染：背景替换为米白底色 + 按风格加深笔迹。

    风格（cover_style）：
      'brush'       — 毛笔：保留原始浓淡，仅 ×0.9 微调。
      'pencil-zen'  — 铅笔禅意版：最暗点映射到 ~55。保留铅笔灰调与石墨颗粒感，
                       高级灰感、符合「极客禅」美学（默认铅笔）。
      'pencil-bold' — 铅笔加浓版：最暗点映射到 ~25。把铅笔字渲染成接近毛笔浓墨的
                       观感，对比度高；A/B 测试备用。

    调用方根据 extract_calligraphy 返回的 medium 传入合适的 style（而非再做推断）。
    """
    import numpy as np
    import cv2

    gray = gray.astype(np.float64)
    bg_val = np.mean(BG_COLOR[:3])  # ~240

    is_pencil = cover_style.startswith('pencil')

    # ── 自适应阈值 ──
    # 旧版写死 _bg_floor=205/_ink_cap=190。she.mp4 的淡铅笔 thumb 实际笔画
    # 在 230-245，全部落在 205 之上 → ink_weight 全为 0 → 任何拉伸都无效。
    # 改为从直方图推：p95 代表纸面实际亮度，dark_pixels 的 p5 代表笔迹最暗段。
    p95 = float(np.percentile(gray, 95))
    dark_pixels = gray[gray < p95 - 5.0]
    if dark_pixels.size > 0:
        p5_ink = float(np.percentile(dark_pixels, 5))
    else:
        p5_ink = p95 - 30.0  # fallback：无暗像素，构造一个合理默认
    _bg_floor = p95 - 3.0                       # 比纸面略暗就开始算笔迹
    _ink_cap = max(p5_ink + 5.0, _bg_floor - 50.0)
    # 保护：确保 _bg_floor > _ink_cap，避免除零/翻转
    if _bg_floor <= _ink_cap:
        _ink_cap = _bg_floor - 10.0
    print(f"   render_ink 自适应: p95={p95:.0f}, p5_ink={p5_ink:.0f}, "
          f"_bg_floor={_bg_floor:.0f}, _ink_cap={_ink_cap:.0f}")

    ink_weight = np.clip((_bg_floor - gray) / max(1.0, _bg_floor - _ink_cap), 0.0, 1.0)
    ink_weight_u8 = (ink_weight * 255).astype(np.uint8)
    ink_weight_smooth = cv2.GaussianBlur(ink_weight_u8, (7, 7), 1.5).astype(np.float64) / 255.0

    # 铅笔动态范围拉伸：把笔迹最暗点映射到 target_dark
    # pencil-zen 自适应：根据输入 darkest 决定拉伸强度，
    # 输入质量参差时输出视觉墨度保持一致。
    if is_pencil and (ink_weight > 0.5).any():
        darkest = float(gray[ink_weight > 0.5].min())
        if cover_style == 'pencil-bold':
            target_dark = 25.0
        else:  # pencil-zen（默认）
            if darkest < 100:
                target_dark = 55.0   # 字够暗，轻拉伸保留禅意灰
            elif darkest < 160:
                target_dark = 40.0   # 中等淡，中等拉伸
            else:
                target_dark = 25.0   # 极淡，强拉伸（等效 pencil-bold）
        source_bright = max(_bg_floor, darkest + 30.0)
        slope = (bg_val - target_dark) / (source_bright - darkest)
        rescaled = np.clip((gray - darkest) * slope + target_dark, 0.0, bg_val)
        gray = rescaled * ink_weight_smooth + gray * (1.0 - ink_weight_smooth)

    # 笔迹像素保持原值（或拉伸后），背景替换为底色
    result = gray * ink_weight_smooth + bg_val * (1.0 - ink_weight_smooth)

    # 毛笔常规微调：整体 ×0.9 微调对比
    if not is_pencil:
        darkest = float(gray[ink_weight > 0.5].min()) if (ink_weight > 0.5).any() else 255
        if darkest > 40:
            enhanced = result * 0.9
            result = enhanced * ink_weight_smooth + result * (1.0 - ink_weight_smooth)

    result = np.clip(result, 0, 255)

    # RGB：暖黑色调 — 越黑的像素暖色偏移越大
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
                   enable_stamp: bool = True,
                   medium: str = 'auto',
                   cover_style: "str | None" = None):
    """生成小红书/Shorts 封面。

    Args:
        medium: 书写介质 'auto'|'brush'|'pencil'，auto 用直方图分类
        cover_style: 渲染风格 'brush'|'pencil-zen'|'pencil-bold'|None
            None（默认）= 根据检测到的 medium 选：brush→'brush'、pencil→'pencil-zen'
    """
    import numpy as np

    # --- 提取书法字（返回检测到的 medium 及质量自动覆盖建议）---
    print(f"   提取书法字 (medium={medium})...")
    calligraphy, detected_medium, auto_override = extract_calligraphy(
        thumb_path, char, medium=medium)

    # 风格决策顺序：
    # 1. 用户显式 cover_style > 2. 质量自动覆盖 > 3. 介质默认
    if cover_style is None:
        if auto_override is not None:
            cover_style = auto_override
        else:
            cover_style = 'brush' if detected_medium == 'brush' else 'pencil-zen'
    print(f"   渲染风格: {cover_style} (detected={detected_medium}, "
          f"auto_override={auto_override})")

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

    # --- 锚点布局（v5：字中心锚 38% 黄金分割点） ---
    char_gap = int(cover_height * CHAR_GAP_RATIO)        # 3%
    top_min = int(cover_height * TOP_MIN_RATIO)          # 10%
    char_center_y = int(cover_height * CHAR_CENTER_Y_RATIO)  # 38%

    # v5 审美精修：极致留白
    # 旧比例 0.48 / 0.70 让字在画面几乎占 50%+，偏"课本插图"。
    # 小红书高赞书法封面的字普遍占画面 15-25%，靠大量留白留出呼吸感。
    # 单字 0.28、多字 0.45 — 单字按宽度定字号，多字保留横向排布空间。
    # max_h 同步下调避免纵向溢出（单字 aspect≈1、多字 aspect>2）。
    if is_multi:
        max_w = int(cover_width * 0.45)
        max_h = int(cover_height * 0.22)
    else:
        max_w = int(cover_width * 0.28)
        max_h = int(cover_height * 0.22)

    scale_w = max_w / cw
    scale_h = max_h / ch
    scale = min(scale_w, scale_h)
    target_w = int(cw * scale)
    target_h = int(ch * scale)

    # v5：字中心定在 38% → paste_y 从 target_h 反推；保留 top_min 边界保护
    paste_y = char_center_y - target_h // 2
    if paste_y < top_min:
        paste_y = top_min

    # 派生 char_bottom / divider_y / title_y（不再是固定常量）
    char_bottom = paste_y + target_h
    divider_y = char_bottom + char_gap   # 圆点在字底下方 3% 处

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

    # 贴入墨迹（v5：改硬覆盖为 2.5% 边缘渐变的 alpha-blend）
    # 旧实现 canvas_gray[bbox] = gray_arr 直接覆盖：bbox 内 noise 纹理被擦掉，
    # 替换成 composite_gray（纸区为无纹理的 bg_gray_val），形成可见的"纸块"方框
    # — 和米白底有明显色差 + 直角边切。
    # 新实现：保留 composite_gray 的中心区，但在 bbox 边缘 2.5% 做 smoothstep
    # 渐变到 canvas_gray —— 纹理 canvas 和 composite 在边缘柔和融合。
    gray_arr = np.array(gray_resized, dtype=np.float64)
    py1, py2 = max(0, paste_y), min(cover_height, paste_y + target_h)
    px1, px2 = max(0, paste_x), min(cover_width, paste_x + target_w)
    sy1, sy2 = py1 - paste_y, py1 - paste_y + (py2 - py1)
    sx1, sx2 = px1 - paste_x, px1 - paste_x + (px2 - px1)

    bh, bw = (py2 - py1), (px2 - px1)
    # 2.5% 的边缘渐变，smoothstep 曲线（两端导数 0 → 边界自然消失感）
    ramp_px = max(2, int(min(bh, bw) * 0.025))
    spatial_blend = np.ones((bh, bw), dtype=np.float64)
    if 0 < ramp_px * 2 < min(bh, bw):
        r = np.linspace(0, 1, ramp_px, dtype=np.float64)
        r_ss = r * r * (3.0 - 2.0 * r)
        spatial_blend[:ramp_px, :] *= r_ss[:, None]
        spatial_blend[-ramp_px:, :] *= r_ss[::-1, None]
        spatial_blend[:, :ramp_px] *= r_ss[None, :]
        spatial_blend[:, -ramp_px:] *= r_ss[None, ::-1]

    composite_in = gray_arr[sy1:sy2, sx1:sx2]
    canvas_gray[py1:py2, px1:px2] = (
        composite_in * spatial_blend
        + canvas_gray[py1:py2, px1:px2] * (1.0 - spatial_blend)
    )

    canvas_rgb = render_ink(canvas_gray, cover_style=cover_style)
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

        # 标题过长时自动缩小字号，确保不超出画面（留 5% 左右边距）
        max_title_w = int(cover_width * 0.90)
        while tw > max_title_w and base_size > 28:
            base_size -= 2
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

    # --- 品牌印章（右下角，竖椭圆白文禅字印）---
    if enable_stamp and os.path.exists(STAMP_IMAGE):
        stamp_src = Image.open(STAMP_IMAGE).convert('RGBA')
        # 宽度约 11%（小红书缩略图下辨识度好）
        stamp_w = int(cover_width * 0.11)
        ratio = stamp_w / stamp_src.width
        stamp_h = int(stamp_src.height * ratio)
        stamp_resized = stamp_src.resize((stamp_w, stamp_h), Image.LANCZOS)

        # 印章图片已预处理（崩边、石纹、旋转2.5°、终值模糊），直接使用
        # 整体透明度 80%
        r, g, b, a = stamp_resized.split()
        a = a.point(lambda x: int(x * 0.80))
        stamp_resized.putalpha(a)
        stamp_rotated = stamp_resized

        # 右下角定位
        margin_x = int(cover_width * 0.05)
        margin_y = int(cover_height * 0.04)
        sx = cover_width - margin_x - stamp_rotated.width
        sy = cover_height - margin_y - stamp_rotated.height

        stamp_layer = Image.new('RGBA', (cover_width, cover_height), (0, 0, 0, 0))
        stamp_layer.paste(stamp_rotated, (sx, sy))
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
    parser.add_argument('--medium', choices=['auto', 'brush', 'pencil'], default='auto',
                        help='书写介质：brush=毛笔, pencil=铅笔, auto=自动 (默认: auto)')
    parser.add_argument('--cover-style',
                        choices=['brush', 'pencil-zen', 'pencil-bold'],
                        default=None,
                        help='封面渲染风格：brush（毛笔默认），pencil-zen（铅笔禅意灰调，默认），'
                             'pencil-bold（铅笔加浓，像毛笔）。不指定时按介质自动选。')
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
            medium=args.medium,
            cover_style=args.cover_style,
        )
    else:
        parser.print_help()
        print("\n❌ 请提供 --thumb（单张模式）或 --batch（批量模式）")
        sys.exit(1)


if __name__ == '__main__':
    main()
