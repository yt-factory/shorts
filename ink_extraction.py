"""
极客禅·墨 — 共享图像处理工具

被 webcam_ink_processor.py 和 xhs_cover.py 共用的笔迹提取基础操作。
不得反向依赖那两个模块——它是被依赖方。

主要函数：
  flat_field_correct      — 平场校正（去光照渐变，纸面归一化到 ~255）
  background_subtract_mask — 背景差分 + Otsu 得笔迹二值 mask
  classify_medium          — 直方图预分类 brush/pencil/empty
  remove_ruled_lines       — 两层去线：先检测、无线则零成本跳过
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import cv2
import numpy as np


Medium = Literal["brush", "pencil", "empty"]


# ============================================================
# 平场校正
# ============================================================

def flat_field_correct(gray: np.ndarray) -> np.ndarray:
    """平场校正：去除光照渐变与纸面阴影，使纸张背景接近纯白 255。

    原理：纸面本身应是接近均匀的高反射率，光照场变化才是低频暗斑。
    用大核灰度膨胀估算这张图「应该有多亮」（每点邻域内最亮值），
    然后原图 / 膨胀得到「相对反射率」——纸面 → 1.0（近 255），
    笔迹 → <1（保持相对暗度）。核尺寸要远大于笔画宽度，但小于光照场
    变化尺度，这里取 ~画面 1/15。

    Args:
        gray: uint8 灰度图

    Returns:
        uint8 平场校正后的灰度图，纸面归一化接近 255
    """
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    h, w = gray.shape[:2]
    # kernel 必须奇数；至少 31 以平滑纸面微纹理
    k = max(31, (min(h, w) // 15) | 1)
    illumination = cv2.dilate(
        gray,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
        iterations=1,
    ).astype(np.float32)
    illumination[illumination == 0] = 1.0  # 防除零
    reflectance = gray.astype(np.float32) / illumination
    flat = np.clip(reflectance * 255.0, 0, 255).astype(np.uint8)
    return flat


# ============================================================
# 背景差分 → 二值 mask
# ============================================================

def background_subtract_mask(
    gray: np.ndarray,
    blur_ksize: int = 5,
    min_threshold: int = 8,
) -> np.ndarray:
    """背景差分得到笔迹二值 mask。

    步骤：
      1. 高斯模糊降底噪（CMOS sensor noise）
      2. 大核灰度膨胀估计「干净纸面」
      3. absdiff 得 diff 图
      4. Otsu 自适应阈值（在 diff 图上通常双峰：噪声近 0、笔迹高值）

    关键：空白帧上 diff 全是近零噪声，Otsu 会在噪声带内瞎选。必须给
    最小阈值（min_threshold=8）作地板，防止空白帧产生满屏假笔迹。

    Args:
        gray: uint8 灰度图（推荐先过 flat_field_correct）
        blur_ksize: 预降噪高斯核大小（0/None 关闭）
        min_threshold: Otsu 结果下限；低于此值用此值兜底

    Returns:
        uint8 二值 mask，笔迹=255，背景=0
    """
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    h, w = gray.shape[:2]

    if blur_ksize and blur_ksize >= 3:
        # 必须奇数
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        blurred = gray

    # 干净纸面估计：核大小略小于 flat_field_correct，抓住字的局部邻域
    bg_k = max(21, (min(h, w) // 30) | 1)
    clean_paper = cv2.dilate(
        blurred,
        cv2.getStructuringElement(cv2.MORPH_RECT, (bg_k, bg_k)),
        iterations=1,
    )
    diff = cv2.absdiff(clean_paper, blurred)

    # Otsu on diff，但要防空白帧
    otsu_t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(min_threshold), float(otsu_t))
    _, mask = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)
    return mask


# ============================================================
# 介质分类（替代 attempt-fallback）
# ============================================================

def classify_medium(
    gray: np.ndarray,
    paper_mask: Optional[np.ndarray] = None,
    brush_max_dark: int = 55,
    empty_min_dark: int = 200,
) -> Medium:
    """直方图预分类：看"纸面内最暗 1% 像素的中位数"。

    - 中位数 < brush_max_dark (55) → 'brush'：真墨色
    - 中位数 >= empty_min_dark (200) → 'empty'：近乎空白帧
    - 其他 → 'pencil'

    用中位数（不是 min）对孤立噪点鲁棒。取最暗 1% 而不是整体均值，
    是因为我们关心「存不存在深色笔迹」，不关心纸面整体亮度。

    paper_mask 会被 erode 几像素，避开纸边阴影被当成"最暗像素"。

    Args:
        gray: uint8 灰度图（最好已 flat_field_correct 过）
        paper_mask: uint8 纸面区域 mask（255=纸面）；None 用中心 60% ROI
        brush_max_dark, empty_min_dark: 分类阈值

    Returns:
        'brush' | 'pencil' | 'empty'
    """
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    h, w = gray.shape[:2]

    if paper_mask is None:
        # 中心 60% ROI 作默认纸面
        paper_mask = np.zeros((h, w), dtype=np.uint8)
        x0 = int(w * 0.2); y0 = int(h * 0.2)
        x1 = int(w * 0.8); y1 = int(h * 0.8)
        paper_mask[y0:y1, x0:x1] = 255
    else:
        # 向内收缩避开纸边阴影污染「最暗」统计
        paper_mask = cv2.erode(
            paper_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
            iterations=1,
        )

    vals = gray[paper_mask > 0]
    if vals.size < 100:
        return "pencil"  # 无足够样本，保守走 pencil（更 robust 的分支）

    k = max(1, vals.size // 100)  # 最暗 1%
    darkest = np.partition(vals, k)[:k]
    dark_med = float(np.median(darkest))

    if dark_med < brush_max_dark:
        return "brush"
    if dark_med >= empty_min_dark:
        return "empty"
    return "pencil"


# ============================================================
# 去线（两层：廉价检测 + 精确减除）
# ============================================================

def _detect_ruled_axis(proj: np.ndarray, img_extent: int) -> list[int]:
    """在一维投影上找「窄 / 高 / 相对规律间距」的尖峰（印刷格线特征）。

    判据：
      - 峰高 > img_extent * 255 * 0.4（线至少占该方向 40% 的长度）
      - 峰宽 <= 6 像素（印刷线通常 1-5px 带反走样）
      - 至少 2 条，且间距波动 < 50%（规律性）

    Returns:
        精确坐标列表（每根线的中心）。空列表 = 无线。
    """
    if proj.size == 0:
        return []

    # 峰高门槛：原本 0.4 要求线覆盖 40%+ 高度——但手部遮挡、纸面不完整、
    # bg_subtract 只在线颜色比邻域明显暗的像素生效，一条线最终在 mask 上
    # 往往只剩 10-20% 的长度。门槛降到 0.10 以抓住部分可见的印刷线。
    peak_min = img_extent * 255 * 0.10
    # 找所有 > peak_min 的连续区段
    above = proj > peak_min
    peaks: list[tuple[int, int]] = []  # (center, width)
    i = 0
    n = above.size
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            width = j - i
            if width <= 6:
                peaks.append(((i + j - 1) // 2, width))
            i = j
        else:
            i += 1

    if len(peaks) < 2:
        return []

    # 规律性校验：相邻间距应大致相等
    centers = [c for c, _ in peaks]
    gaps = np.diff(centers)
    if gaps.size == 0:
        return []
    gap_med = float(np.median(gaps))
    if gap_med < 20:  # 间距太近不太像印刷格
        return []
    rel_spread = float(np.std(gaps)) / max(1.0, gap_med)
    if rel_spread > 0.5:
        return []

    return centers


def remove_ruled_lines(
    mask: np.ndarray,
    paper_mask: Optional[np.ndarray] = None,
    h_lines: bool = True,
    v_lines: str | bool = "auto",
    shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """两层去线：
      第一层 — 投影扫描，廉价检测有没有规律格线。
      第二层 — 有线：只在那几条 ±3px 窄带上减；无线：原样返回。

    关键洞察：白纸是常态。对白纸图无脑跑 morphological opening 会伤汉字
    长竖/长横（"申"、"中"的竖画往往 > 36px）。先检测再动刀，零误伤。

    Args:
        mask: uint8 笔迹二值 mask
        paper_mask: 只在纸面区域内找线（选传；None = 全图）
        h_lines: 是否处理横线（True/False/'auto'）
        v_lines: 是否处理竖线（True/False/'auto'）
        shape: (H, W)，仅当 mask 可能是 None 时用

    Returns:
        uint8 减线后的 mask
    """
    if mask is None:
        return mask
    h, w = mask.shape[:2] if shape is None else shape
    result = mask.copy()

    work_mask = mask
    if paper_mask is not None:
        work_mask = cv2.bitwise_and(mask, paper_mask)

    # 用 paper_mask 的实际尺寸估算可能的线长度——用全图尺寸会
    # 因为 paper 只占部分画面而过严（如 4K 视频纸面只占 74%）。
    if paper_mask is not None:
        rows_any = np.any(paper_mask > 0, axis=1)
        cols_any = np.any(paper_mask > 0, axis=0)
        paper_h = int(rows_any.sum()) if rows_any.any() else h
        paper_w = int(cols_any.sum()) if cols_any.any() else w
    else:
        paper_h, paper_w = h, w

    # ── 竖线 ──
    do_v = v_lines is True or v_lines == "auto"
    if do_v:
        col_proj = np.sum(work_mask > 0, axis=0) * 255  # 标量化到 255 量级
        col_peaks = _detect_ruled_axis(col_proj, img_extent=paper_h)
        if v_lines is True and not col_peaks:
            col_peaks = _fallback_ruled_v(work_mask, h, w)
        for cx in col_peaks:
            result[:, max(0, cx - 3): min(w, cx + 4)] = 0

    # ── 横线 ──
    do_h = h_lines is True or h_lines == "auto"
    if do_h:
        row_proj = np.sum(work_mask > 0, axis=1) * 255
        row_peaks = _detect_ruled_axis(row_proj, img_extent=paper_w)
        if h_lines is True and not row_peaks:
            row_peaks = _fallback_ruled_h(work_mask, h, w)
        for cy in row_peaks:
            result[max(0, cy - 3): min(h, cy + 4), :] = 0

    return result


def _fallback_ruled_v(mask: np.ndarray, h: int, w: int) -> list[int]:
    """投影检测不确定时的 fallback：用大核 morphological opening。
    line_len 取 min(h,w)/4，比任何汉字最长笔画都长，比印刷线短（1–3px 但全幅）。
    实际上印刷线的绝对长度~h，汉字最长笔画 < h/4，所以这个阈值安全。"""
    line_len = max(50, min(h, w) // 4)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
    # 取每条检测出来的线的 x 中心
    cs, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [int(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2) for c in cs]


def _fallback_ruled_h(mask: np.ndarray, h: int, w: int) -> list[int]:
    """水平方向 fallback：同上，但用横核。"""
    line_len = max(50, min(h, w) // 4)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk)
    cs, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [int(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2) for c in cs]
