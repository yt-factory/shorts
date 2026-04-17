#!/usr/bin/env python3
"""
极客禅·墨 — 视频后处理脚本 v3.0

完整 14 步处理流水线：
  ① 翻转（自动检测或手动指定）
  ② 防抖（两遍式视频稳定）
  ③ 墨迹检测（定位字的位置和大小）
  ④ 裁剪居中（以字为中心）
  ⑤ 放大（lanczos 高质量缩放）
  ⑥ 白平衡/亮度校正（白纸变白、墨迹变黑）
  ⑦ 锐化（根据放大倍数动态调整）
  ⑧ 淡入（开头 0.5s 渐显）
  ⑨ 去除背景音
  ⑩ 拼接静止画面（自动检测无手帧）
  ⑪ 淡出（结尾 1s 渐隐）
  ⑫ 合并旁白（标准化到 -14 LUFS）
  ⑬ 叠加金句字幕（可选）
  ⑭ 生成缩略图（可选）

依赖：
  pip install opencv-python-headless numpy
  brew install ffmpeg  (或 sudo apt install ffmpeg)
  # 防抖需要 ffmpeg 编译时包含 vidstab：大多数发行版预编译版已包含

用法：
  # 基本用法（自动检测方向）
  python3 ink_video_processor.py input.mp4 -o output.mp4

  # 完整用法（含旁白 + 字幕）
  python3 ink_video_processor.py input.mp4 -v voice.m4a -o final.mp4 \\
      --flip --subtitle "吃茶去。——赵州禅师"

  # 查看所有选项
  python3 ink_video_processor.py --help
"""

import argparse
import asyncio
import subprocess
import sys
import os
import json
import tempfile
import shutil


# ============================================================
# 常量
# ============================================================

VERSION = "3.0"
DEFAULT_OUTPUT_WIDTH = 1080
DEFAULT_OUTPUT_HEIGHT = 1920
DEFAULT_FILL_RATIO = 0.6
DEFAULT_HOLD_SECONDS = 4
DEFAULT_FADE_IN = 0.5
DEFAULT_FADE_OUT = 1.0
DEFAULT_MIN_SCALE = 1.5
TARGET_LUFS = -14
SHORTS_MAX_DURATION = 60
# 楷体最搭「极客禅·墨」书法主题，WSL 下直接读 Windows 字体
DEFAULT_FONT = '/mnt/c/Windows/Fonts/simkai.ttf'


# ============================================================
# 依赖检查
# ============================================================

def check_dependencies():
    """检查必要的外部依赖，给出清晰的安装提示"""
    errors = []

    # ffmpeg / ffprobe
    for tool in ['ffmpeg', 'ffprobe']:
        try:
            subprocess.run([tool, '-version'], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            errors.append(f"  - {tool}: brew install ffmpeg (macOS) / sudo apt install ffmpeg (Ubuntu)")

    # Python 库
    try:
        import cv2
    except ImportError:
        errors.append("  - opencv: pip install opencv-python-headless")

    try:
        import numpy
    except ImportError:
        errors.append("  - numpy: pip install numpy")

    if errors:
        print("❌ 缺少依赖，请安装：")
        for e in errors:
            print(e)
        sys.exit(1)


def check_vidstab_support():
    """检查 ffmpeg 是否支持 vidstab（视频防抖）"""
    result = subprocess.run(
        ['ffmpeg', '-filters'], capture_output=True, text=True
    )
    has_vidstab = 'vidstabdetect' in (result.stdout + result.stderr)
    if not has_vidstab:
        print("   ⚠️  ffmpeg 未包含 vidstab 支持，跳过防抖步骤")
        print("   （如需防抖：brew install ffmpeg --with-libvidstab）")
    return has_vidstab


# ============================================================
# 视频/音频信息工具
# ============================================================

def get_video_info(video_path):
    """
    获取视频的基本信息。

    Returns:
        dict: {'width', 'height', 'duration', 'fps'}
    """
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe 无法读取: {video_path}")

    info = json.loads(result.stdout)
    vs = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    if not vs:
        raise ValueError(f"没有视频流: {video_path}")

    fps_str = vs.get('r_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = int(num) / int(den) if int(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    return {
        'width': int(vs['width']),
        'height': int(vs['height']),
        'duration': float(info['format']['duration']),
        'fps': fps,
    }


def get_audio_duration(audio_path):
    """获取音频文件的时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe 无法读取音频: {audio_path}")
    return float(json.loads(result.stdout)['format']['duration'])


# ============================================================
# ffmpeg 辅助函数
# ============================================================

def run_ffmpeg(cmd, step_name="ffmpeg"):
    """运行 ffmpeg 命令，统一错误处理"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = result.stderr[-1000:] if result.stderr else "未知错误"
        print(f"   ❌ {step_name} 失败:")
        # 只显示最后几行有意义的错误信息
        for line in error_msg.strip().split('\n')[-5:]:
            print(f"      {line}")
        sys.exit(1)
    return result


def extract_last_frame(video_path, output_path):
    """从视频中提取最后一帧为 PNG"""
    run_ffmpeg([
        'ffmpeg', '-y', '-sseof', '-0.1', '-i', video_path,
        '-vframes', '1', '-update', '1', output_path
    ], "提取最后一帧")


def create_still_video(image_path, output_path, duration, fps):
    """用一张图片生成指定时长的静止视频"""
    run_ffmpeg([
        'ffmpeg', '-y',
        '-loop', '1', '-i', image_path,
        '-t', str(duration),
        '-vf', f'fps={int(fps)}',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_path
    ], "生成静止视频")


def concat_videos(video_paths, output_path):
    """拼接多个视频文件（使用 concat demuxer）"""
    tmpdir = os.path.dirname(os.path.abspath(video_paths[0]))
    concat_file = os.path.join(tmpdir, f'concat_{os.getpid()}.txt')
    with open(concat_file, 'w') as f:
        for v in video_paths:
            escaped_v = v.replace("'", "'\\''")
            f.write(f"file '{escaped_v}'\n")

    run_ffmpeg([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        output_path
    ], "视频拼接")


# ============================================================
# 帧提取工具
# ============================================================

def extract_frame(video_path, position='last'):
    """
    从视频中提取一帧用于分析。

    Args:
        position: 'last' 取最后帧，'middle' 取中间帧
    Returns:
        numpy.ndarray: BGR 格式的帧图像
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError("视频帧数为 0")

    offsets = [2, 5, 10, 15, 1] if position == 'last' else [total // 2]
    for off in offsets:
        idx = max(0, total - off) if position == 'last' else off
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            cap.release()
            return frame

    cap.release()
    raise ValueError("无法读取视频帧")


# ============================================================
# Step ①: 方向检测（Feature 6）
# ============================================================

def _edge_book_score(gray_region):
    """计算一个区域的书脊得分。
    深色像素权重 ×10：书脊有大面积暗色文字，纸边/阴影只有薄线条。
    """
    import cv2
    import numpy as np
    dark = float(np.mean(gray_region < 100))
    edges = float(np.mean(cv2.Canny(gray_region, 50, 150)))
    return dark * 10 + edges


def detect_orientation(video_path):
    """
    检测视频方向，确定需要的旋转角度。

    目标：旋转后书脊永远在画面正上方，字在书脊下方。
    - 书脊在顶部 → 不旋转
    - 书脊在底部 → 旋转 180°
    - 书脊在右侧 → 顺时针旋转 90°（右→上）
    - 书脊在左侧 → 逆时针旋转 90°（左→上）

    Returns:
        dict: {
            'rotation': str,        # 'none'|'180'|'cw90'|'ccw90'
            'raw_book_edge': str,   # 原始画面书脊位置
            'confidence': str,      # 'high'|'medium'|'low'
            'reason': str,
        }
    """
    import cv2
    import numpy as np

    frame = extract_frame(video_path, 'last')
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 扫描四个边缘区域
    edge_scores = {
        'top': _edge_book_score(gray[:int(h * 0.20), :]),
        'bottom': _edge_book_score(gray[int(h * 0.80):, :]),
        'left': _edge_book_score(gray[:, :int(w * 0.20)]),
        'right': _edge_book_score(gray[:, int(w * 0.80):]),
    }

    # 书脊所在边 = 得分最高的边
    raw_book_edge = max(edge_scores, key=edge_scores.get)
    best_score = edge_scores[raw_book_edge]

    # 与次高分比较，判断置信度
    sorted_scores = sorted(edge_scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
    score_ratio = best_score / max(second_score, 0.001)

    if score_ratio > 2.0 and best_score > 0.05:
        confidence = 'high'
    elif score_ratio > 1.3 and best_score > 0.03:
        confidence = 'medium'
    else:
        confidence = 'low'

    # 旋转方向：将书脊所在边旋转到顶部
    # CW90 (transpose=1): left→top, right→bottom
    # CCW90 (transpose=2): right→top, left→bottom
    rotation_map = {
        'top': 'none',      # 已在顶部
        'bottom': '180',    # 底部→顶部 = 180°
        'right': 'ccw90',   # 右侧→顶部 = 逆时针 90°
        'left': 'cw90',     # 左侧→顶部 = 顺时针 90°
    }

    # 始终使用检测到的最高分边作为旋转依据（即使置信度低）
    rotation = rotation_map[raw_book_edge]

    edge_names = {'top': '顶部', 'bottom': '底部', 'left': '左侧', 'right': '右侧'}
    rotation_names = {'none': '不旋转', '180': '旋转180°', 'cw90': '顺时针90°', 'ccw90': '逆时针90°'}
    reason = f"书脊在{edge_names[raw_book_edge]} → {rotation_names[rotation]}"

    return {
        'rotation': rotation,
        'raw_book_edge': raw_book_edge,
        'confidence': confidence,
        'reason': reason,
    }


# ============================================================
# Step ③: 墨迹检测
# ============================================================

def _apply_rotation(frame, rotation):
    """对 cv2 帧应用旋转"""
    import cv2
    if rotation == '180':
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 'cw90':
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 'ccw90':
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def detect_ink_region(video_path, rotation='none'):
    """
    检测墨迹（字）的位置和大小。
    先应用旋转使书脊在顶部，再排除顶部 30% 书脊区域。

    Args:
        video_path: 视频文件路径
        rotation: 旋转方式 ('none'|'180'|'cw90'|'ccw90')

    Returns:
        dict: {'x', 'y', 'w', 'h', 'cx', 'cy'}
    """
    import cv2
    import numpy as np

    frame = extract_frame(video_path, 'last')
    frame = _apply_rotation(frame, rotation)

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 旋转后书脊永远在顶部 → 排除顶部 30%
    a_top, a_bot = int(h * 0.38), int(h * 0.95)
    a_left, a_right = 0, w

    gc = gray[a_top:a_bot, a_left:a_right]
    # 全局阈值检测深色像素
    _, mask = cv2.threshold(gc, 55, 255, cv2.THRESH_BINARY_INV)
    # 白纸遮罩：只保留白纸区域上的墨迹（排除书脊上的印刷字）
    # 白纸 = RGB 三通道均 > 180；彩色书脊至少有一个通道偏低
    color_region = frame[a_top:a_bot, a_left:a_right]
    paper_mask = np.all(color_region > 150, axis=2).astype(np.uint8) * 255
    # 膨胀白纸遮罩，覆盖墨迹笔画本身（笔画不是白色但在白纸上）
    paper_mask = cv2.dilate(paper_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)), iterations=2)
    mask = cv2.bitwise_and(mask, paper_mask)
    # 先腐蚀去噪点，再适度膨胀连接笔画（不过度扩张）
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("   ⚠️  未检测到墨迹，使用画面中心")
        cy = (a_top + a_bot) // 2
        cx = (a_left + a_right) // 2
        return {'x': cx - w//4, 'y': cy - h//8, 'w': w//2, 'h': h//4, 'cx': cx, 'cy': cy}

    # 面积过滤：最小 0.3%，最大 10%（单个汉字不会超过画面 10%）
    # 形状过滤：排除长条形轮廓（宽高比 > 4:1），这通常是书脊/直线而非汉字
    mn, mx = (w * h) * 0.003, (w * h) * 0.10
    def is_ink_on_paper(c):
        """判断轮廓是否是白纸上的墨迹（排除书脊上的印刷文字）"""
        cx, cy, cw, ch = cv2.boundingRect(c)
        # 形状过滤：排除极端长条（宽高比>4，通常是书脊/直线）
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if not (mn < cv2.contourArea(c) < mx and aspect < 4.0):
            return False
        # 背景亮度检测：检查轮廓周围 10px 的背景是否为白纸
        pad = 10
        ry1 = max(0, cy - pad)
        ry2 = min(gc.shape[0], cy + ch + pad)
        rx1 = max(0, cx - pad)
        rx2 = min(gc.shape[1], cx + cw + pad)
        bg = gc[ry1:ry2, rx1:rx2]
        # 白纸背景：亮度 > 180 的像素占 40%以上
        if bg.size > 0 and np.mean(bg > 180) < 0.4:
            return False  # 背景不是白纸，是书脊
        return True

    valid = [c for c in contours if is_ink_on_paper(c)]
    if not valid:
        # 放宽条件：只过滤面积，不过滤形状
        valid = [c for c in contours if cv2.contourArea(c) > mn] or [max(contours, key=cv2.contourArea)]

    pts = np.vstack(valid)
    x, yr, bw, bh = cv2.boundingRect(pts)

    # 如果合并后的框太扁（宽高比>3），说明散落的噪点拉宽了框
    # 用空间聚类：以最大轮廓为锚点，只保留中心距离在其 2 倍半径内的轮廓
    merged_aspect = max(bw, bh) / max(min(bw, bh), 1)
    if merged_aspect > 3.0 and len(valid) > 1:
        anchor = max(valid, key=cv2.contourArea)
        ax, ay, aw, ah = cv2.boundingRect(anchor)
        acx, acy = ax + aw // 2, ay + ah // 2
        radius = max(aw, ah) * 2
        clustered = []
        for c in valid:
            cx_c, cy_c, _, _ = cv2.boundingRect(c)
            mx_c = cx_c + cv2.boundingRect(c)[2] // 2
            my_c = cy_c + cv2.boundingRect(c)[3] // 2
            if abs(mx_c - acx) < radius and abs(my_c - acy) < radius:
                clustered.append(c)
        if clustered:
            valid = clustered
            pts = np.vstack(valid)
            x, yr, bw, bh = cv2.boundingRect(pts)
            print(f"      聚类过滤: {len(clustered)} 个轮廓（去除散落噪点）")

    # 将分析区域坐标映射回全帧坐标
    x = x + a_left
    y = yr + a_top

    # 呼吸空间 20%
    px, py = int(bw * 0.20), int(bh * 0.20)
    x, y = max(0, x - px), max(0, y - py)
    bw, bh = min(w - x, bw + 2*px), min(h - y, bh + 2*py)

    cx, cy = x + bw//2, y + bh//2

    # 调试图
    dbg = frame.copy()
    cv2.rectangle(dbg, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
    cv2.circle(dbg, (cx, cy), 10, (0, 0, 255), -1)
    # 画分析区域边界线（水平 + 垂直）
    cv2.line(dbg, (0, a_top), (w, a_top), (255, 0, 0), 2)
    cv2.line(dbg, (0, a_bot), (w, a_bot), (255, 0, 0), 2)
    cv2.line(dbg, (a_left, 0), (a_left, h), (255, 0, 0), 2)
    cv2.line(dbg, (a_right, 0), (a_right, h), (255, 0, 0), 2)
    dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ink_debug.png')
    cv2.imwrite(dp, dbg)

    pct = (bw * bh) / (w * h) * 100
    print(f"   ✅ 墨迹: ({x},{y}) {bw}x{bh}, 占画面 {pct:.1f}%")
    print(f"      中心: ({cx},{cy}), 分析区域: x={a_left}~{a_right}, y={a_top}~{a_bot}")
    print(f"      调试图: {dp}")

    return {'x': x, 'y': y, 'w': bw, 'h': bh, 'cx': cx, 'cy': cy}


# ============================================================
# Step ⑤: 手部检测（找到干净的最后一帧）
# ============================================================

def find_clean_last_frame(video_path, rotation='none'):
    """
    从视频末尾向前搜索，找到第一帧没有手的画面。

    原理：手部有大面积肤色。在 YCrCb 色彩空间中，
    肤色的 Cr 通道值在 133-173 之间，Cb 在 77-127 之间。

    Returns:
        numpy.ndarray: 干净的帧（无手），如果找不到则返回最后一帧
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    best_frame = None
    min_skin_ratio = 1.0

    # 从最后往前扫描 30 帧
    for offset in range(1, min(31, total)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - offset)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = _apply_rotation(frame, rotation)

        # 检测肤色区域
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # 肤色范围
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower, upper)

        skin_ratio = np.mean(skin_mask > 0)

        if skin_ratio < min_skin_ratio:
            min_skin_ratio = skin_ratio
            best_frame = frame.copy()

        # 如果肤色占比低于 1%，认为没有手
        if skin_ratio < 0.01:
            cap.release()
            print(f"      找到干净帧: 倒数第 {offset} 帧, 肤色占比 {skin_ratio*100:.2f}%")
            return frame

    cap.release()
    if best_frame is not None:
        print(f"      最佳帧: 肤色占比 {min_skin_ratio*100:.2f}%")
        return best_frame

    # 兜底：直接返回最后一帧
    frame = extract_frame(video_path, 'last')
    return _apply_rotation(frame, rotation)


# ============================================================
# Step ⑥: 白平衡/亮度校正参数
# ============================================================

def build_color_correction_filter(paper_p95: float = None,
                                  target_white: float = 240.0) -> str:
    """
    生成白平衡/亮度校正的 ffmpeg 滤镜。

    两种模式：

    **自适应模式** (`paper_p95` 传入实测值)：
      用 lutyuv 对 Y 通道做线性 gain 归一化，把纸面 p95 映射到 target_white。
      下游假设「纸面近白」的代码（generate_calligraphy_thumbnail 的 pencil
      curves、xhs_cover 的 flat_field）天然满足假设，不再级联失败。
      不动 chroma → 颜色保真。

    **Fallback 模式** (`paper_p95=None`)：
      保留老的固定 eq，兼容未传参的调用路径（如 ink_video_processor 自己
      的 process_video 仍走老路径）。

    Args:
        paper_p95: 实测纸面 p95 亮度（0-255）。由 sample_paper_brightness 算出。
        target_white: 目标纸面亮度，默认 240。设 240 而非 255 是给笔锋/飞白
                      留动态范围（避免 clip 掉近白像素）。
    """
    if paper_p95 is None or paper_p95 <= 0:
        # 老路径 — 为 webcam 原始录制条件（mean≈195）手调的固定 eq
        return "eq=brightness=0.06:contrast=1.25:saturation=1.0"

    gain = target_white / max(float(paper_p95), 1.0)
    # 安全边界：gain 极端时会放大噪点或 clip 掉字
    gain = min(max(gain, 0.8), 2.5)

    # v5 审美精修：归一化 + 去噪两步组合。
    # gain=1.4+ 会把纸面真实的纤维纹理和微小光照不均一起放大，输出像"水泥墙"。
    # 在 lutyuv 之后、下游锐化之前插 hqdn3d，平掉低频纸噪但保留笔画边缘。
    # 参数 4:3:6:4.5 = luma_spatial:chroma_spatial:luma_tmp:chroma_tmp，
    # 经典"去纸面/保边"组合；temporal 分量利用多帧相关性，静态纸面受益最大。
    # `\,` 转义见 lutyuv 说明；hqdn3d 自身无内部逗号，不需要额外转义。
    return (
        f"lutyuv=y='clip(val*{gain:.4f}\\,0\\,255)',"
        f"hqdn3d=4:3:6:4.5"
    )


def sample_paper_brightness(video_path: str,
                            paper_mask=None,
                            frame_position: str = 'last') -> float:
    """
    采样纸面区域的 p95 亮度，用作色彩校正的归一化锚点。

    **淡出鲁棒性**：若输入视频自带 fade-out（比如用户把上一轮输出当输入重跑），
    最后一帧会是近黑，单帧采样会得出 p95≈16，触发最大 gain (2.5)，把本来
    已经白的纸拉到过曝。改为扫描多个候选帧（offsets 30,60,120,240 以及中间帧），
    取第一个 mean > 100（未淡出）的帧测 p95。若全部候选都淡出，回退到默认 240。

    Args:
        video_path: 视频路径
        paper_mask: 二值 mask (H, W) uint8，255 标记纸面。None 则用整帧。
        frame_position: 兼容性保留；函数内部会尝试多个位置。

    Returns:
        纸面 p95 亮度（0-255 float）。失败时返回 240.0（gain≈1.0 → 画面不变）。
    """
    import cv2
    import numpy as np

    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return 240.0

        # 候选帧：从尾向前跳过 fade-out 区（典型 fade=1s=30 帧 @ 30fps），
        # 再加中间帧作二次兜底。顺序按「越接近视频结尾（书写已完成）越靠前」排序。
        candidates = [30, 60, 120, 240, total // 2, total // 4]
        # 去重 + 范围夹取
        seen = set()
        ordered = []
        for off in candidates:
            idx = max(0, min(total - 1, total - off if off <= total else total // 2))
            if idx in seen:
                continue
            seen.add(idx)
            ordered.append(idx)

        for idx in ordered:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(np.mean(gray)) < 100:
                # 整帧均值太低 → 很可能还在淡出/淡入区
                continue

            if paper_mask is not None and paper_mask.shape == gray.shape:
                pixels = gray[paper_mask > 0]
                if pixels.size < 100:
                    pixels = gray.ravel()
            else:
                pixels = gray.ravel()

            cap.release()
            return float(np.percentile(pixels, 95))

        cap.release()
        print("   ⚠️  所有候选帧均过暗（可能全视频淡出），使用默认值 240")
        return 240.0
    except Exception as e:
        print(f"   ⚠️  纸面亮度采样失败（{e}），使用默认值 240")
        return 240.0


# ============================================================
# Step ⑦: 动态锐化（Feature 7）
# ============================================================

def build_sharpen_filter(scale_factor):
    """
    根据放大倍数动态生成锐化滤镜。
    放大越多，锐化越强。

    双通道锐化策略（适合墨迹/书法）：
    1. 大半径 unsharp：恢复放大丢失的整体笔画轮廓
    2. 小半径 unsharp：补回笔锋等细节

    unsharp=luma_x:luma_y:luma_amount:chroma_x:chroma_y:chroma_amount
    """
    if scale_factor <= 1.2:
        # 无放大或极小放大：轻锐化
        return "unsharp=5:5:0.6:3:3:0.0"
    elif scale_factor <= 1.5:
        # ~1.5x：轮廓恢复 + 细节补偿
        return "unsharp=5:5:1.2:3:3:0.0,unsharp=3:3:0.6:3:3:0.0"
    elif scale_factor <= 2.5:
        # 1.5-2.5x：强锐化 + 细节补偿
        return "unsharp=5:5:1.5:3:3:0.0,unsharp=3:3:0.7:3:3:0.0"
    elif scale_factor <= 4.0:
        # 2.5-4x：强锐化
        return "unsharp=7:7:1.8:3:3:0.0,unsharp=3:3:0.8:3:3:0.0"
    else:
        # >4x：最强锐化
        return "unsharp=7:7:2.0:3:3:0.0,unsharp=3:3:1.0:3:3:0.0"


# ============================================================
# Step ⑪.5: TTS 语音合成 + 同步字幕
# ============================================================

DEFAULT_TTS_VOICE = 'zh-CN-YunxiNeural'
DEFAULT_TTS_BUFFER = 3  # 留白秒数
CHARS_PER_SECOND = 3.5  # 中文平均语速（字/秒）


def split_sentences(text: str) -> list[str]:
    """将文本按句子切分（中英文标点均支持）"""
    import re
    # 按中文/英文句末标点切分，保留标点
    parts = re.split(r'(?<=[。！？；\.\!\?\;])', text.strip())
    return [s.strip() for s in parts if s.strip()]


async def _tts_generate(text: str, voice: str, rate: str,
                        audio_path: str) -> list[dict]:
    """
    调用 Edge TTS 生成音频并收集句子时间戳。

    Returns:
        list[dict]: [{'offset_ms': int, 'duration_ms': int, 'text': str}, ...]
    """
    import edge_tts

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    boundaries = []

    with open(audio_path, 'wb') as f:
        async for chunk in communicate.stream():
            if chunk['type'] == 'audio':
                f.write(chunk['data'])
            elif chunk['type'] == 'SentenceBoundary':
                boundaries.append({
                    'offset_ms': chunk['offset'] // 10000,  # 100ns → ms
                    'duration_ms': chunk['duration'] // 10000,
                    'text': chunk['text'],
                })
    return boundaries


def generate_tts(text: str, target_duration: float, output_dir: str,
                 voice: str = DEFAULT_TTS_VOICE,
                 buffer: float = DEFAULT_TTS_BUFFER,
                 max_retries: int = 3) -> tuple[str, str]:
    """
    生成 TTS 音频和同步 SRT 字幕。

    自动调节语速使旁白 fit 进 target_duration（秒）。

    Args:
        text: 旁白文本
        target_duration: 可用旁白时长（秒）
        output_dir: 输出目录
        voice: Edge TTS 语音 ID
        buffer: 额外留白时间（秒）
        max_retries: 最大语速调整重试次数

    Returns:
        (audio_path, srt_path)
    """
    available = target_duration - buffer
    if available < 3:
        available = target_duration  # 时间太短就不留白了

    audio_path = os.path.join(output_dir, 'tts_voice.mp3')
    srt_path = os.path.join(output_dir, 'tts_subtitles.srt')

    # 估算初始语速
    char_count = len(text.replace(' ', '').replace('\n', ''))
    estimated_secs = char_count / CHARS_PER_SECOND
    if estimated_secs > available:
        speed_ratio = estimated_secs / available
        rate_pct = int((speed_ratio - 1) * 100)
        rate = f'+{min(rate_pct, 80)}%'  # 最快加速 80%
    else:
        rate = '+0%'

    print(f"      文字: {char_count} 字, 可用: {available:.0f}s, 初始语速: {rate}")

    # 迭代生成，确保 fit 进时间窗口
    actual_dur = 0.0
    boundaries = []
    for attempt in range(max_retries):
        boundaries = asyncio.run(_tts_generate(text, voice, rate, audio_path))

        actual_dur = get_audio_duration(audio_path)
        print(f"      第{attempt+1}次: 时长 {actual_dur:.1f}s (目标 ≤{available:.0f}s), 语速 {rate}")

        if actual_dur <= available + 0.5:
            break  # 在目标范围内

        # 超时，加快语速
        speed_ratio = actual_dur / available
        rate_pct = int((speed_ratio - 1) * 100) + 5  # 多加 5% 余量
        rate = f'+{min(rate_pct, 80)}%'
    else:
        print(f"      ⚠️  {max_retries} 次后仍超时 ({actual_dur:.1f}s > {available:.0f}s)，使用最后结果")

    # 生成 SRT 字幕（逐句高亮）
    sentences = split_sentences(text)
    _generate_srt(boundaries, sentences, srt_path)

    print(f"      ✅ TTS: {actual_dur:.1f}s, 语速: {rate}, 字幕: {len(sentences)} 句")
    return audio_path, srt_path


def _generate_srt(boundaries: list[dict], sentences: list[str],
                  srt_path: str) -> None:
    """
    根据 Edge TTS 的 SentenceBoundary 时间戳，
    生成逐句同步的 SRT 字幕文件。
    """
    if not boundaries:
        with open(srt_path, 'w', encoding='utf-8') as f:
            pass
        return

    # SentenceBoundary 已经按句子返回时间戳，直接写入 SRT
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, b in enumerate(boundaries, 1):
            start_ms = b['offset_ms']
            end_ms = start_ms + b['duration_ms']
            f.write(f"{i}\n")
            f.write(f"{_ms_to_srt_time(start_ms)} --> {_ms_to_srt_time(end_ms)}\n")
            f.write(f"{b['text']}\n\n")


def _ms_to_srt_time(ms: int) -> str:
    """毫秒转 SRT 时间格式 HH:MM:SS,mmm"""
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    mil = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{mil:03d}"


def build_srt_subtitle_filter(srt_path: str, font_path: str = None,
                              output_width: int = DEFAULT_OUTPUT_WIDTH,
                              output_height: int = DEFAULT_OUTPUT_HEIGHT) -> str:
    """
    生成基于 SRT 文件的字幕滤镜（比 drawtext 更精准的时间控制）。

    使用 ffmpeg 的 subtitles 滤镜，支持字体、字号、位置等样式。
    为避免路径转义问题，先复制 SRT 到简单路径。
    """
    # 复制 SRT 到同目录下的简单文件名，避免 ffmpeg subtitles 滤镜的路径转义问题
    # 使用 PID 避免并发冲突
    simple_srt = os.path.join(os.path.dirname(srt_path), f'subs_{os.getpid()}.srt')
    shutil.copy2(srt_path, simple_srt)

    font_size = int(output_width * 0.055)
    # 构建 ASS 样式覆盖（ffmpeg filter chain 中逗号需要转义为 \,）
    # PlayResX/Y 设为视频分辨率，这样 FontSize 就是实际像素值
    style_parts = [
        f"PlayResX={output_width}",
        f"PlayResY={output_height}",
        f"FontSize={font_size}",
        "PrimaryColour=&H00FFFFFF",  # 白色文字 (AABBGGRR)
        "OutlineColour=&H00000000",  # 黑色描边
        "BackColour=&H80000000",     # 半透明黑色底框
        "BorderStyle=4",             # 底框+描边
        "Outline=1",
        "Shadow=0",
        "Alignment=2",               # 底部居中
        f"MarginV={int(output_width * 0.15)}",  # 底部边距
        f"MarginL={int(output_width * 0.08)}",  # 左右留白
        f"MarginR={int(output_width * 0.08)}",
    ]
    if font_path and os.path.exists(font_path):
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        style_parts.append(f"FontName={font_name}")

    # 用 \, 转义逗号，防止被 ffmpeg 解析为 filter chain 分隔符
    style = "\\,".join(style_parts)
    return f"subtitles={simple_srt}:force_style='{style}'"


# ============================================================
# Step ⑫: 音频标准化
# ============================================================

def normalize_audio(input_audio, output_audio, target_lufs=TARGET_LUFS):
    """
    将音频标准化到目标响度（默认 -14 LUFS，YouTube 标准）。

    两遍处理：
    1. 第一遍：分析当前响度
    2. 第二遍：应用 loudnorm 滤镜校正
    """
    # 第一遍：测量
    measure_cmd = [
        'ffmpeg', '-y', '-i', input_audio,
        '-af', f'loudnorm=I={target_lufs}:TP=-1:LRA=11:print_format=json',
        '-f', 'null', '-'
    ]
    result = subprocess.run(measure_cmd, capture_output=True, text=True)

    # 解析测量结果（如果第一遍失败，直接用简单模式）
    if result.returncode != 0:
        print("      第一遍测量失败，使用简单 loudnorm 模式")
        simple_cmd = [
            'ffmpeg', '-y', '-i', input_audio,
            '-af', f'loudnorm=I={target_lufs}:TP=-1:LRA=11',
            '-ar', '48000',
            '-c:a', 'aac', '-b:a', '192k',
            output_audio
        ]
        run_ffmpeg(simple_cmd, "音频标准化(简单模式)")
        return

    stderr = result.stderr
    try:
        # 找到 JSON 输出
        json_start = stderr.rfind('{')
        json_end = stderr.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            loudnorm_stats = json.loads(stderr[json_start:json_end])
            measured_i = loudnorm_stats.get('input_i', '-24')
            measured_tp = loudnorm_stats.get('input_tp', '-1')
            measured_lra = loudnorm_stats.get('input_lra', '11')
            measured_thresh = loudnorm_stats.get('input_thresh', '-30')

            print(f"      原始响度: {measured_i} LUFS")

            # 第二遍：应用校正
            normalize_cmd = [
                'ffmpeg', '-y', '-i', input_audio,
                '-af', (
                    f'loudnorm=I={target_lufs}:TP=-1:LRA=11:'
                    f'measured_I={measured_i}:measured_TP={measured_tp}:'
                    f'measured_LRA={measured_lra}:measured_thresh={measured_thresh}'
                ),
                '-ar', '48000',
                '-c:a', 'aac', '-b:a', '192k',
                output_audio
            ]
            run_ffmpeg(normalize_cmd, "音频标准化")
            print(f"      目标响度: {target_lufs} LUFS ✅")
            return
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # 如果解析失败，使用简单的 loudnorm
    print("      使用简单 loudnorm 模式")
    simple_cmd = [
        'ffmpeg', '-y', '-i', input_audio,
        '-af', f'loudnorm=I={target_lufs}:TP=-1:LRA=11',
        '-ar', '48000',
        '-c:a', 'aac', '-b:a', '192k',
        output_audio
    ]
    run_ffmpeg(simple_cmd, "音频标准化(简单模式)")


# ============================================================
# Step ⑬: 字幕叠加
# ============================================================

def find_cjk_font(user_font=None):
    """
    查找系统中支持 CJK（中日韩）字符的字体文件。

    Args:
        user_font: 用户通过 --font 指定的字体路径，优先使用
    Returns:
        str | None: 字体文件路径，找不到返回 None
    """
    if user_font:
        if os.path.exists(user_font):
            return user_font
        print(f"   ⚠️  指定字体不存在: {user_font}，尝试自动查找...")

    candidates = [
        # macOS
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
        # Linux (Noto CJK)
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc',
        # Linux (WenQuanYi)
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc',
        '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
        # Linux (Droid)
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        # Windows via WSL（书法风格优先，更搭「墨」主题）
        '/mnt/c/Windows/Fonts/simkai.ttf',     # 楷体
        '/mnt/c/Windows/Fonts/msyh.ttc',       # 微软雅黑
        '/mnt/c/Windows/Fonts/msyhbd.ttc',     # 微软雅黑 粗体
        '/mnt/c/Windows/Fonts/simhei.ttf',     # 黑体
        '/mnt/c/Windows/Fonts/simsun.ttc',     # 宋体
        '/mnt/c/Windows/Fonts/msjh.ttc',       # 微软正黑
        '/mnt/c/Windows/Fonts/mingliub.ttc',   # 细明体
        '/mnt/c/Windows/Fonts/malgun.ttf',     # Malgun Gothic (韩文但含CJK)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # fc-list fallback: 搜索任意 CJK 字体
    try:
        result = subprocess.run(
            ['fc-list', ':lang=zh', 'file'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().split('\n')[0]
            font_path = first_line.split(':')[0].strip()
            if os.path.exists(font_path):
                return font_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def build_subtitle_filter(text, output_width, output_height, font_path=None):
    """
    生成字幕叠加的 ffmpeg drawtext 滤镜。

    字幕位置：画面底部 15% 处，居中，白色文字，半透明黑色背景。
    需要指定支持 CJK 的字体文件，否则汉字会显示为方框。

    Args:
        font_path: 用户指定的字体路径，None 时自动查找
    """
    # 转义特殊字符
    escaped = (text.replace("\\", "\\\\")
                    .replace("'", "'\\''")
                    .replace(":", "\\:")
                    .replace("%", "\\%"))

    font_size = int(output_width * 0.055)  # 字号约为画面宽度的 5.5%
    y_pos = int(output_height * 0.85)
    box_padding = int(font_size * 0.4)

    # 查找 CJK 字体
    font_path = find_cjk_font(font_path)
    # ffmpeg drawtext 需要对路径中的 : \ ' 转义
    if font_path:
        escaped_font = font_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
        font_opt = f":fontfile='{escaped_font}'"
    else:
        font_opt = ""
    if font_path:
        print(f"   字体: {font_path}")
    else:
        print("   ⚠️  未找到 CJK 字体，中文可能显示为方框")
        print("   安装: sudo apt install fonts-noto-cjk (Ubuntu) / brew install font-noto-sans-cjk (macOS)")

    # drawtext 滤镜
    return (
        f"drawtext=text='{escaped}'"
        f"{font_opt}"
        f":fontsize={font_size}"
        f":fontcolor=white"
        f":x=(w-text_w)/2"
        f":y={y_pos}"
        f":box=1"
        f":boxcolor=black@0.5"
        f":boxborderw={box_padding}"
    )


# ============================================================
# Step ⑭: 缩略图生成
# ============================================================

def generate_thumbnail(video_path, output_path, timestamp=None):
    """
    旧版缩略图：直接从视频提取一帧。仅作兼容保留。
    新代码请用 generate_calligraphy_thumbnail()。
    """
    if timestamp is None:
        info = get_video_info(video_path)
        timestamp = info['duration'] * 0.6

    run_ffmpeg([
        'ffmpeg', '-y', '-ss', str(timestamp),
        '-i', video_path,
        '-vframes', '1', '-update', '1',
        '-q:v', '2',
        output_path
    ], "缩略图生成")


def generate_calligraphy_thumbnail(clean_frame_processed_path, output_path,
                                    output_width=DEFAULT_OUTPUT_WIDTH,
                                    output_height=DEFAULT_OUTPUT_HEIGHT,
                                    thumb_fill=0.78,
                                    medium='auto'):
    """
    从已处理的干净帧（已裁剪、缩放、色彩校正）生成精美书法缩略图。

    输入是 1080x1920 的处理后帧（字在白纸上，无手、无字幕）。
    针对 Shorts/小红书优化：
    1. 在处理后帧中检测墨迹 → 二次紧裁剪（字占画面 ~78%）
    2. 缩放回 1080x1920
    3. 强化对比度（纯白纸 + 浓墨黑）
    4. 强锐化
    5. 轻微暗角（引导视线聚焦中心）
    """
    import cv2
    import numpy as np

    frame = cv2.imread(clean_frame_processed_path)
    if frame is None:
        print("   ⚠️  无法读取帧")
        return False

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- 在已处理帧中检测笔迹 ---
    # brush: 全局阈值 80（毛笔墨色深）
    # pencil: 自适应阈值（局部对比度，对铅笔/淡墨稳健）
    # auto: 先 brush 后 pencil
    def _brush_thumb():
        _, im = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        im = cv2.erode(im, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        im = cv2.dilate(im, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
        cs, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cs, im

    def _pencil_thumb():
        block_size = max(31, (min(h, w) // 20) | 1)
        im = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            block_size, 15
        )
        # 去除 ruled paper 横/竖线
        line_len = max(50, min(h, w) // 30)
        h_lines = cv2.morphologyEx(im, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1)))
        v_lines = cv2.morphologyEx(im, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len)))
        lines = cv2.dilate(cv2.bitwise_or(h_lines, v_lines),
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        im = cv2.bitwise_and(im, cv2.bitwise_not(lines))
        im = cv2.morphologyEx(im, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        im = cv2.dilate(im, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
        cs, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cs, im

    if medium == 'brush':
        attempts = [_brush_thumb]
    elif medium == 'pencil':
        attempts = [_pencil_thumb]
    else:
        attempts = [_brush_thumb, _pencil_thumb]

    contours, ink_mask = [], None
    for fn in attempts:
        cs, im = fn()
        if any(cv2.contourArea(c) > 100 for c in cs):
            contours, ink_mask = cs, im
            break
    else:
        contours, ink_mask = cs, im  # type: ignore[possibly-undefined]

    # 排除帧边缘伪影：色彩校正/锐化会在帧边产生 1-2px 暗条，
    # 被聚类拉到 y=0 时会让二次裁剪偏向画面顶部。
    edge = 3

    def _is_edge_artifact(c):
        cx, cy, cw_, ch_ = cv2.boundingRect(c)
        touches_edge = (cx <= edge or cy <= edge
                        or cx + cw_ >= w - edge or cy + ch_ >= h - edge)
        aspect = max(cw_, ch_) / max(1, min(cw_, ch_))
        return touches_edge and aspect > 4  # 紧贴边缘且细长 → 伪影

    if not contours:
        print("   ⚠️  缩略图未检测到墨迹，使用中心裁剪")
        ink = {'cx': w // 2, 'cy': h // 2, 'w': w // 3, 'h': h // 4}
    else:
        # 空间聚类：以最大轮廓为锚点
        valid = sorted([c for c in contours if not _is_edge_artifact(c)],
                       key=cv2.contourArea, reverse=True)
        if not valid:
            valid = sorted(contours, key=cv2.contourArea, reverse=True)
        anchor = valid[0]
        ax, ay, aw, ah = cv2.boundingRect(anchor)
        acx, acy = ax + aw // 2, ay + ah // 2
        radius = max(aw, ah) * 2
        clustered = [c for c in valid[:30]
                     if abs(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] // 2 - acx) < radius
                     and abs(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2 - acy) < radius
                     and cv2.contourArea(c) > 30]
        if clustered:
            valid = clustered
        pts = np.vstack(valid)
        x, y, bw, bh = cv2.boundingRect(pts)
        ink = {'cx': x + bw // 2, 'cy': y + bh // 2, 'w': bw, 'h': bh}

    # --- 二次紧裁剪：字占画面 thumb_fill ---
    # 留 5% 边距，避免裁入帧边缘的桌面/纸边噪点
    margin_y = int(h * 0.02)
    crop = calculate_crop(
        ink, w, h, thumb_fill, output_width, output_height,
        min_scale=1.0, y_min=margin_y, y_max=h - margin_y,
    )

    # --- ffmpeg：裁剪 + 缩放 + 调色 + 强锐化 ---
    # 毛笔（墨色 <50）：linear eq brightness+contrast 仍然最优
    # 铅笔（字 180-210、纸 220-240）：linear eq 会把笔画和纸面一起推到 240+
    #   → 改用 curves 做非线性映射：把 180 灰度段推到 25，纸面 >250 保持不动。
    # 对照点：0/0（黑不变）, 0.5/0.15（中灰压到深灰）, 0.7/0.1（180→25）, 1/1（白不变）
    if medium == 'pencil':
        tone_filter = "curves=all='0/0 0.5/0.15 0.7/0.1 1/1'"
    else:
        tone_filter = "eq=brightness=0.10:contrast=1.5:saturation=0.9"
    vf_parts = [
        f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}",
        f"scale={output_width}:{output_height}:flags=lanczos",
        tone_filter,
        "unsharp=5:5:1.5:3:3:0.0,unsharp=3:3:0.7:3:3:0.0",
    ]

    run_ffmpeg([
        'ffmpeg', '-y', '-i', clean_frame_processed_path,
        '-vf', ','.join(vf_parts),
        '-q:v', '1',
        output_path
    ], "缩略图生成")

    return True


# ============================================================
# 裁剪参数计算
# ============================================================

def calculate_crop(ink, src_w, src_h, fill_ratio, out_w, out_h,
                   min_scale=DEFAULT_MIN_SCALE, y_min=0, y_max=0):
    """
    根据墨迹位置和目标填充比例，计算裁剪区域。
    保持输出的 9:16 宽高比。
    保证最小放大倍数（即使墨迹区域很大也会放大）。
    裁剪框不会延伸到 y_min 之上或 y_max 之下（避免包含书脊）。

    Returns:
        dict: {'x', 'y', 'w', 'h', 'scale_factor'}
    """
    if y_max <= 0:
        y_max = src_h
    aspect = out_w / out_h

    cw = int(ink['w'] / fill_ratio)
    ch = int(ink['h'] / fill_ratio)

    # 保持宽高比
    if cw / ch > aspect:
        ch = int(cw / aspect)
    else:
        cw = int(ch * aspect)

    cw = min(cw, src_w)
    ch = min(ch, src_h)

    # 保证最小放大倍数：如果裁剪区太大导致 scale < min_scale，缩小裁剪区
    if min_scale > 1.0 and out_w / cw < min_scale:
        cw = int(out_w / min_scale)
        ch = int(cw / aspect)
        cw = min(cw, src_w)
        ch = min(ch, src_h)

    # 裁剪高度不超过可用区域
    available_h = y_max - y_min
    if ch > available_h:
        ch = available_h
        cw = int(ch * aspect)
    cw -= cw % 2  # ffmpeg 要求偶数
    ch -= ch % 2

    cx = max(0, min(ink['cx'] - cw//2, src_w - cw))
    cy = max(y_min, min(ink['cy'] - ch//2, y_max - ch))

    return {
        'x': cx, 'y': cy, 'w': cw, 'h': ch,
        'scale_factor': out_w / cw
    }


# ============================================================
# 主处理流程
# ============================================================

def process_video(input_path, output_path, voiceover_path=None,
                  target_fill_ratio=DEFAULT_FILL_RATIO,
                  hold_seconds=DEFAULT_HOLD_SECONDS,
                  output_width=DEFAULT_OUTPUT_WIDTH,
                  output_height=DEFAULT_OUTPUT_HEIGHT,
                  force_rotation=None,
                  enable_stabilize=True,
                  enable_color_correct=True,
                  fade_in=DEFAULT_FADE_IN,
                  fade_out=DEFAULT_FADE_OUT,
                  subtitle_text=None,
                  tts_text=None,
                  tts_voice=DEFAULT_TTS_VOICE,
                  font_path=None,
                  generate_thumb=True):
    """
    主处理函数：完成 14 步流水线。
    """

    print("=" * 60)
    print(f"极客禅·墨 视频处理器 v{VERSION}")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    clean_frame_raw = None
    clean_frame_processed = None  # 干净帧处理后路径（用于缩略图）

    try:
        # ========================================
        # Step 1: 视频信息
        # ========================================
        print("\n📹 Step 1: 分析视频...")
        vi = get_video_info(input_path)
        print(f"   尺寸: {vi['width']}x{vi['height']}, 时长: {vi['duration']:.1f}s, 帧率: {vi['fps']:.0f}fps")

        # ========================================
        # Step 2: 方向检测 (Feature 6)
        # ========================================
        print("\n🔄 Step 2: 检测方向...")
        if force_rotation is not None:
            rotation = force_rotation
            print(f"   手动: {rotation}")
        else:
            orient = detect_orientation(input_path)
            rotation = orient['rotation']
            print(f"   结果: {orient['reason']} ({orient['confidence']})")
            if orient['confidence'] == 'low':
                print("   ⚠️  置信度低，可用 --rotate 手动指定")

        # 90° 旋转后宽高互换
        if rotation in ('cw90', 'ccw90'):
            eff_w, eff_h = vi['height'], vi['width']
        else:
            eff_w, eff_h = vi['width'], vi['height']
        print(f"   旋转后尺寸: {eff_w}x{eff_h}")

        # ========================================
        # Step 3: 防抖
        # ========================================
        stabilized_input = input_path
        if enable_stabilize:
            print("\n📐 Step 3: 视频防抖...")
            has_vidstab = check_vidstab_support()
            if has_vidstab:
                # 第一遍：检测运动
                transforms_file = os.path.join(tmpdir, 'transforms.trf')
                # tripod=1: 固定机位模式，只修正微小晃动，不会把手部运动误判为相机抖动
                detect_cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-vf', f'vidstabdetect=shakiness=3:accuracy=9:tripod=1:result={transforms_file}',
                    '-f', 'null', '-'
                ]
                result = subprocess.run(detect_cmd, capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(transforms_file):
                    # 第二遍：应用稳定（tripod=1 保持原始帧位置，只做微调）
                    stabilized_path = os.path.join(tmpdir, 'stabilized.mp4')
                    stabilize_cmd = [
                        'ffmpeg', '-y', '-i', input_path,
                        '-vf', f'vidstabtransform=input={transforms_file}:tripod=1:zoom=0',
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                        '-pix_fmt', 'yuv420p', '-an',
                        stabilized_path
                    ]
                    result2 = subprocess.run(stabilize_cmd, capture_output=True, text=True)
                    if result2.returncode == 0:
                        stabilized_input = stabilized_path
                        print("   ✅ 防抖完成")
                    else:
                        print("   ⚠️  防抖应用失败，使用原始视频")
                else:
                    print("   ⚠️  防抖检测失败，使用原始视频")
            # else: 已在 check_vidstab_support 中打印提示
        else:
            print("\n📐 Step 3: 防抖（已跳过）")

        # ========================================
        # Step 4: 墨迹检测
        # ========================================
        print("\n🔍 Step 4: 检测墨迹...")
        ink = detect_ink_region(stabilized_input, rotation)

        # ========================================
        # Step 5: 计算裁剪参数（旋转后书脊永远在顶部 → 排除顶部 30%）
        # ========================================
        crop_y_min = int(eff_h * 0.38)
        crop_y_max = eff_h
        crop = calculate_crop(
            ink, eff_w, eff_h,
            target_fill_ratio, output_width, output_height,
            y_min=crop_y_min, y_max=crop_y_max,
        )
        sf = crop['scale_factor']

        print(f"\n✂️  Step 5: 裁剪参数")
        print(f"   区域: ({crop['x']},{crop['y']}) {crop['w']}x{crop['h']}")
        print(f"   放大: {sf:.2f}x, 目标字占比: ~{target_fill_ratio*100:.0f}%")

        # ========================================
        # Step 6-8: 构建一体化滤镜链
        # （旋转 + 裁剪 + 缩放 + 白平衡 + 锐化 + 淡入）
        # ========================================
        print(f"\n🎬 Step 6-8: 视频处理...")
        vf_parts = []

        # ① 旋转
        rotation_names = {'none': '不旋转', '180': '180°', 'cw90': '顺时针90°', 'ccw90': '逆时针90°'}
        if rotation == '180':
            vf_parts.append("vflip,hflip")
            print(f"   ✓ 旋转 180°")
        elif rotation == 'cw90':
            vf_parts.append("transpose=1")
            print(f"   ✓ 顺时针旋转 90°")
        elif rotation == 'ccw90':
            vf_parts.append("transpose=2")
            print(f"   ✓ 逆时针旋转 90°")

        # ④ 裁剪
        vf_parts.append(f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}")
        print(f"   ✓ 裁剪居中")

        # ⑤ 缩放
        vf_parts.append(f"scale={output_width}:{output_height}:flags=lanczos")
        print(f"   ✓ 缩放 {output_width}x{output_height}")

        # ⑥ 白平衡/亮度校正
        if enable_color_correct:
            vf_parts.append(build_color_correction_filter())
            print("   ✓ 白平衡校正")

        # ⑦ 锐化（始终应用，放大后尤其重要）
        sharp = build_sharpen_filter(sf)
        vf_parts.append(sharp)
        print(f"   ✓ 锐化 ({sharp.split('=')[0]})")

        # ⑧ 淡入
        if fade_in > 0:
            vf_parts.append(f"fade=t=in:st=0:d={fade_in}")
            print(f"   ✓ 淡入 {fade_in}s")

        # 合并为一条滤镜链
        vf_filter = ",".join(vf_parts)

        # ⑨ 去除背景音 + 一次性处理
        cropped_video = os.path.join(tmpdir, 'processed.mp4')
        run_ffmpeg([
            'ffmpeg', '-y', '-i', stabilized_input,
            '-vf', vf_filter,
            '-an',  # 去除所有音频
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            cropped_video
        ], "主视频处理")
        print("   ✅ 视频处理完成")

        # ========================================
        # Step 10: 拼接静止画面（无手检测）
        # ========================================
        current_video = cropped_video

        if hold_seconds > 0:
            print(f"\n⏸️  Step 10: 静止画面 ({hold_seconds}s)...")

            # 找到干净的最后一帧（无手）
            print("   检测无手帧...")
            clean_frame = find_clean_last_frame(stabilized_input, rotation)

            # 保存干净帧（已在 find_clean_last_frame 中旋转），用 ffmpeg 处理以保持一致
            import cv2
            clean_frame_raw = os.path.join(tmpdir, 'clean_raw.png')
            cv2.imwrite(clean_frame_raw, clean_frame)

            # 对帧应用相同的裁剪+缩放+色彩校正
            clean_frame_processed = os.path.join(tmpdir, 'clean_processed.png')
            vf_still = []
            vf_still.append(f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}")
            vf_still.append(f"scale={output_width}:{output_height}:flags=lanczos")
            if enable_color_correct:
                vf_still.append(build_color_correction_filter())
            vf_still.append(build_sharpen_filter(sf))

            run_ffmpeg([
                'ffmpeg', '-y', '-i', clean_frame_raw,
                '-vf', ','.join(vf_still),
                clean_frame_processed
            ], "处理静止帧")

            # 生成静止视频
            hold_video = os.path.join(tmpdir, 'hold.mp4')
            create_still_video(clean_frame_processed, hold_video, hold_seconds, vi['fps'])

            # 拼接
            merged = os.path.join(tmpdir, 'merged.mp4')
            concat_videos([cropped_video, hold_video], merged)
            current_video = merged
            print("   ✅ 静止画面完成")

        # ========================================
        # Step 11.5: TTS 语音合成（如果提供了文本）
        # ========================================
        tts_audio_path = None
        tts_srt_path = None

        if tts_text and not voiceover_path:
            print(f"\n🗣️  Step 11.5: TTS 语音合成...")
            cur_dur = get_video_info(current_video)['duration']
            # 旁白与视频「同时」播放（混音于 t=0），不是接在视频后。
            # 预算 = 视频长度，让旁白以自然语速覆盖书写过程；
            # 上限 SHORTS_MAX_DURATION-1 防止延长后超 60s。
            tts_budget = min(cur_dur, SHORTS_MAX_DURATION - 1)
            print(f"   视频: {cur_dur:.0f}s, Shorts上限: {SHORTS_MAX_DURATION}s, 旁白预算: {tts_budget:.0f}s")
            tts_audio_path, tts_srt_path = generate_tts(
                tts_text, tts_budget, tmpdir,
                voice=tts_voice,
            )
            # TTS 生成的音频作为旁白
            voiceover_path = tts_audio_path

        # ========================================
        # Step 11 + 13: 淡出 + 字幕（合并为单次编码，减少世代损失）
        # ========================================
        # Step 12: 合并旁白（先延长视频再合并，确保淡出在最终时长上计算）
        # ========================================
        if voiceover_path and os.path.exists(voiceover_path):
            print(f"\n🎙️  Step 12: 合并旁白...")

            # 12a: 标准化音频响度
            print("   标准化响度...")
            normalized_audio = os.path.join(tmpdir, 'normalized.m4a')
            normalize_audio(voiceover_path, normalized_audio)

            # 12b: 检查时长匹配
            vid_dur = get_video_info(current_video)['duration']
            vo_dur = get_audio_duration(normalized_audio)
            print(f"   视频: {vid_dur:.1f}s, 旁白: {vo_dur:.1f}s")

            # 如果旁白比视频长，延长静止画面（此时还没淡出，帧是亮的）
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

            # 12c: 合并音频（stream copy 视频）
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
            if not voiceover_path:
                print(f"\n📝 Step 12: 无旁白，输出静音视频")

        # 保存无字幕视频路径，用于缩略图（只展示字，不含字幕文字）
        thumb_source_video = current_video

        # ========================================
        # Step 11 + 13: 淡出 + 字幕（在最终时长上执行，单次编码）
        # 放在音频合并之后，确保淡出基于完整时长计算
        # ========================================
        post_filters = []

        if fade_out > 0:
            print(f"\n🌑 Step 11: 淡出 ({fade_out}s)...")
            cur_info = get_video_info(current_video)
            fade_start = cur_info['duration'] - fade_out
            post_filters.append(f"fade=t=out:st={fade_start:.2f}:d={fade_out}")
            print(f"   淡出起始: {fade_start:.1f}s (总时长 {cur_info['duration']:.1f}s)")

        # 字幕：优先用 TTS 生成的 SRT（时间同步），否则用 --subtitle 静态文字
        if tts_srt_path and os.path.exists(tts_srt_path):
            print(f"\n📝 Step 13: 叠加同步字幕（SRT）...")
            sub_filter = build_srt_subtitle_filter(tts_srt_path, font_path, output_width, output_height)
            post_filters.append(sub_filter)
            print("   ✅ SRT 字幕准备就绪")
        elif subtitle_text:
            print(f"\n📝 Step 13: 叠加字幕...")
            print(f"   \"{subtitle_text}\"")
            sub_filter = build_subtitle_filter(subtitle_text, output_width, output_height, font_path)
            post_filters.append(sub_filter)
            print("   ✅ 字幕准备就绪")
        else:
            print(f"\n📝 Step 13: 字幕（已跳过）")

        # 一次编码应用所有后期滤镜
        if post_filters:
            post_video = os.path.join(tmpdir, 'post_processed.mp4')
            run_ffmpeg([
                'ffmpeg', '-y', '-i', current_video,
                '-vf', ','.join(post_filters),
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'copy',
                '-pix_fmt', 'yuv420p',
                post_video
            ], "淡出+字幕处理")
            current_video = post_video
            print("   ✅ 后期处理完成")

        # ========================================
        # 输出最终文件
        # ========================================
        shutil.copy2(current_video, output_path)

        # ========================================
        # Step 14: 缩略图（可选）
        # ========================================
        if generate_thumb:
            thumb_path = os.path.splitext(output_path)[0] + '_thumb.jpg'
            print(f"\n🖼️  Step 14: 生成缩略图...")
            # 优先用原始分辨率的干净帧（紧裁剪 + 强化对比 + 暗角）
            if clean_frame_processed and os.path.exists(clean_frame_processed):
                ok = generate_calligraphy_thumbnail(clean_frame_processed, thumb_path,
                                                     output_width, output_height)
            else:
                ok = False
            if not ok:
                generate_thumbnail(thumb_source_video, thumb_path)
            print(f"   ✅ {thumb_path}")

        # ========================================
        # 最终报告
        # ========================================
        fi = get_video_info(output_path)
        sz = os.path.getsize(output_path) / 1024 / 1024

        print(f"\n{'=' * 60}")
        print(f"✅ 处理完成!")
        print(f"{'=' * 60}")
        print(f"   文件: {output_path}")
        print(f"   尺寸: {fi['width']}x{fi['height']}")
        print(f"   时长: {fi['duration']:.1f}s")
        print(f"   大小: {sz:.1f}MB")

        steps_done = []
        if rotation != 'none': steps_done.append(f"🔄{rotation_names[rotation]}")
        if enable_stabilize: steps_done.append("📐防抖")
        steps_done.append("✂️裁剪")
        steps_done.append(f"🔎放大{sf:.1f}x")
        if enable_color_correct: steps_done.append("🎨色彩")
        steps_done.append("🔍锐化")
        if fade_in > 0: steps_done.append(f"▶️淡入{fade_in}s")
        if fade_out > 0: steps_done.append(f"⏹淡出{fade_out}s")
        steps_done.append("🔇去音")
        if hold_seconds > 0: steps_done.append(f"⏸停留{hold_seconds}s")
        if voiceover_path: steps_done.append("🎙️旁白-14LUFS")
        if subtitle_text: steps_done.append("📝字幕")

        print(f"   处理: {' → '.join(steps_done)}")
        print(f"   {'📱 Shorts可用 (≤60s)' if fi['duration'] <= SHORTS_MAX_DURATION else '⚠️ 超60s'}")
        print(f"{'=' * 60}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=f'极客禅·墨 视频处理器 v{VERSION}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（自动检测方向）
  python3 ink_video_processor.py raw.mp4 -o output.mp4

  # 完整用法（旁白 + 字幕 + 强制翻转）
  python3 ink_video_processor.py raw.mp4 -v voice.m4a -o final.mp4 \\
      --flip --subtitle "吃茶去。——赵州禅师"

  # 自定义参数
  python3 ink_video_processor.py raw.mp4 -v vo.m4a -o final.mp4 \\
      --fill 0.65 --hold 5 --fade-in 0.3 --fade-out 1.5

  # 跳过色彩校正（加速处理）
  python3 ink_video_processor.py raw.mp4 -o fast.mp4 --no-color-correct

  # 手持拍摄时启用防抖
  python3 ink_video_processor.py raw.mp4 -o stable.mp4 --stabilize

  # 高清输出
  python3 ink_video_processor.py raw.mp4 -o hd.mp4 --width 1080 --height 1920
        """
    )

    # 输入输出
    parser.add_argument('input', help='原始视频文件路径')
    parser.add_argument('--output', '-o', default='output.mp4', help='输出文件路径')
    parser.add_argument('--voiceover', '-v', help='旁白音频文件 (m4a/mp3/wav)')

    # 画面参数
    parser.add_argument('--fill', '-f', type=float, default=DEFAULT_FILL_RATIO,
                        help=f'字占画面比例 (默认: {DEFAULT_FILL_RATIO})')
    parser.add_argument('--hold', type=float, default=DEFAULT_HOLD_SECONDS,
                        help=f'停留秒数 (默认: {DEFAULT_HOLD_SECONDS})')
    parser.add_argument('--width', type=int, default=DEFAULT_OUTPUT_WIDTH,
                        help=f'输出宽度 (默认: {DEFAULT_OUTPUT_WIDTH})')
    parser.add_argument('--height', type=int, default=DEFAULT_OUTPUT_HEIGHT,
                        help=f'输出高度 (默认: {DEFAULT_OUTPUT_HEIGHT})')

    # 旋转控制
    parser.add_argument('--rotate',
                        choices=['none', '180', 'cw90', 'ccw90'],
                        help='手动指定旋转方式（覆盖自动检测）')
    flip = parser.add_mutually_exclusive_group()
    flip.add_argument('--flip', action='store_true', help='强制翻转 180°（等同 --rotate 180）')
    flip.add_argument('--no-flip', action='store_true', help='强制不旋转（等同 --rotate none）')

    # 效果开关
    parser.add_argument('--fade-in', type=float, default=DEFAULT_FADE_IN,
                        help=f'淡入时长秒 (默认: {DEFAULT_FADE_IN}, 0=关闭)')
    parser.add_argument('--fade-out', type=float, default=DEFAULT_FADE_OUT,
                        help=f'淡出时长秒 (默认: {DEFAULT_FADE_OUT}, 0=关闭)')
    parser.add_argument('--stabilize', action='store_true',
                        help='启用防抖（默认关闭，固定机位不需要）')
    parser.add_argument('--no-color-correct', action='store_true', help='跳过白平衡校正')
    parser.add_argument('--no-thumbnail', action='store_true', help='不生成缩略图')

    # 字幕
    parser.add_argument('--subtitle', help='叠加的金句字幕文字（静态，无语音）')
    parser.add_argument('--font', default=DEFAULT_FONT,
                        help=f'字幕字体文件路径 (默认: {DEFAULT_FONT})')

    # TTS 语音合成（自动朗读 + 同步字幕）
    tts_group = parser.add_mutually_exclusive_group()
    tts_group.add_argument('--text', help='TTS 旁白文本（直接输入）')
    tts_group.add_argument('--text-file', help='TTS 旁白文本文件路径')
    parser.add_argument('--voice', default=DEFAULT_TTS_VOICE,
                        help=f'TTS 语音 (默认: {DEFAULT_TTS_VOICE})')

    args = parser.parse_args()

    # 验证
    if not os.path.exists(args.input):
        print(f"❌ 找不到: {args.input}"); sys.exit(1)
    if args.voiceover and not os.path.exists(args.voiceover):
        print(f"❌ 找不到: {args.voiceover}"); sys.exit(1)
    if not 0.1 <= args.fill <= 0.95:
        print(f"❌ fill 应在 0.1~0.95 之间"); sys.exit(1)

    # 读取 TTS 文本
    tts_text = None
    if args.text:
        tts_text = args.text
    elif args.text_file:
        if not os.path.exists(args.text_file):
            print(f"❌ 找不到文本文件: {args.text_file}"); sys.exit(1)
        with open(args.text_file, 'r', encoding='utf-8') as f:
            tts_text = f.read().strip()

    if tts_text and args.voiceover:
        print("⚠️  同时提供了 --text 和 --voiceover，使用 voiceover 音频（忽略 TTS）")
        tts_text = None

    # --rotate 优先；否则 --flip/--no-flip 兼容
    if args.rotate:
        force_rotation = args.rotate
    elif args.flip:
        force_rotation = '180'
    elif args.no_flip:
        force_rotation = 'none'
    else:
        force_rotation = None

    check_dependencies()

    process_video(
        input_path=args.input,
        output_path=args.output,
        voiceover_path=args.voiceover,
        target_fill_ratio=args.fill,
        hold_seconds=args.hold,
        output_width=args.width,
        output_height=args.height,
        force_rotation=force_rotation,
        enable_stabilize=args.stabilize,
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
