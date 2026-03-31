# Shorts — 极客禅·墨 视频后处理器

将手机倒置拍摄的书法写字视频，自动处理为 YouTube Shorts 竖屏短视频。

> 拍摄方式：手机倒置，后置摄像头朝向自己，从上方俯拍桌面书写过程（后置摄像头画质更好）。

## 功能

14 步全自动流水线：

| 步骤 | 功能 | 技术 |
|------|------|------|
| 1 | 视频分析 | ffprobe 提取元数据 |
| 2 | 方向检测/翻转 | 上下区域深色像素+边缘密度对比 |
| 3 | 防抖 | vidstabdetect + vidstabtransform (tripod模式, 默认关闭) |
| 4 | 墨迹检测 | 二值化+白纸遮罩+形态学+轮廓检测 |
| 5 | 裁剪居中 | 以字为中心, 保持9:16宽高比, 最小放大1.5x |
| 6 | 白平衡/亮度 | eq滤镜 (YUV空间, 不影响肤色) |
| 7 | 锐化 | 双通道unsharp mask (轮廓+细节) |
| 8 | 淡入 | 开头0.5s渐显 |
| 9 | 去背景音 | 移除原始音频 |
| 10 | 静止画面 | 肤色检测找到无手帧, 拼接4s停留 |
| 11 | 淡出 | 结尾1s渐隐 (在最终时长上计算) |
| 11.5 | TTS语音 | Edge TTS自动朗读+逐句同步字幕 |
| 12 | 合并旁白 | loudnorm -14 LUFS两遍标准化 |
| 13 | 字幕叠加 | SRT同步字幕或静态drawtext |
| 14 | 缩略图 | 自动提取60%位置帧 |

## 快速开始

```bash
# 安装依赖
uv sync

# 系统依赖
sudo apt install ffmpeg fonts-noto-cjk  # Ubuntu/WSL

# 基本用法（自动检测方向）
uv run python3 ink_video_processor.py raw.mp4 -o output.mp4

# 完整用法（TTS旁白 + 同步字幕）
uv run python3 ink_video_processor.py raw.mp4 -o final.mp4 \
    --text "吃茶去。赵州禅师对每个来访者只说三个字。"

# 从文件读取旁白
uv run python3 ink_video_processor.py raw.mp4 -o final.mp4 \
    --text-file voiceover.txt

# 手动旁白 + 静态字幕
uv run python3 ink_video_processor.py raw.mp4 -v voice.m4a -o final.mp4 \
    --flip --subtitle "吃茶去。——赵州禅师"

# 高清输出（默认已是1080x1920）
uv run python3 ink_video_processor.py raw.mp4 -o hd.mp4 --width 1080 --height 1920
```

## 所有参数

```
用法: ink_video_processor.py [-h] [-o OUTPUT] [-v VOICEOVER] [-f FILL]
                             [--hold HOLD] [--width WIDTH] [--height HEIGHT]
                             [--flip | --no-flip] [--fade-in FADE_IN]
                             [--fade-out FADE_OUT] [--stabilize]
                             [--no-color-correct] [--no-thumbnail]
                             [--subtitle SUBTITLE] [--font FONT]
                             [--text TEXT | --text-file TEXT_FILE]
                             [--voice VOICE]
                             input

参数说明:
  input                 原始视频文件路径
  -o, --output          输出文件路径 (默认: output.mp4)
  -v, --voiceover       旁白音频文件 (m4a/mp3/wav)
  -f, --fill            字占画面比例 (默认: 0.6)
  --hold                写完后停留秒数 (默认: 4)
  --width               输出宽度 (默认: 1080)
  --height              输出高度 (默认: 1920)
  --flip                强制翻转 180°
  --no-flip             强制不翻转
  --fade-in             淡入时长 (默认: 0.5, 0=关闭)
  --fade-out            淡出时长 (默认: 1.0, 0=关闭)
  --stabilize           启用防抖（默认关闭, 固定机位不需要）
  --no-color-correct    跳过白平衡校正
  --no-thumbnail        不生成缩略图
  --subtitle            静态金句字幕文字
  --font                字幕字体路径 (默认: 楷体 simkai.ttf)
  --text                TTS 旁白文本（自动朗读+同步字幕）
  --text-file           TTS 旁白文本文件
  --voice               TTS 语音 (默认: zh-CN-YunxiNeural)
```

## TTS 语音选项

| 语音 ID | 性别 | 风格 |
|---------|------|------|
| `zh-CN-YunxiNeural` | 男 | 知性稳重 (默认) |
| `zh-CN-YunjianNeural` | 男 | 播音腔 |
| `zh-CN-XiaoxiaoNeural` | 女 | 温柔自然 |
| `zh-CN-XiaoyiNeural` | 女 | 活泼 |

## 拍摄建议

- **分辨率**: 1080p 或 4K（原始分辨率越高, 裁剪放大后越清晰）
- **固定机位**: 手机倒置在支架上, 后置摄像头朝向自己俯拍
- **光线**: 均匀白光, 避免强侧光（减少阴影）
- **纸张**: 白色无格纸, 放在深色书本上方
- **书写**: 字写在纸的中间偏下位置, 远离书脊

## 技术细节

- **墨迹检测**: 全局阈值(55) + 白纸RGB遮罩(>150) 排除书脊印刷字
- **裁剪约束**: 裁剪框不会延伸到书脊区域 (y_min/y_max)
- **防抖**: tripod模式, 只修正微小晃动, 不把手部运动误判为抖动
- **翻转**: vflip+hflip (像素精确, 不引入亚像素抖动)
- **色彩**: eq滤镜在YUV空间调亮度/对比度, 不影响肤色色相
- **TTS**: Edge TTS (免费, 无需API Key), 自动语速控制 fit 进 Shorts 时间窗口
- **字幕同步**: SentenceBoundary 时间戳 → SRT → ffmpeg subtitles 滤镜

## 依赖

- Python 3.14+
- ffmpeg (含 libvidstab, libass)
- opencv-python-headless
- numpy
- edge-tts
