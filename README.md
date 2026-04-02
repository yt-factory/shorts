# Shorts — 极客禅·墨 视频后处理器

将书法写字视频自动处理为 YouTube Shorts 竖屏短视频 + 小红书封面图。

## 两种处理器

| 处理器 | 脚本 | 输入源 | 特点 |
|--------|------|--------|------|
| **Phone** | `ink_video_processor.py` | 手机（后置摄像头，倒置/侧置） | 自动旋转检测（180°/90° CW/CCW） |
| **Webcam** | `webcam_ink_processor.py` | 电脑摄像头（1080p，俯拍） | 白纸轮廓检测，隔离桌面纹理 |

另有 `xhs_cover.py` 从缩略图生成小红书风格封面。

## 快速开始

```bash
# 安装依赖
make install   # 或: cd shorts && uv sync

# 系统依赖
sudo apt install ffmpeg fonts-noto-cjk  # Ubuntu/WSL
```

### 一条龙（推荐）

```bash
# Phone: 从 Win Downloads 导入 → 处理 → 生成XHS封面 → 全部导出回 Win
make short-full IN=guan.mp4 CHAR="观" TITLE="观自在菩萨" TEXT="旁白文本"

# Webcam: 同上
make webcam-full IN=xi.mp4 CHAR="息" TITLE="它是意识和无意识之间的桥"

# 输出到 Win Downloads: 视频.mp4 + 视频_thumb.jpg + 视频_cover.jpg
```

### 分步操作

```bash
# 导入导出
make short-import IN=zhi.mp4                    # Win Downloads → WSL
make short-export IN=zhi.mp4                    # WSL → Win Downloads

# Phone 处理
make short IN=zhi.mp4                           # 基本处理
make short IN=zhi.mp4 TEXT="旁白"               # 处理 + TTS + 同步字幕
make short IN=zhi.mp4 TF=voiceover.txt          # TTS 从文本文件

# Webcam 处理
make webcam IN=xi.mp4                           # 基本处理
make webcam IN=xi.mp4 TEXT="旁白"               # 处理 + TTS + 同步字幕

# 小红书封面（需先生成 thumb）
make xhs-cover IN=xi.mp4 CHAR="息" TITLE="标题"

# 查看所有参数
make shorts-help                                # Phone 参数
make webcam-help                                # Webcam 参数
make xhs-help                                   # XHS 封面参数
```

### 直接调用脚本

```bash
cd shorts

# Phone
uv run python3 ink_video_processor.py files/input/guan.mp4 -o files/output/guan.mp4 \
    --text "旁白文本"

# Webcam
uv run python3 webcam_ink_processor.py files/input/xi.mp4 -o files/output/xi.mp4

# 小红书封面
uv run python3 xhs_cover.py --thumb files/output/xi_thumb.jpg \
    --char "息" --title "标题" -o files/output/xi_cover.jpg
```

## Phone 处理器参数

```
ink_video_processor.py [-h] [-o OUTPUT] [-v VOICEOVER] [-f FILL]
                       [--hold HOLD] [--width WIDTH] [--height HEIGHT]
                       [--rotate {none,180,cw90,ccw90}]
                       [--flip | --no-flip]
                       [--fade-in FADE_IN] [--fade-out FADE_OUT]
                       [--stabilize] [--no-color-correct] [--no-thumbnail]
                       [--subtitle SUBTITLE] [--font FONT]
                       [--text TEXT | --text-file TEXT_FILE]
                       [--voice VOICE]
                       input
```

旋转控制：
- `--rotate cw90|ccw90|180|none` — 手动指定旋转方式
- `--flip` / `--no-flip` — 向后兼容（= `--rotate 180` / `--rotate none`）
- 默认自动检测：扫描画面四边，找到书脊位置，自动选择旋转角度

## Webcam 处理器参数

```
webcam_ink_processor.py [-h] [-o OUTPUT] [-v VOICEOVER] [-f FILL]
                        [--hold HOLD] [--width WIDTH] [--height HEIGHT]
                        [--fade-in FADE_IN] [--fade-out FADE_OUT]
                        [--no-color-correct] [--no-thumbnail]
                        [--subtitle SUBTITLE] [--font FONT]
                        [--text TEXT | --text-file TEXT_FILE]
                        [--voice VOICE]
                        input
```

## XHS 封面生成器

```bash
# 单张
python3 xhs_cover.py --thumb thumb.jpg --char "茶" --title "赵州禅师只说三个字" -o cover.jpg

# 批量（JSON配置）
python3 xhs_cover.py --batch covers.json -o output_dir/
```

批量配置格式：
```json
[
  {"thumb": "tea_thumb.jpg", "char": "茶", "title": "赵州禅师只说三个字"},
  {"thumb": "kong_thumb.jpg", "char": "空", "title": "空不是没有，是无限可能"}
]
```

## TTS 语音选项

| 语音 ID | 性别 | 风格 |
|---------|------|------|
| `zh-CN-YunxiNeural` | 男 | 知性稳重 (默认) |
| `zh-CN-YunjianNeural` | 男 | 播音腔 |
| `zh-CN-XiaoxiaoNeural` | 女 | 温柔自然 |
| `zh-CN-XiaoyiNeural` | 女 | 活泼 |

## 拍摄建议

### Phone
- 手机倒置/侧置在支架上，后置摄像头朝向自己俯拍
- 书本放在纸张一侧，系统自动检测书本位置并旋转
- 白色无格纸，均匀白光

### Webcam
- 电脑摄像头从上方俯拍桌面
- 白纸放在桌面上，书本在左侧
- 系统自动检测白纸轮廓，隔离桌面纹理

## 依赖

- Python 3.14+ (via uv)
- ffmpeg (含 libvidstab, libass)
- opencv-python-headless, numpy, edge-tts, Pillow
- Noto Serif/Sans CJK SC 字体 (XHS 封面用)
