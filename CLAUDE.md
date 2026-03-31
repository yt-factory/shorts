# Shorts — 极客禅·墨 视频后处理器

Part of the [YT-Factory](../CLAUDE.md) platform. Processes handwritten calligraphy videos into YouTube Shorts.

## Architecture

Single-file Python CLI (`ink_video_processor.py`, ~1500 lines, 28 functions) with a 14-step processing pipeline.

```
Raw video (phone upside-down, back camera facing user) → ink_video_processor.py → YouTube Shorts (9:16, 1080x1920)
```

### Pipeline Flow

```
① Flip detection → ② Stabilize (opt) → ③ Ink detect → ④ Crop → ⑤ Scale
→ ⑥ Color correct → ⑦ Sharpen → ⑧ Fade in → ⑨ Strip audio → ⑩ Hold frame
→ ⑪.5 TTS (opt) → ⑫ Merge voiceover → ⑪ Fade out → ⑬ Subtitles → ⑭ Thumbnail
```

**Critical ordering**: Fade-out must happen AFTER video extension (Step 12), not before. Otherwise extended frames are black.

### Key Design Decisions

- **Ink detection**: Global threshold(55) + white paper RGB mask(>150) to exclude book spine text. Spatial clustering via largest-contour anchor to remove scattered noise.
- **Crop constraint**: `y_min`/`y_max` prevents crop from extending into book area, even when character is written close to book spine.
- **Flip**: Uses `vflip,hflip` (pixel-exact), NOT `rotate=PI` (causes sub-pixel jitter).
- **Stabilization**: Disabled by default (fixed camera). When enabled, uses `tripod=1` mode.
- **Color correction**: `eq` filter in YUV space (brightness/contrast only), NOT `curves=all` (distorts skin tones).
- **Sharpening**: Dual-pass unsharp mask (large radius for contour + small radius for fine detail). Strength scales with zoom factor.
- **TTS**: Edge TTS (free, no API key). Auto-adjusts speaking rate to fit within Shorts time budget. Generates SRT with SentenceBoundary timestamps.
- **Font**: Default 楷体 (simkai.ttf) via WSL Windows fonts path.

### Constants

| Constant | Value | Notes |
|----------|-------|-------|
| OUTPUT_WIDTH | 1080 | Shorts standard |
| OUTPUT_HEIGHT | 1920 | 9:16 aspect |
| FILL_RATIO | 0.6 | Character fills 60% of crop width |
| MIN_SCALE | 1.5 | Minimum zoom factor |
| HOLD_SECONDS | 4 | Static frame after writing |
| TARGET_LUFS | -14 | YouTube loudness standard |
| SHORTS_MAX_DURATION | 60 | YouTube Shorts limit |
| TTS_VOICE | zh-CN-YunxiNeural | Edge TTS male voice |
| FONT | simkai.ttf | 楷体 via WSL |

## Quick Reference

```bash
uv sync                          # Install dependencies
uv run python3 ink_video_processor.py --help   # All options
uv run python3 ink_video_processor.py raw.mp4 -o out.mp4  # Basic
uv run python3 ink_video_processor.py raw.mp4 -o out.mp4 --text "旁白文本"  # With TTS
```

## Environment

- Python 3.14+ (via uv)
- WSL2 on Windows (accesses Windows fonts at /mnt/c/Windows/Fonts/)
- ffmpeg with libvidstab and libass support
- No API keys required (Edge TTS is free)

## Known Limitations

- Book spine detection relies on fixed 30% exclusion zone; books taller than 30% of frame may still appear
- Ink detection may include book text when character is written very close to book spine (mitigated by white paper mask but not perfect)
- Edge TTS only returns SentenceBoundary (not WordBoundary) for Chinese — subtitles are per-sentence, not per-word
- `main.py` is a scaffold placeholder, not used
