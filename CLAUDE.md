# Shorts — 极客禅·墨 视频后处理器

Part of the [YT-Factory](../CLAUDE.md) platform. Processes handwritten calligraphy videos into YouTube Shorts.

## Two Processors

| Processor | Source | File | Key Difference |
|-----------|--------|------|----------------|
| **Phone** | `ink_video_processor.py` | Phone (back camera, upside-down) | Auto-rotation (180°/90° CW/CCW) based on book edge detection |
| **Webcam** | `webcam_ink_processor.py` | Webcam (4K/1080p, top-down) | No rotation; row/column brightness scan to isolate paper from desk |
| **XHS Cover** | `xhs_cover.py` | Thumbnail → 小红书封面 (1242×1660) | Ink extraction + anchor-based layout + simplified rendering |

`webcam_ink_processor.py` imports shared functions (ffmpeg utils, TTS, subtitles, color correction, sharpening, etc.) from `ink_video_processor.py`.

## Architecture

### Phone Pipeline (`ink_video_processor.py`)

```
Raw video (phone, any orientation) → auto-detect rotation → ink_video_processor.py → YouTube Shorts (9:16, 1080x1920)
```

```
① Rotation detection (4-edge scan) → ② Stabilize (opt) → ③ Ink detect → ④ Crop → ⑤ Scale
→ ⑥ Color correct → ⑦ Sharpen → ⑧ Fade in → ⑨ Strip audio → ⑩ Hold frame
→ ⑪.5 TTS (opt) → ⑫ Merge voiceover → ⑪ Fade out → ⑬ Subtitles → ⑭ Thumbnail
```

### Webcam Pipeline (`webcam_ink_processor.py`)

```
Raw video (webcam, 1920x1080, no rotation) → paper + ink detection → webcam_ink_processor.py → YouTube Shorts (9:16, 1080x1920)
```

```
① Paper region detect (adaptive threshold) → ② Ink detect (contour mask) → ③ Crop → ④ Scale
→ ⑤ Color correct → ⑥ Sharpen → ⑦ Fade in → ⑧ Strip audio → ⑨ Hold frame
→ ⑩ TTS/voiceover (opt) → ⑪ Fade out → ⑫ Subtitles → ⑬ Thumbnail
```

**Critical ordering**: Fade-out must happen AFTER video extension, not before. Otherwise extended frames are black.

### Key Design Decisions

**Shared:**
- **Color correction**: `eq` filter in YUV space (brightness/contrast only), NOT `curves=all` (distorts skin tones).
- **Sharpening**: Dual-pass unsharp mask. Strength scales with zoom factor.
- **TTS**: Edge TTS (free, no API key). Auto-adjusts speaking rate to fit within Shorts time budget.
- **Font**: Default 楷体 (simkai.ttf) via WSL Windows fonts path.

**Phone-specific:**
- **Rotation detection**: Scans all 4 edges (top/bottom/left/right) for book spine features (dark pixels + edge density). Maps book edge → rotation to bring book to top. Supports 180°, 90° CW, 90° CCW.
- **Ink detection**: Global threshold(55) + white paper RGB mask(>150). Book at top after rotation → exclude top 30%.
- **Rotation**: Uses `vflip,hflip` for 180° (pixel-exact), `transpose=1`/`transpose=2` for 90° CW/CCW.

**Webcam-specific:**
- **Paper detection**: Row/column median brightness scan (>180 = paper, <180 = desk). No morphological operations — robust against desk stripe width at any resolution.
- **Ink detection**: Dark pixels (threshold=80) within paper contour mask. Much lower min area (100px vs 0.3%) since webcam characters are small relative to frame.
- **No rotation**: Webcam video is always correctly oriented.
- **Crop constraint bug fixed**: `calculate_crop` now uses `y_max - ch` (not `src_h - ch`) to prevent crop from overflowing paper bounds into desk area.

**XHS Cover (`xhs_cover.py`):**
- **Ink extraction**: Character-first strategy — find dark contours, filter desk stripes (aspect>5 + frame-spanning), spatial clustering around largest stroke.
- **Rendering**: Simplified — background replacement to #F5F0EB with 3px Gaussian feathering, optional ×0.9 linear darkening. No Gamma/soft-mask/sharpening (trust input quality).
- **Layout**: Anchor-based — divider line fixed at 52%, character bottom aligned 3% above divider, title at 58%, subtitle at 66%. All anchors fixed regardless of character size.
- **Size unification**: Single char = 48% canvas width, multi-char = 70% width. `--char` length determines mode.

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
# Phone: one-shot (import → process → XHS cover → export video+thumb+cover)
make short-full IN=guan.mp4 CHAR="观" TITLE="观自在菩萨"
make short-full IN=guan.mp4 CHAR="观" TITLE="标题" TEXT="旁白"  # With TTS

# Webcam: one-shot
make webcam-full IN=xi.mp4 CHAR="息" TITLE="它是意识和无意识之间的桥"

# Process only (no import/export/cover)
make short IN=zhi.mp4
make webcam IN=xi.mp4

# XHS cover only (from existing thumb)
make xhs-cover IN=xi.mp4 CHAR="息" TITLE="标题"

# Options
make shorts-help / make webcam-help / make xhs-help

# Shared
make short-import IN=zhi.mp4                     # Win Downloads → WSL input
make short-export IN=zhi.mp4                     # WSL output → Win Downloads
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
