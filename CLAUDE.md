# Shorts — 极客禅·墨 视频后处理器

Part of the [YT-Factory](../CLAUDE.md) platform. Processes handwritten calligraphy videos into YouTube Shorts.

## Two Processors

| Processor | Source | File | Key Difference |
|-----------|--------|------|----------------|
| **Phone** | `ink_video_processor.py` | Phone (back camera, upside-down) | Auto-rotation (180°/90° CW/CCW) based on book edge detection |
| **Webcam** | `webcam_ink_processor.py` | Webcam (4K/1080p, top-down) | No rotation; Otsu + connected-components paper detection; skin-masked ink detection |
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
  - **TTS budget = video duration, NOT remaining time to 60s.** Audio is mixed at `t=0` and plays simultaneously with the video, not appended after. Setting budget to `SHORTS_MAX_DURATION - cur_dur` (the old bug) would compress 121 chars into ~15s for a 45s video, finishing the narration when only ~36% of writing was done. Correct budget is `min(cur_dur, SHORTS_MAX_DURATION - 1)`.
- **Calligraphy thumbnail (`generate_calligraphy_thumbnail`)**: Second-pass tight crop on the clean processed frame. Filters out edge-touching contours with extreme aspect ratio (color-correction artifacts at frame borders) before clustering, otherwise they get pulled into the bbox and shift the crop toward the frame edge.
- **Font**: Default 楷体 (simkai.ttf) via WSL Windows fonts path.

**Phone-specific:**
- **Rotation detection**: Scans all 4 edges (top/bottom/left/right) for book spine features (dark pixels + edge density). Maps book edge → rotation to bring book to top. Supports 180°, 90° CW, 90° CCW.
- **Ink detection**: Global threshold(55) + white paper RGB mask(>150). Book at top after rotation → exclude top 30%.
- **Rotation**: Uses `vflip,hflip` for 180° (pixel-exact), `transpose=1`/`transpose=2` for 90° CW/CCW.

**Webcam-specific:**
- **Paper detection**: **Otsu threshold + 25×25 morphological opening + largest connected component.** Replaced the row/column median scan because median fundamentally fails when paper covers <50% of frame area (4K source with paper in one corner) — desk pixels dominate the median and the algorithm collapses to a tiny "purely white" strip. Otsu is light-invariant; CC finds the largest white blob regardless of size.
- **Ink detection**: Dark pixels (threshold=80) within the paper contour mask, with two extra filters:
  1. **Skin masking**: A dilated YCrCb skin mask is *subtracted* from the ink mask. The writer's hand is in nearly every webcam frame and its finger shadows/nail outlines fall below the dark threshold; without this, the hand becomes the largest "ink" contour and the cluster anchors on the wrong location.
  2. **Paper-edge artifact filter**: Contours that touch the paper bound (within 30 px) **and** have aspect ratio > 2.5 are dropped. Paper-edge folds/shadows form long thin strips along the edge (e.g. `hui.mp4` had a 42×163 strip with area ~5500 — bigger than any character stroke at 4K) that would otherwise win the largest-area anchor selection.
- **Clean frame search (`find_clean_last_frame_webcam`)**: Two-pass search across the last 90 frames with **adaptive** brightness threshold (skip frames below 85% of the search-window max). Necessary for two cases:
  1. Re-processing a previously-processed video (its trailing 30 frames are a fade-out — all 0% skin but black);
  2. Sources where the paper itself is dim (e.g. `dao_v2.mp4` max brightness is only ~190, so a fixed 220 threshold would reject every frame). The adaptive ratio handles both `xi.mp4` (max ≈ 247) and `dao_v2.mp4` (max ≈ 190) without per-source tuning.
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
