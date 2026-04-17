# Shorts — 极客禅·墨 视频后处理器

Part of the [YT-Factory](../CLAUDE.md) platform. Processes handwritten calligraphy videos into YouTube Shorts.

## Two Processors

| Processor | Source | File | Key Difference |
|-----------|--------|------|----------------|
| **Phone** | `ink_video_processor.py` | Phone (back camera, upside-down) | Auto-rotation (180°/90° CW/CCW) based on book edge detection |
| **Webcam** | `webcam_ink_processor.py` | Webcam (4K/1080p, top-down) | No rotation; Otsu + connected-components paper detection; skin-masked ink detection |
| **XHS Cover (Brush)** | `xhs_cover.py` | Thumbnail → 小红书封面 (1242×1660) | Ink extraction + anchor-based layout + render_ink rendering |
| **XHS Cover (Pencil)** | `xhs_cover_pencil.py` | Raw cover frame → 小红书封面 (1242×1660) | Sigmoid background replacement, preserves pencil gray tones |

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
- **Shared module (`ink_extraction.py`)**: `flat_field_correct` (illumination normalization), `background_subtract_mask` (Otsu on diff with min-floor), `classify_medium` (histogram-based brush/pencil/empty), `remove_ruled_lines` (two-layer: projection detect + narrow subtract). Zero-cost on plain white paper.
- **Writing medium (`--medium`)**: `brush` / `pencil` / `auto` (default). `auto` runs `classify_medium` on the darkest 1% of flat-field-corrected paper pixels: median < 55 → brush, ≥ 200 → empty (error), else pencil. Replaces the fragile "try brush, fall back on pencil" loop that was fooled by finger shadows.
  - **brush**: Global threshold 80 on `flat_gray`. Success requires ≥1 contour with area > 1000 AND median gray < 60 (real ink, not shadow).
  - **pencil**: `bg_subtract_mask ∪ threshold<200` on flat_gray — union handles both ruled-paper (bg_subtract wins) and pure white paper (bg_subtract's dilation pulls pencil into the "clean paper" estimate and misses ~90% of strokes; simple threshold rescues). Then `remove_ruled_lines` (projection-detect + ±3px subtract), single-pixel connected-component filter, dense-cluster focus picking the blob with highest `mask_density / aspect^1.5`. Returns the cluster bbox as one synthetic contour (stops downstream area>100 filter from shredding thin pencil strokes).
  - **Manual override (`--char-region`)**: Skip auto-detection, specify character region directly. Format: `"cx,cy"` (auto-sizes to ¼ frame), `"cx,cy,w,h"`, or `"x:y:w:h"`. Numbers <1 = frame fraction; ≥1 = pixels.
- **Paper detection**: Otsu + 25×25 morphological opening + largest connected component. Light-invariant, works when paper occupies <50% of the frame.
- **Paper mask erode scales with resolution**: `max(15, min(h,w)//60)` kernel × 2 iterations. 1080p → 36px total; 4K → 72px. Fixed-15px was letting wood-grain texture bleed past the paper boundary at 4K.
- **Skin masking**: A dilated YCrCb skin mask is subtracted from the ink mask. The writer's hand is in nearly every webcam frame; without this, the hand becomes the largest "ink" contour.
- **Paper-edge artifact filter**: Contours within 30 px of paper edge AND aspect ratio > 2.5 are dropped (paper folds/shadows).
- **Clean frame search — two separate paths (video vs cover)**:
  1. `find_clean_last_frame_webcam`: last 90 frames, adaptive brightness floor (85% of window max) to skip fade-outs and dim-paper sources, then 8-frame median composite of the lowest-skin candidates (relative ranking — no absolute 2% threshold). Used for the **hold-still video tail**.
  2. `find_best_cover_frame`: **full-video scan** (1.5 fps sample + dense tail of last 30 frames). Writers often lift hand briefly only in the last <1s; pure sampling would miss that. Metric is **skin coverage over the character bbox** (direct "hand off the character" signal), NOT ink_area — hand/arm shadow creates false ink signal that inflates ink_area when the hand is on the character. `char_ink ≥ 500` is an existence check. Top-8 by skin_over_char get BGR-reread and median-composited. Returns None if <3 candidates; caller falls back to hold-still frame. Adds ~8s to a 25s 4K video.
- **Crop constraint bug fixed**: `calculate_crop` uses `y_max - ch` (not `src_h - ch`).
- **X-direction crop clamping preserves aspect ratio** when paper is narrower than the calculated crop.
- **Debug images conditional**: `--debug` controls `paper_debug.png` / `ink_webcam_debug.png`.

**Pencil thumbnail rendering (`generate_calligraphy_thumbnail` in `ink_video_processor.py`, used by webcam too):**
- Brush keeps the historical `eq=brightness=0.10:contrast=1.5`. Pencil would wash out under that (strokes at 180-210 + paper at 220+ → both pushed to 240+).
- Pencil switches to `curves=all='0/0 0.5/0.15 0.7/0.1 1/1'` — non-linear map: 128→38, 180→25, paper near-255 untouched. Preserves faint strokes while keeping paper clean.

**XHS Cover (`xhs_cover.py`):**
- **Ink extraction**: Stage A white-paper detect → **Stage A.5 flat-field correct** → Stage B `classify_medium` + brush/pencil extract on `flat_gray`. Pencil uses simple `flat<220` threshold (works because flat-field pushes paper to ~255) plus `remove_ruled_lines`.
- **Alpha from `flat_cropped`, not `cropped`** — flat-field has already removed paper shadows, so alpha directly reflects ink density (no false-ink shadow residue).
- **CLAHE on pencil's alpha source**: `cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))` applied to `flat_cropped` before computing alpha. Pencil strokes often land in 230-245, giving alpha=10-25 (near-transparent); CLAHE pulls them to 60-200, alpha becomes meaningfully opaque. Brush skips CLAHE (already has strong contrast, CLAHE would amplify paper-fiber noise).
- **Quality gate**: `p95 - p1` of `flat_cropped` measures real contrast (p95-p5 undershoots for sparse faint strokes because p5 stays near paper). `<25` raises ValueError with remediation hints; `<55` on pencil sets `auto_override_style='pencil-bold'`.
- **Render priority**: user `--cover-style` > `auto_override_style` > medium default (brush→brush, pencil→pencil-zen).
- **`render_ink` adaptive thresholds**: `_bg_floor = p95 - 3`, `_ink_cap = max(p5_of_dark_pixels + 5, _bg_floor - 50)`. Hardcoded 205/190 would zero-out `ink_weight` on pencil thumbs (strokes at 230+ all above 205), making any dynamic-range stretch a no-op. Adaptive thresholds ride the actual histogram.
- **`render_ink` cover styles**:
  - `'brush'` — x0.9 microcontrast, no range stretch
  - `'pencil-zen'` (pencil default) — adaptive target_dark: darkest<100→55 (lightly stretched "禅意灰"), <160→40, ≥160→25 (effectively pencil-bold)
  - `'pencil-bold'` — always target_dark=25 (max contrast, brush-like)
- **Layout**: divider at 52%, character bottom 3% above divider, title 58%, subtitle 66%. Fixed anchors regardless of char size.
- **Size**: single char = 48% canvas width, multi-char = 70% (`--char` length determines mode).
- **Title auto-sizing**: reduces by char count, then by rendered width until within 90% canvas width.

**XHS Cover Pencil (`xhs_cover_pencil.py`):**
- **Input source**: `xxx_cover_frame.png` — the raw composite frame (crop+scale+color+sharpen) BEFORE `generate_calligraphy_thumbnail`'s pencil-curves processing. NOT the thumb (which has pencil gray crushed to near-black).
- **Design**: "方向 A 极简留白" — pure BG_COLOR background (no texture noise), character floats directly on white.
- **Rendering**: Sigmoid background replacement + `*0.8` darkening. No render_ink, no alpha-based tone, no ink_color. RGB channels carry the original pencil grayscale directly.
- **Character detection**: Adaptive threshold (C=12) + connected-component spatial filter (area<200 deleted; 200-800 kept if near anchor, deleted if far; >800 always kept).
- **Paper cleanup**: Grayscale snap (result_mean > bg_mean-45 → snap to BG_COLOR) + morphological open (5x5 ellipse) removes isolated paper texture survivors. No ink_detect mask dependency.
- **Edge blend**: 8% bbox-edge feather ramp blends sigmoid result toward BG_COLOR at bbox boundary.
- **Per-channel BG_COLOR**: Sigmoid targets (245, 240, 235) per-channel, not a single gray mean — eliminates color-mismatch rectangle.
- **Layout**: Same as brush version — char center at 37%, single char 28% width, golden ratio positioning.
- **Makefile routing**: `phone-full` defaults to pencil cover; only `MEDIUM=brush` routes to `xhs_cover.py`. Standalone: `make pencil-cover`.
- **`phone_ink_processor.py` change**: Saves `xxx_cover_frame.png` (copy of `cover_processed` before tmpdir cleanup) for pencil cover input.

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
# Phone pencil: one-shot (import → process → pencil cover → export)
make phone-full IN=she.mp4 CHAR="舍" TITLE="舍得之间" TF=/tmp/she.txt ROTATE=cw

# Phone brush: one-shot (same but routes to xhs_cover.py)
make phone-full IN=zhi.mp4 CHAR="知" TITLE="标题" MEDIUM=brush

# Old phone pipeline (ink_video_processor.py, not phone_ink_processor.py)
make short-full IN=guan.mp4 CHAR="观" TITLE="观自在菩萨"
make short-full IN=guan.mp4 CHAR="观" TITLE="标题" TEXT="旁白"  # With TTS

# Webcam: one-shot
make webcam-full IN=xi.mp4 CHAR="息" TITLE="它是意识和无意识之间的桥"

# Webcam: pencil mode (flat-field + bg_subtract∪threshold + ruled-line removal)
make webcam-full IN=she.mp4 CHAR="舍" TITLE="标题" MEDIUM=pencil

# Webcam: faint pencil on lined paper — manually specify character region
# REGION="cx,cy" (center, auto-sized) | "cx,cy,w,h" | "x:y:w:h" — numbers <1 = frame fraction
make webcam-full IN=she.mp4 CHAR="舍" TITLE="标题" MEDIUM=pencil REGION="0.55,0.5"

# Override cover rendering style (A/B testing pencil treatments)
# pencil-zen (default, adaptive) | pencil-bold (max contrast) | brush
make webcam-full IN=she.mp4 CHAR="舍" TITLE="标题" MEDIUM=pencil COVER_STYLE=pencil-bold

# Process only (no import/export/cover)
make short IN=zhi.mp4
make webcam IN=xi.mp4

# XHS cover only — brush (from existing thumb)
make xhs-cover IN=xi.mp4 CHAR="息" TITLE="标题"

# XHS cover only — pencil (from existing cover_frame)
make pencil-cover IN=she.mp4 CHAR="舍" TITLE="标题"

# Options
make shorts-help / make webcam-help / make xhs-help

# Debug (saves detection overlay images)
cd shorts && uv run python3 webcam_ink_processor.py files/input/xi.mp4 -o files/output/xi.mp4 --debug

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
- Pencil cover depends on the writer briefly lifting hand at end. `find_best_cover_frame` requires ≥3 frames with low skin_over_char; if the writer keeps their hand on the page the entire video, even the best-picked frames will have the hand occluding the character and median composite can't remove it.
- For extremely faint pencil (reflectance diff <15%), the bbox may still underestimate the full character; use `--char-region` to override.
- `main.py` is a scaffold placeholder, not used
