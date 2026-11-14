# MovingPattern

`MovingPattern.dll` (Linux: `libmovingpattern.so`) is an AviSynth+ source filter that makes synthetic moving test
clips of any size, frame rate, scan and planar YUV / Y format. It needs AviSynth+ 3.6 or newer and nothing else.
UConvert's benchmark (`uconvert/bench/`) uses it as its source.

Every output sample is computed at its own position and instant. In an interlaced clip each field is its own
instant (two per frame), so the clip carries real interlacing, as a camera records it. Sizes and speeds are in
units of the picture height: a pattern looks the same at every size, only sharper or coarser.

```
MovingPattern(int "width" = 1920, int "height" = 1080, string "pixel_type" = "YUV422P10",
              int "fps_num" = 25, int "fps_den" = 1, string "scan" = "tff",
              int "frames" = period, int "period" = 4 s, string "pattern" = "motion", string "storage" = "memory",
              int "matrix" = 1, int "primaries" = 1, int "transfer" = 1, int "range" = 1,
              int "chroma_location" = 0, int "sar_num" = 1, int "sar_den" = 1, int "seed" = 1, int "threads" = 0)
```

| argument | default | what it does |
| --- | --- | --- |
| `width`, `height` | 1920, 1080 | picture size; a multiple of the chroma subsampling (interlaced: height a multiple of twice the vertical subsampling) |
| `pixel_type` | `"YUV422P10"` | planar YUV 4:2:0 / 4:2:2 / 4:4:4 or Y, 8, 10, 12 or 16 bit (`"YUV420P8"` / `"YV12"`, `"YUV422P10"`, `"YUV444P16"`, `"Y10"`, ...) |
| `fps_num`, `fps_den` | 25, 1 | frame rate (frames, not fields) |
| `scan` | `"tff"` | `"tff"` / `"bff"`: interlaced, each field at its own instant (top or bottom field first); `"p"`: progressive |
| `frames` | `period` | length of the clip; the content repeats every `period` frames |
| `period` | 4 s of frames | frames generated (and kept, see `storage`); at least 2 |
| `pattern` | `"motion"` | the content, see below |
| `storage` | `"memory"` | `"memory"`: the whole period is generated when the clip is created (on all threads) and kept in RAM; frames are returned without copying. `"disk"`: generated into a file in the OS temp folder and read back per frame; the file is removed when the clip is destroyed, and by the OS if the process ends without that (Windows: delete-on-close; Linux: unlinked right after creation). `"none"`: every frame is rendered when it is requested |
| `matrix`, `primaries`, `transfer` | 1, 1, 1 | frame properties `_Matrix`, `_Primaries`, `_Transfer` (ITU-T H.273 codes; 1 = BT.709). They label the clip; the content is the same |
| `range` | 1 | `_ColorRange`: 1 limited (luma 16-235, chroma 16-240 at 8 bit, scaled for higher depths), 0 full |
| `chroma_location` | 0 | `_ChromaLocation` of subsampled formats (0 left, 1 centre, 2 top left, 3 top, 4 bottom left, 5 bottom); horizontal centre siting (1, 3, 5) moves the chroma samples half a luma sample to the right |
| `sar_num`, `sar_den` | 1, 1 | `_SARNum`, `_SARDen` |
| `seed` | 1 | other textures and block patterns |
| `threads` | 0 | threads for generating the period (0: all) |

Every frame also carries `_FieldBased` (0 progressive, 1 bff, 2 tff) and `_DurationNum` / `_DurationDen`.
Memory per period: frames x frame size, e.g. 100 frames 1080i25 YUV422P10 = 0.8 GB; for large pictures use a
shorter `period`, `storage="disk"` or `"none"`.

## Patterns

- `"motion"` (default): a stress scene for deinterlacing, motion estimation and frame-rate conversion. A texture
  with a 1/f amplitude spectrum (like natural pictures) pans at sub-pixel speed; an inset zooms into a second
  texture; a bar with 2-line horizontal stripes (interlace twitter) and 1/34 H columns and a disc with a block
  texture moving with it cross the picture, the disc in front of the bar (occlusion); luma grain (+-1 % of the
  range, new for every instant). At half the period a hard cut leads to a second scene with other textures,
  directions and zoom.
- `"pan"`: only the panning background of `"motion"` (with its cut), no grain.
- `"still"`: the background texture of the first scene, not moving, no grain.

## Examples

```
LoadPlugin("MovingPattern.dll")
MovingPattern()                                                    # 1080i25 YUV422P10, 4 s, kept in memory
MovingPattern(3840, 2160, "YUV420P10", 50, 1, "p", frames=500, period=50)   # UHD 50p, 10 s from a 1 s period
MovingPattern(720, 576, "YV12", scan="bff", matrix=5, primaries=5, transfer=6, sar_num=16, sar_den=15)
```
