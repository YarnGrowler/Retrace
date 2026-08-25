# Retrace

Retrace turns a PNG or JPG drawing into a natural-looking speed-drawing video.

## Why Retrace is different

Most simple draw-on tools expect a small number of clean vector paths. Retrace starts with the finished PNG or JPG and reconstructs a drawing process from it. That means it can work with highly detailed illustrations containing thousands of small marks, preserve the original grayscale appearance, and show the pencil moving through the artwork.

The result is not just an image sliding onto the screen—it is a generated sequence of individual drawing paths that rebuilds the PNG.

## Use case

Retrace is useful for portfolio demos, art-process videos, visual explanations, and experiments with computer vision and human-like motion.

## City demo — 25 seconds

![Inline 25-second city drawing preview](output/city-preview.gif)

The preview plays directly in the README. The full-resolution H.264 version is [`output/city.mp4`](output/city.mp4).

Input: [`city.png`](city.png)

## Command-line parameters

Run `python draw.py --help` to see these options:

| Parameter | Default | What it controls | Example |
| --- | --- | --- | --- |
| `image` | required | Input PNG, JPG, or other OpenCV-readable image | `city.png` |
| `--duration` | `25` | Target video length in seconds; use `auto` to give every stroke visible time | `--duration 40` |
| `--fps` | `30` | Output frames per second | `--fps 60` |
| `--output` | `output/<image>.mp4` | Exact output video path | `--output renders/city.mp4` |
| `--threshold` | `auto` | Ink threshold; `auto` uses Otsu, or pass `0`–`255` | `--threshold 180` |
| `--cursor` | `pencil` | Cursor style: `pencil`, `dot`, or `none` | `--cursor dot` |
| `--seed` | `42` | Seed for repeatable ordering and rendering details | `--seed 7` |
| `--debug` | off | Prints extra processing statistics | `--debug` |
| `--no-preview` | off | Disables the live preview window | `--no-preview` |

## Installation and running

Retrace is currently a source-based Python project rather than a published pip package. Install it with a virtual environment:

```powershell
git clone https://github.com/YarnGrowler/Retrace.git
cd Retrace
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python draw.py city.png --duration 25 --fps 30 --no-preview
```

The output is written to `output/city.mp4`. For automatic timing, use:

```powershell
python draw.py city.png --duration auto --no-preview
```

For a custom pencil and output path, use:

```powershell
python draw.py city.png --duration 35 --cursor pencil --output renders/city.mp4
```

## In one sentence

**Give Retrace a finished PNG; it reconstructs a detailed, pencil-led drawing video from the pixels.**
