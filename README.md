# Retrace

Retrace turns a finished raster drawing into a natural-looking speed-drawing video. It extracts ink, skeletonizes it into drawable paths, orders those paths, and progressively reveals the original grayscale pixels as the pencil travels. The goal is an actual drawing process—not a broad wipe, paint reveal, or a handful of manually authored vector paths.

## Examples

### City — 25 seconds

<video src="output/city.mp4" controls width="900"></video>

[Download or play the 25-second H.264 city video](output/city.mp4)

Input: [`city.png`](city.png) · Output: [`output/city.mp4`](output/city.mp4)

### Town

[Download or play the town example](output/town.mp4)

Input: [`town.png`](town.png) · Output: [`output/town.mp4`](output/town.mp4)

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python draw.py town.png
```

Use a fixed duration:

```powershell
python draw.py city.png --duration 25 --fps 30 --no-preview
```

Use automatic timing when every small stroke needs visible pencil time:

```powershell
python draw.py town.png --duration auto
```

The H.264 video is written as `output/<input-name>.mp4`. The original grayscale artwork controls the final appearance, while the extracted trajectories control when each pixel becomes visible. `--debug` writes intermediate binary, skeleton, stroke-order, and reconstruction-difference images.

## Why shaded drawings are harder

The city and town images are mostly line art, so a one-pixel skeleton is a reasonable description of the source. A train, motorcycle, or heavily shaded illustration is different: a dark patch may be an outline, a shadow, cross-hatching, or several overlapping surfaces. If all of those pixels are skeletonized together, the tracer can create long parallel paths and the renderer can leave large blank patches until a later pass. That produces the horizontal “eraser streak” appearance.

The next improvement should be a shading-aware intermediate representation:

1. Separate contours, hatching, and filled-tone regions before skeletonization.
2. Divide dense tone regions into small connected patches with a coverage score.
3. Finish each patch locally before moving to a distant patch.
4. Use local stroke ownership inside a patch while preserving the source grayscale values.
5. Add an ordering check that prevents a region from being left with unresolved coverage debt.

This keeps the faithful 2D result as the default while making a future depth-aware or 2.5D mode possible.
