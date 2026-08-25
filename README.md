# Retrace

Retrace turns a finished raster drawing into a natural-looking speed-drawing video. It extracts ink, skeletonizes it into drawable paths, orders those paths, and progressively reveals the original grayscale pixels as the pencil travels.

## Use case

Retrace is for artists, developers, and researchers who want to present a finished black-and-white illustration as if it is being drawn live. It is especially useful for portfolio demos, process studies, visual explanations, and experiments in computer vision and human-like motion.

## City demo — 25 seconds

![Inline 25-second city drawing preview](output/city-preview.gif)

The preview plays directly in the README. The full-resolution H.264 version is [`output/city.mp4`](output/city.mp4).

Input: [`city.png`](city.png)

## Documentation at a glance

| Part | What it does | Main file |
| --- | --- | --- |
| Input | Loads a raster drawing and extracts dark ink | `drawing/preprocess.py` |
| Skeleton | Reduces ink to drawable centerlines while preserving shading guides | `drawing/preprocess.py` |
| Tracing | Converts the skeleton graph into individual paths | `drawing/trace.py` |
| Ordering | Assigns structural, medium, and detail stages | `drawing/order.py` |
| Rendering | Reveals source pixels along the pencil trajectories | `drawing/render.py` |
| CLI | Runs the complete pipeline and writes the MP4 | `draw.py` |

## Installation and running

This is currently a source-based Python project, not a published pip package. Install it with a virtual environment:

```powershell
git clone https://github.com/YarnGrowler/Retrace.git
cd Retrace
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python draw.py city.png --duration 25 --fps 30 --no-preview
```

The output is written to `output/city.mp4`. Use `--duration auto` when every small stroke needs visible time under the pencil. Use `--debug` to write binary, skeleton, stroke-order, and reconstruction-difference images.

## Q&A

**Is Retrace a pip package?**  
Not yet. The virtual-environment workflow above is the right installation method while the command-line interface and shading model are still changing.

**Would a pip package be better later?**  
Yes. Once the pipeline is stable, a package with a `retrace` command would make installation cleaner and allow other programs to call the renderer as a library. Packaging should come after the input/output API is settled; it will not by itself solve the computer-vision problems.

**Why does the city result work better than heavily shaded illustrations?**  
The city is mostly clean line art. In a heavily shaded drawing, outlines, shadows, cross-hatching, and filled dark regions all become similar binary foreground during preprocessing. The tracer can then create parallel guides, and the ordering pass can leave blank patches until a later pass, producing horizontal eraser-like streaks.

**What is the next technical fix?**  
Separate contours, hatching, and filled-tone regions before skeletonization; divide dense tone into local patches; finish each patch before moving away; and use local pixel ownership plus a coverage check. That is more important than adding 3D at this stage.

**What does the renderer preserve?**  
The final appearance comes from the original grayscale image. The extracted trajectories control when pixels appear, so the result can use thousands of small paths without flattening the source into a few thick vector strokes.
