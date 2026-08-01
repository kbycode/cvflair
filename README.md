# cvflair

[![PyPI](https://img.shields.io/pypi/v/cvflair)](https://pypi.org/project/cvflair/)
[![CI](https://github.com/kbycode/cvflair/actions/workflows/ci.yml/badge.svg)](https://github.com/kbycode/cvflair/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/cvflair)](https://pypi.org/project/cvflair/)
[![Downloads](https://img.shields.io/pypi/dm/cvflair)](https://pypi.org/project/cvflair/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/kbycode/cvflair/blob/main/LICENSE)

*Türkçe sürüm: [README.tr.md](https://github.com/kbycode/cvflair/blob/main/README.tr.md).
Ayrıntılı belgeler Türkçe.*

A thin layer that draws computer vision detections on screen in three lines,
with themes that already look finished.

The camera loop, the themes and the drawing are one package with **no
dependencies beyond numpy and opencv**. It is model-agnostic: anything that
produces boxes -- YOLO, MediaPipe, InsightFace, your own model -- is drawn by the
same theme.

![cvflair demo](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/demo.gif)

*The same detections, four themes. Produced by `tools/make_demo_gif.py`: boxes
drawn over `docs/city.png`. Use `--background <path>` for another image.*

**Nine box styles, five ready themes, hand and pose skeletons -- all from one
`Theme(...)` line.** Try them without installing anything:
**[theme playground →](https://kbycode.github.io/cvflair/)**
Boxes and skeletons have separate preview modes; change the settings, copy the
generated Python. The page runs entirely in the browser.

![box styles](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/box-styles.png)

There is also drawing that sits on top of the box: a lock-on pulse and the trail
a tracked object leaves behind.

![pulse and trace](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/motion.png)

## Install

Python 3.10 or newer.

```bash
pip install cvflair
```

For YOLO, the Ultralytics extra (licence note at the bottom):

```bash
pip install "cvflair[yolo]"
```

## Quick start

No code needed -- the package ships a command:

```bash
cvflair 0 --theme neon --model yolov8n.pt
```

The same thing from Python:

```python
from cvflair import Camera

cam = Camera(source=0, theme="neon")
for frame in cam.stream():
    cam.show(frame)
```

The camera opens, frames are read on their own thread, and the window closes on
`q` or ESC -- no `release()` call, no `while True`.

That loop shows bare frames: a theme draws only when there is something to draw.
Attach a model and every step becomes a `(frame, detections)` pair with the theme
applied for you:

```python
cam = Camera(source=0, theme="hud")
for frame, detections in cam.stream(model="yolov8n.pt"):
    cam.show(frame, detections)
```

Instead of `model` you can pass your own function returning `Detections` -- that
is what keeps the library model-agnostic.

**Hand and pose skeletons** are drawn beside the boxes; the points come from your
model as well:

```python
from cvflair import HAND_21, KeyPoints

cam.show(frame, keypoints=KeyPoints(xy=hand_points), skeleton=HAND_21)
```

In Jupyter or Colab, where `cv2.imshow` has no window to draw on:

```python
import cvflair

theme.annotate(frame, detections)
cvflair.notebook.show(frame)
```

## Documentation

The detailed docs are in Turkish; this page and the playground are bilingual.

| | |
|---|---|
| [Themes and box styles](https://github.com/kbycode/cvflair/blob/main/docs/temalar.md) | Five themes, nine styles, pulse and trace, accent colour, palettes, stats panel, writing your own theme |
| [Command line and video writing](https://github.com/kbycode/cvflair/blob/main/docs/komut-satiri.md) | The `cvflair` command, sources and options, `VideoWriter` |
| [Key points and skeletons](https://github.com/kbycode/cvflair/blob/main/docs/noktalar.md) | Hand and pose skeletons, `KeyPoints`, shipped topologies, MediaPipe, your own layout |
| [Models and detections](https://github.com/kbycode/cvflair/blob/main/docs/modeller.md) | `stream(model=...)`, your own detector, `Detections`, Ultralytics settings, video files |
| [API summary and internals](https://github.com/kbycode/cvflair/blob/main/docs/api.md) | The whole public surface, thread and queue behaviour, measured performance |
| [Example gallery](https://github.com/kbycode/cvflair/blob/main/examples/README.md) | Ten working examples; which need a camera and which do not |
| [Contributing](https://github.com/kbycode/cvflair/blob/main/CONTRIBUTING.md) | Setup, scope boundaries, how to add a theme |

Three examples run without a camera:

```bash
python examples/motion_detection.py       # real detection, no neural network (needs a camera)
python examples/theme_preview.py          # no camera; writes a PNG per theme
python examples/video_file.py input.mp4   # annotates a file into a copy
```

## Why it is built this way

- **The queue holds one frame.** A new frame replaces the waiting one, so lag
  does not pile up when processing slows down and the screen always shows the
  newest frame. `drop_frames=False` reverses this for video files, where every
  frame counts.
- **Drawing objects are built once** and reused on every frame.
- **The dependency surface is deliberately narrow.** `import cvflair` takes about
  0.3 s; the install is around 170 MB, nearly all of it opencv and numpy.
- **Models stay outside the package.** No weights and no model code are bundled.

Measured numbers and the reasoning behind them:
[API and internals](https://github.com/kbycode/cvflair/blob/main/docs/api.md#ne-kadar-sürüyor).

## Development

```bash
git clone https://github.com/kbycode/cvflair.git
cd cvflair
pip install -e ".[dev]"

pytest              # no camera required
ruff check .
mypy                # the package ships py.typed, so the claim is checked
```

Details and the contribution flow:
[CONTRIBUTING.md](https://github.com/kbycode/cvflair/blob/main/CONTRIBUTING.md)

## Licence

MIT -- see [LICENSE](https://github.com/kbycode/cvflair/blob/main/LICENSE). Both
dependencies are permissively licensed (`opencv-python` Apache 2.0, `numpy` BSD).

No YOLO weights and no Ultralytics code are bundled here. If you use
Ultralytics, meeting its AGPL-3.0 terms is the responsibility of the project
that uses it.
