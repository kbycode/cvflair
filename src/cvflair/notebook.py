"""
Showing frames inside Jupyter and Colab.

``cv2.imshow`` needs a desktop window server and there is none in a notebook --
the call either does nothing or takes the kernel down with it. The usual
workaround is a matplotlib figure plus a channel swap, which is easy to get
wrong: OpenCV keeps pixels as BGR and everything else expects RGB, so the
picture comes back with red and blue traded.

This module encodes the frame and hands it to the notebook's own display, so
neither matplotlib nor a manual swap is needed.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

__all__ = ["show", "to_png", "in_notebook"]


def in_notebook() -> bool:
    """True when running inside an IPython kernel with rich display."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    # Terminal IPython has a shell but no way to draw an image.
    return shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"


def to_png(frame: np.ndarray, *, bgr: bool = True) -> bytes:
    """
    Encode a frame as PNG bytes.

    ``bgr`` says how the array is laid out; OpenCV frames are BGR, arrays coming
    from PIL or matplotlib are RGB.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Boş kare gösterilemez.")

    image = np.asarray(frame)
    if image.dtype != np.uint8:
        # Kayan noktalı kareler 0-1 ya da 0-255 olabiliyor; ölçek buradan anlaşılıyor.
        scale = 255.0 if float(image.max(initial=0.0)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)

    if image.ndim == 3 and image.shape[2] == 3 and not bgr:
        image = image[:, :, ::-1]

    ok, buffer = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover - encoder failure needs a broken build
        raise ValueError("Kare PNG olarak kodlanamadı.")
    return bytes(buffer)


def show(frame: np.ndarray, *, bgr: bool = True, width: int | None = None) -> Any:
    """
    Display ``frame`` inline in a notebook.

    Returns the image object, so it also works as the last expression in a
    cell::

        detections = detector(frame)
        theme.annotate(frame, detections)
        cvflair.notebook.show(frame)

    Outside a notebook there is nothing to draw on; the object is returned
    unshown rather than raising, so the same code runs in both places.
    """
    try:
        from IPython.display import Image, display
    except ImportError as error:
        raise ImportError(
            "Not defteri gösterimi IPython gerektiriyor. Jupyter ya da Colab "
            "dışında kareyi dosyaya yazmak için cv2.imwrite kullanılabilir."
        ) from error

    image = Image(data=to_png(frame, bgr=bgr), format="png", width=width)
    if in_notebook():
        display(image)
    return image
