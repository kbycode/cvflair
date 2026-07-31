"""
Key points and the skeletons that connect them.

Boxes are one family, joints are the other. cvflair does not estimate poses --
the points come from whatever model the caller runs (MediaPipe, YOLO-pose,
OpenPose); what lives here is the container and the wiring between indices.

A skeleton is plain data: a tuple of index pairs. The shipped ones follow the
ordering of the models named beside them, and a caller with a different layout
passes its own list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "KeyPoints",
    "Skeleton",
    "HAND_21",
    "POSE_17",
    "SKELETONS",
    "is_keypoints",
    "resolve_skeleton",
]

#: A skeleton is the list of point pairs to connect.
Skeleton = tuple[tuple[int, int], ...]

#: MediaPipe Hands ordering: wrist 0, thumb 1-4, index 5-8, middle 9-12,
#: ring 13-16, pinky 17-20.
HAND_21: Skeleton = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
)

#: COCO ordering, which YOLO-pose also uses: 0 nose, 1-2 eyes, 3-4 ears,
#: 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees,
#: 15-16 ankles.
POSE_17: Skeleton = (
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # shoulders and arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
)

#: Lookup by name, for ``cam.show(..., skeleton="hand")``.
SKELETONS: dict[str, Skeleton] = {"hand": HAND_21, "pose": POSE_17}


@dataclass
class KeyPoints:
    """
    Point coordinates for one or more skeletons.

    ``xy`` is shaped ``(N, K, 2)``: N skeletons, K points each, in pixels. A
    single skeleton may be handed over as ``(K, 2)`` and is reshaped.
    """

    xy: np.ndarray
    #: Per-point confidence; points below the theme's threshold are not drawn.
    confidence: np.ndarray | None = None
    #: Per-skeleton class; colour lookup reads this.
    class_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=np.float32)
        if self.xy.ndim == 2:  # a single skeleton was handed over
            self.xy = self.xy[None, ...]
        if self.xy.ndim != 3 or self.xy.shape[-1] != 2:
            raise ValueError(f"xy must be shaped (N, K, 2) or (K, 2), got {self.xy.shape}.")

        count, points = self.xy.shape[0], self.xy.shape[1]
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(count, -1)
            if self.confidence.shape[1] != points:
                raise ValueError(
                    f"confidence has {self.confidence.shape[1]} points but xy has {points}."
                )
        if self.class_id is not None:
            self.class_id = np.asarray(self.class_id, dtype=int).reshape(-1)
            if len(self.class_id) != count:
                raise ValueError(
                    f"class_id has {len(self.class_id)} entries but xy has {count} skeletons."
                )

    def __len__(self) -> int:
        return len(self.xy)

    @property
    def point_count(self) -> int:
        """Points per skeleton."""
        return int(self.xy.shape[1]) if len(self.xy) else 0

    @classmethod
    def empty(cls, point_count: int = 0) -> KeyPoints:
        return cls(xy=np.empty((0, point_count, 2), dtype=np.float32))

    @classmethod
    def from_normalized(cls, xy: Any, width: int, height: int, **fields: Any) -> KeyPoints:
        """
        Scale 0-1 coordinates into pixels.

        Libraries like MediaPipe report points relative to the frame; the
        scaling happens here so no model knowledge enters the package.
        """
        scaled = np.asarray(xy, dtype=np.float32) * np.array([width, height], dtype=np.float32)
        return cls(xy=scaled, **fields)

    def __repr__(self) -> str:
        return f"KeyPoints({len(self)} skeletons, {self.point_count} points)"


def is_keypoints(value: Any) -> bool:
    """True when ``value`` carries points that can be drawn."""
    return hasattr(value, "xy") and hasattr(value, "__len__")


def resolve_skeleton(skeleton: Any) -> Skeleton:
    """Turn a name or a pair list into a skeleton."""
    if isinstance(skeleton, str):
        try:
            return SKELETONS[skeleton.strip().lower()]
        except KeyError:
            raise ValueError(
                f"Unknown skeleton {skeleton!r}. Available: {', '.join(sorted(SKELETONS))}."
            ) from None
    return tuple((int(first), int(second)) for first, second in skeleton)
