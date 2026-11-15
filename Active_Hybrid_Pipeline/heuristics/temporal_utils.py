import collections
from typing import Any, Dict, List


def _as_frame_list(value: Any) -> List[Any]:
    if isinstance(value, dict):
        return value.get("per_frame", [])
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    return []


def smooth_frame_sequence(labels: List[Any], window: int = 3) -> List[Any]:
    """Smooth a sequence of labels using a rolling majority vote that preserves a stable label on ties."""
    if not labels:
        return []
    if len(labels) == 1:
        return labels[:]

    if window < 1:
        window = 1

    smoothed: List[Any] = []
    prev_label = labels[0]
    for i in range(len(labels)):
        start = max(0, i - window + 1)
        window_labels = labels[start:i + 1]
        if not window_labels:
            smoothed.append(labels[i])
            continue

        counts = collections.Counter(str(x) for x in window_labels if x is not None)
        if not counts:
            smoothed.append(labels[i])
            continue

        most_common = counts.most_common(2)
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            smoothed.append(prev_label)
        else:
            chosen = most_common[0][0]
            smoothed.append(chosen)
            prev_label = chosen

    return smoothed


def summarize_label_sequence(labels: List[Any], default: str = "none", window: int = 3) -> str:
    """Summarize a sequence of labels by choosing the dominant label after smoothing."""
    if not labels:
        return default
    smoothed = smooth_frame_sequence(labels, window=window)
    if not smoothed:
        return default
    counts = collections.Counter(str(x) for x in smoothed if x is not None)
    if not counts:
        return default
    return counts.most_common(1)[0][0]


def build_structured_sign_descriptor(modules: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact, structured description from module outputs for downstream avatar generation."""
    descriptor: Dict[str, Any] = {"summary": {}, "phases": []}

    def pick_summary(key: str, default: str = "none") -> str:
        values = _as_frame_list(modules.get(key, []))
        if not values:
            return default
        return summarize_label_sequence(values, default=default, window=3)

    descriptor["summary"]["handshape"] = pick_summary("handshape", "hamflathand")
    descriptor["summary"]["orientation"] = pick_summary("orientation", "hampalmd")
    descriptor["summary"]["location"] = pick_summary("location", "hamchest")
    descriptor["summary"]["movement"] = pick_summary("movement1", "hamnomotion")
    descriptor["summary"]["contact"] = pick_summary("contact", "none")

    descriptor["phases"] = [
        {
            "name": "start",
            "handshape": pick_summary("handshape", "hamflathand"),
            "orientation": pick_summary("orientation", "hampalmd"),
            "location": pick_summary("location", "hamchest"),
            "movement": pick_summary("movement1", "hamnomotion"),
            "contact": pick_summary("contact", "none"),
        }
    ]
    return descriptor
