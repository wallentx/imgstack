"""Subject-centric scale and position stabilization for image stacks."""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from subject_detectors import Detection, VitTracker, create_detector, model_path, union_detection


def _select_detections(
    detections: Sequence[Detection], selection: str, count: int
) -> List[Detection]:
    if not detections:
        return []
    if selection == "largest":
        selected = [max(detections, key=lambda item: item.area)]
    elif selection == "leftmost":
        selected = [min(detections, key=lambda item: item.center[0])]
    elif selection == "rightmost":
        selected = [max(detections, key=lambda item: item.center[0])]
    else:
        selected = sorted(detections, key=lambda item: item.score, reverse=True)
        if count > 0:
            selected = selected[:count]
    return selected


def _subject_size(detection: Detection, metric: str) -> float:
    if metric == "width":
        return detection.width
    if metric == "height":
        return detection.height
    if metric == "area":
        return math.sqrt(detection.area)
    return math.hypot(detection.width, detection.height)


def _draw_debug(
    image: np.ndarray,
    detections: Sequence[Detection],
    selected: Sequence[Detection],
    group: Detection,
    output_path: str,
) -> None:
    preview = image.copy()
    selected_ids = {id(item) for item in selected}
    line = max(2, round(max(image.shape[:2]) / 700))
    for detection in detections:
        color = (0, 220, 0) if id(detection) in selected_ids else (120, 120, 120)
        p1 = (round(detection.x1), round(detection.y1))
        p2 = (round(detection.x2), round(detection.y2))
        cv2.rectangle(preview, p1, p2, color, line)
        cv2.putText(
            preview, f"{detection.label} {detection.score:.2f}",
            (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, max(1, line - 1), cv2.LINE_AA,
        )
        if detection.keypoints is not None:
            for point in np.asarray(detection.keypoints):
                cv2.circle(preview, tuple(np.rint(point[:2]).astype(int)), line + 1, (0, 0, 255), -1)
    cv2.rectangle(
        preview,
        (round(group.x1), round(group.y1)),
        (round(group.x2), round(group.y2)),
        (0, 220, 255), line + 1,
    )
    cv2.imwrite(output_path, preview)


def _track_group(
    images: Sequence[np.ndarray],
    reference_index: int,
    reference_group: Detection,
    fallback_groups: Sequence[Detection],
) -> List[Detection]:
    groups: List[Optional[Detection]] = [None] * len(images)
    groups[reference_index] = reference_group

    def run(indices: Sequence[int]) -> None:
        tracker = VitTracker(model_path("vittrack"))
        tracker.initialize(images[reference_index], reference_group)
        for index in indices:
            tracked = tracker.update(images[index], reference_group.label)
            groups[index] = tracked if tracked is not None and tracked.score >= 0.1 else fallback_groups[index]

    run(range(reference_index + 1, len(images)))
    run(range(reference_index - 1, -1, -1))
    return [group if group is not None else fallback_groups[index] for index, group in enumerate(groups)]


def align_subject_stack(
    images: Sequence[np.ndarray],
    names: Sequence[str],
    reference_index: int,
    detector_name: str,
    selection: str = "all",
    count: int = 0,
    class_name: str = "person",
    confidence: Optional[float] = None,
    explicit_model: Optional[str] = None,
    size_metric: str = "diagonal",
    tracker_name: str = "none",
    debug_dir: Optional[str] = None,
) -> Tuple[List[np.ndarray], List[Detection]]:
    detector = create_detector(detector_name, confidence, explicit_model)
    all_detections: List[List[Detection]] = []
    selected_groups: List[Detection] = []

    requested_class = class_name if detector_name in ("nanodet", "yolox") else None
    for index, image in enumerate(images):
        detections = detector.detect(image, requested_class)
        selected = _select_detections(detections, selection, count)
        if not selected:
            raise RuntimeError(
                f"No {requested_class or detector_name} subject detected in frame {index}: {names[index]}"
            )
        group = union_detection(selected, requested_class or detector_name)
        all_detections.append(detections)
        selected_groups.append(group)
        if debug_dir:
            _draw_debug(
                image, detections, selected, group,
                os.path.join(debug_dir, f"{index:02d}_subject_detections.jpg"),
            )

    if tracker_name == "vittrack":
        selected_groups = _track_group(
            images, reference_index, selected_groups[reference_index], selected_groups
        )

    reference = selected_groups[reference_index]
    reference_size = _subject_size(reference, size_metric)
    if reference_size <= 1:
        raise RuntimeError("Reference subject has an invalid size")
    reference_center = reference.center
    output_height, output_width = images[reference_index].shape[:2]

    warped_images: List[np.ndarray] = []
    warped_masks: List[np.ndarray] = []
    for image, group in zip(images, selected_groups):
        current_size = _subject_size(group, size_metric)
        if current_size <= 1:
            raise RuntimeError("Detected subject has an invalid size")
        scale = reference_size / current_size
        center_x, center_y = group.center
        matrix = np.asarray([
            [scale, 0.0, reference_center[0] - scale * center_x],
            [0.0, scale, reference_center[1] - scale * center_y],
        ], dtype=np.float32)
        warped_images.append(cv2.warpAffine(
            image, matrix, (output_width, output_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        ))
        valid = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped_masks.append(cv2.warpAffine(
            valid, matrix, (output_width, output_height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        ))

    common = warped_masks[0].copy()
    for mask in warped_masks[1:]:
        common = cv2.bitwise_and(common, mask)
    points = cv2.findNonZero(common)
    if points is None:
        raise RuntimeError("Subject transforms have no common output area")
    x, y, width, height = cv2.boundingRect(points)
    if width < 2 or height < 2:
        raise RuntimeError("Subject transforms produced an unusable common crop")
    return [image[y:y + height, x:x + width] for image in warped_images], selected_groups
