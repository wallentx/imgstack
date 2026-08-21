"""OpenCV-native detectors used by imgstack subject locking.

The model adapters are based on the Apache-2.0/MIT OpenCV Zoo examples, but
keep the public surface deliberately small: every detector returns boxes in
the original image coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import sys
from typing import Any, List, Optional, Sequence

import cv2
import numpy as np


COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

MODEL_FILENAMES = {
    "yunet": "face_detection_yunet_2023mar.onnx",
    "nanodet": "object_detection_nanodet_2022nov.onnx",
    "yolox": "object_detection_yolox_2022nov.onnx",
    "person": "person_detection_mediapipe_2023mar.onnx",
    "pose": "pose_estimation_mediapipe_2023mar.onnx",
    "vittrack": "object_tracking_vittrack_2023sep.onnx",
}


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    keypoints: Optional[np.ndarray] = None
    raw: Any = field(default=None, repr=False)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def model_path(detector_name: str, explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise RuntimeError(f"Model file does not exist: {explicit_path}")
        return explicit_path

    filename = MODEL_FILENAMES[detector_name]
    candidates: List[str] = []
    model_dir = os.environ.get("IMGSTACK_MODEL_DIR")
    if model_dir:
        candidates.append(os.path.join(model_dir, filename))
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidates.append(os.path.join(bundle_dir, "models", filename))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", filename))
    candidates.append(os.path.join(sys.prefix, "share", "imgstack", "models", filename))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError(
        f"Missing {detector_name} model {filename}. Set IMGSTACK_MODEL_DIR or use --subject-model."
    )


def _clip_detection(detection: Detection, image: np.ndarray) -> Detection:
    height, width = image.shape[:2]
    detection.x1 = float(np.clip(detection.x1, 0, width))
    detection.y1 = float(np.clip(detection.y1, 0, height))
    detection.x2 = float(np.clip(detection.x2, 0, width))
    detection.y2 = float(np.clip(detection.y2, 0, height))
    return detection


class YuNetDetector:
    def __init__(self, path: str, confidence: float = 0.8):
        self.detector = cv2.FaceDetectorYN.create(path, "", (320, 320), confidence, 0.3, 5000)

    def detect(self, image: np.ndarray, class_name: Optional[str] = None) -> List[Detection]:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _status, faces = self.detector.detect(image)
        if faces is None:
            return []
        detections = []
        for face in faces:
            x, y, w, h = map(float, face[:4])
            points = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
            detections.append(_clip_detection(
                Detection(x, y, x + w, y + h, float(face[14]), "face", points), image
            ))
        return detections


def _letterbox_center(image: np.ndarray, size: int, value: float = 0.0) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas = np.full((size, size, 3), value, dtype=resized.dtype)
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas, scale, left, top


class NanoDetDetector:
    def __init__(self, path: str, confidence: float = 0.35):
        self.strides = (8, 16, 32, 64)
        self.size = 416
        self.reg_max = 7
        self.confidence = confidence
        self.net = cv2.dnn.readNet(path)
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.anchors = []
        for stride in self.strides:
            count = self.size // stride
            x_values, y_values = np.meshgrid(np.arange(count), np.arange(count))
            centers = np.column_stack((
                x_values.ravel() * stride + 0.5 * (stride - 1),
                y_values.ravel() * stride + 0.5 * (stride - 1),
            ))
            self.anchors.append(centers)

    def _infer(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = cv2.dnn.blobFromImage((rgb - self.mean) / self.std)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        class_scores, box_outputs = outputs[::2], outputs[1::2]
        all_boxes, all_scores = [], []
        for stride, scores, box_data, anchors in zip(self.strides, class_scores, box_outputs, self.anchors):
            scores = np.squeeze(scores, axis=0) if scores.ndim == 3 else scores
            box_data = np.squeeze(box_data, axis=0) if box_data.ndim == 3 else box_data
            exp_data = np.exp(box_data.reshape(-1, self.reg_max + 1))
            distances = exp_data / np.sum(exp_data, axis=1, keepdims=True)
            distances = np.dot(distances, self.project).reshape(-1, 4) * stride
            if scores.shape[0] > 1000:
                keep = scores.max(axis=1).argsort()[::-1][:1000]
                scores, distances, anchors = scores[keep], distances[keep], anchors[keep]
            boxes = np.column_stack((
                anchors[:, 0] - distances[:, 0], anchors[:, 1] - distances[:, 1],
                anchors[:, 0] + distances[:, 2], anchors[:, 1] + distances[:, 3],
            ))
            all_boxes.append(np.clip(boxes, 0, self.size))
            all_scores.append(scores)
        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        xywh = boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(), confidences.tolist(), self.confidence, 0.6
        )
        if len(indices) == 0:
            return np.empty((0, 6), dtype=np.float32)
        indices = np.asarray(indices).reshape(-1)
        return np.column_stack((boxes[indices], confidences[indices], class_ids[indices]))

    def detect(self, image: np.ndarray, class_name: Optional[str] = None) -> List[Detection]:
        boxed, scale, left, top = _letterbox_center(image, self.size)
        predictions = self._infer(boxed)
        detections = []
        for prediction in predictions:
            class_id = int(prediction[5])
            label = COCO_CLASSES[class_id]
            if class_name and label != class_name:
                continue
            x1, y1, x2, y2 = prediction[:4]
            detection = Detection(
                (x1 - left) / scale, (y1 - top) / scale,
                (x2 - left) / scale, (y2 - top) / scale,
                float(prediction[4]), label,
            )
            detection = _clip_detection(detection, image)
            if detection.area > 0:
                detections.append(detection)
        return detections


class YoloXDetector:
    def __init__(self, path: str, confidence: float = 0.5):
        self.size = 640
        self.confidence = confidence
        self.net = cv2.dnn.readNet(path)
        grids, expanded = [], []
        for stride in (8, 16, 32):
            count = self.size // stride
            x_values, y_values = np.meshgrid(np.arange(count), np.arange(count))
            grid = np.stack((x_values, y_values), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            expanded.append(np.full((*grid.shape[:2], 1), stride))
        self.grids = np.concatenate(grids, axis=1)
        self.expanded = np.concatenate(expanded, axis=1)

    def detect(self, image: np.ndarray, class_name: Optional[str] = None) -> List[Detection]:
        height, width = image.shape[:2]
        scale = min(self.size / width, self.size / height)
        new_width, new_height = int(width * scale), int(height * scale)
        resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (new_width, new_height))
        padded = np.full((self.size, self.size, 3), 114.0, dtype=np.float32)
        padded[:new_height, :new_width] = resized.astype(np.float32)
        blob = np.transpose(padded, (2, 0, 1))[None, ...]
        self.net.setInput(blob)
        output = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0].copy()
        output[:, :2] = (output[:, :2] + self.grids[0]) * self.expanded[0]
        output[:, 2:4] = np.exp(output[:, 2:4]) * self.expanded[0]
        scores = output[:, 4:5] * output[:, 5:]
        confidences = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        boxes = np.column_stack((
            output[:, 0] - output[:, 2] / 2,
            output[:, 1] - output[:, 3] / 2,
            output[:, 2], output[:, 3],
        ))
        try:
            indices = cv2.dnn.NMSBoxesBatched(
                boxes.tolist(), confidences.tolist(), class_ids.tolist(), self.confidence, 0.5
            )
        except AttributeError:
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.confidence, 0.5)
        if len(indices) == 0:
            return []
        detections = []
        for index in np.asarray(indices).reshape(-1):
            class_id = int(class_ids[index])
            label = COCO_CLASSES[class_id]
            if class_name and label != class_name:
                continue
            x, y, w, h = boxes[index] / scale
            detection = _clip_detection(
                Detection(x, y, x + w, y + h, float(confidences[index]), label), image
            )
            if detection.area > 0:
                detections.append(detection)
        return detections


def _person_anchors() -> np.ndarray:
    anchors = []
    for grid_size, anchors_per_cell in ((28, 2), (14, 2), (7, 6)):
        for y in range(grid_size):
            for x in range(grid_size):
                center = ((x + 0.5) / grid_size, (y + 0.5) / grid_size)
                anchors.extend([center] * anchors_per_cell)
    return np.asarray(anchors, dtype=np.float32)


class MPPersonDetector:
    def __init__(self, path: str, confidence: float = 0.5):
        self.size = 224
        self.confidence = confidence
        self.net = cv2.dnn.readNet(path)
        self.anchors = _person_anchors()

    def _raw_detections(self, image: np.ndarray) -> List[np.ndarray]:
        height, width = image.shape[:2]
        ratio = min(self.size / height, self.size / width)
        new_height, new_width = int(height * ratio), int(width * ratio)
        resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (new_width, new_height))
        top, left = (self.size - new_height) // 2, (self.size - new_width) // 2
        canvas = cv2.copyMakeBorder(
            resized, top, self.size - new_height - top,
            left, self.size - new_width - left, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        blob = ((canvas.astype(np.float32) / 255.0) - 0.5) * 2.0
        self.net.setInput(np.transpose(blob, (2, 0, 1))[None, ...])
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        scores = np.clip(outputs[1][0, :, 0].astype(np.float64), -100, 100)
        scores = 1.0 / (1.0 + np.exp(-scores))
        deltas = outputs[0][0]
        center_delta = deltas[:, :2] / self.size
        size_delta = deltas[:, 2:4] / self.size
        scale = max(width, height)
        xy1 = (center_delta - size_delta / 2 + self.anchors) * scale
        xy2 = (center_delta + size_delta / 2 + self.anchors) * scale
        pad_bias = np.asarray([left / ratio, top / ratio], dtype=np.float32)
        boxes = np.column_stack((xy1, xy2)) - np.tile(pad_bias, 2)
        xywh = boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
        indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.confidence, 0.3, top_k=5000)
        if len(indices) == 0:
            return []
        rows = []
        for index in np.asarray(indices).reshape(-1):
            keypoints = deltas[index, 4:].reshape(4, 2) / self.size + self.anchors[index]
            keypoints = keypoints * scale - pad_bias
            rows.append(np.r_[boxes[index], keypoints.ravel(), scores[index]])
        return rows

    def detect(self, image: np.ndarray, class_name: Optional[str] = None) -> List[Detection]:
        detections = []
        for row in self._raw_detections(image):
            keypoints = row[4:12].reshape(4, 2)
            center, full_body = keypoints[0], keypoints[1]
            radius = float(np.linalg.norm(center - full_body))
            if radius <= 1:
                continue
            detection = Detection(
                center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius,
                float(row[12]), "person", keypoints, raw=row,
            )
            detections.append(_clip_detection(detection, image))
        return detections


class MPPoseDetector:
    """MediaPipe person detector refined with the MediaPipe pose model."""

    def __init__(self, person_path: str, pose_path: str, confidence: float = 0.5):
        self.person = MPPersonDetector(person_path, confidence)
        self.pose = cv2.dnn.readNet(pose_path)
        self.confidence = confidence
        self.size = 256

    def _pose_detection(self, image: np.ndarray, person: Detection) -> Optional[Detection]:
        points = np.asarray(person.raw[4:12], dtype=np.float32).reshape(4, 2)
        center, full_body = points[0], points[1]
        radius = float(np.linalg.norm(center - full_body))
        if radius <= 1:
            return None
        x1, y1 = np.floor(center - radius).astype(int)
        x2, y2 = np.ceil(center + radius).astype(int)
        crop_x1, crop_y1 = max(0, x1), max(0, y1)
        crop_x2, crop_y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop = cv2.copyMakeBorder(
            crop, crop_y1 - y1, y2 - crop_y2, crop_x1 - x1, x2 - crop_x2,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        local_center = center - np.asarray([x1, y1])
        local_full = full_body - np.asarray([x1, y1])
        angle = math.degrees(math.pi / 2 - math.atan2(
            -(local_full[1] - local_center[1]), local_full[0] - local_center[0]
        ))
        rotation = cv2.getRotationMatrix2D(tuple(local_center), angle, 1.0)
        rotated = cv2.warpAffine(crop, rotation, (crop.shape[1], crop.shape[0]))
        input_image = cv2.resize(rotated, (self.size, self.size)).astype(np.float32)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) / 255.0
        self.pose.setInput(input_image[None, ...])
        outputs = self.pose.forward(self.pose.getUnconnectedOutLayersNames())
        confidence = float(outputs[1][0][0])
        if confidence < self.confidence:
            return None
        landmarks = outputs[0][0].reshape(-1, 5)
        landmarks[:, 3:] = 1.0 / (1.0 + np.exp(-landmarks[:, 3:]))
        scale = np.asarray([crop.shape[1], crop.shape[0]], dtype=np.float32) / self.size
        local = (landmarks[:, :2] - self.size / 2.0) * scale
        point_rotation = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
        local = np.dot(local, point_rotation)
        inverse = cv2.invertAffineTransform(rotation)
        rotated_center = cv2.transform(local_center.reshape(1, 1, 2), rotation)[0, 0]
        image_points = local + rotated_center
        image_points = cv2.transform(image_points.reshape(1, -1, 2), inverse)[0]
        image_points += np.asarray([x1, y1])
        visible = (landmarks[:, 3] >= 0.5) & (landmarks[:, 4] >= 0.5)
        usable = image_points[:33][visible[:33]]
        if len(usable) < 4:
            usable = image_points[:33]
        low, high = usable.min(axis=0), usable.max(axis=0)
        center_box = (low + high) / 2.0
        half = (high - low) * 0.625
        return _clip_detection(Detection(
            center_box[0] - half[0], center_box[1] - half[1],
            center_box[0] + half[0], center_box[1] + half[1],
            confidence, "person", image_points[:33], raw=person.raw,
        ), image)

    def detect(self, image: np.ndarray, class_name: Optional[str] = None) -> List[Detection]:
        results = []
        for person in self.person.detect(image):
            pose = self._pose_detection(image, person)
            # A turned or heavily occluded body can fail pose refinement even
            # when the person detector is confident. Keep the detector box so
            # one difficult frame does not abort the whole image stack.
            results.append(pose if pose is not None else person)
        return results


class VitTracker:
    def __init__(self, path: str):
        if not hasattr(cv2, "TrackerVit_create"):
            raise RuntimeError("This OpenCV build does not provide TrackerVit")
        params = cv2.TrackerVit_Params()
        params.net = path
        params.backend = cv2.dnn.DNN_BACKEND_OPENCV
        params.target = cv2.dnn.DNN_TARGET_CPU
        self.tracker = cv2.TrackerVit_create(params)

    def initialize(self, image: np.ndarray, detection: Detection) -> None:
        box = (
            int(round(detection.x1)), int(round(detection.y1)),
            max(1, int(round(detection.width))), max(1, int(round(detection.height))),
        )
        self.tracker.init(image, box)

    def update(self, image: np.ndarray, label: str) -> Optional[Detection]:
        found, box = self.tracker.update(image)
        if not found:
            return None
        x, y, width, height = map(float, box)
        score = float(self.tracker.getTrackingScore())
        return _clip_detection(Detection(x, y, x + width, y + height, score, label), image)


def create_detector(
    name: str,
    confidence: Optional[float] = None,
    explicit_model: Optional[str] = None,
):
    if name == "yunet":
        return YuNetDetector(model_path("yunet", explicit_model), confidence or 0.8)
    if name == "nanodet":
        return NanoDetDetector(model_path("nanodet", explicit_model), confidence or 0.35)
    if name == "yolox":
        return YoloXDetector(model_path("yolox", explicit_model), confidence or 0.5)
    if name == "person":
        return MPPersonDetector(model_path("person", explicit_model), confidence or 0.5)
    if name == "pose":
        if explicit_model:
            raise RuntimeError("--subject-model is ambiguous for pose; use IMGSTACK_MODEL_DIR")
        return MPPoseDetector(model_path("person"), model_path("pose"), confidence or 0.5)
    raise RuntimeError(f"Unknown subject detector: {name}")


def union_detection(detections: Sequence[Detection], label: str = "subject") -> Detection:
    if not detections:
        raise ValueError("Cannot combine an empty detection list")
    return Detection(
        min(item.x1 for item in detections), min(item.y1 for item in detections),
        max(item.x2 for item in detections), max(item.y2 for item in detections),
        min(item.score for item in detections), label,
    )
