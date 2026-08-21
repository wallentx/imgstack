#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import shutil
import subprocess

from subject_detectors import COCO_CLASSES
from subject_lock import align_subject_stack

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(input_dir: str) -> List[str]:
    entries = []
    for name in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTS:
            entries.append(os.path.join(input_dir, name))
    return entries


def load_images(paths: List[str]) -> List[np.ndarray]:
    images = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        images.append(img)
    return images


def fit_images_to_common_canvas(
    images: List[np.ndarray], canvas_width: int, canvas_height: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Scale images down and center them on a shared canvas.

    The returned masks mark real image pixels. This lets later alignment and
    intersection cropping discard letterbox padding while keeping every frame
    exactly the same size.
    """
    fitted: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []

    for image in images:
        height, width = image.shape[:2]
        scale = min(canvas_height / height, canvas_width / width)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if new_width == width and new_height == height:
            resized = image
        else:
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas = np.zeros((canvas_height, canvas_width, image.shape[2]), dtype=image.dtype)
        canvas[y:y + new_height, x:x + new_width] = resized

        valid = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        valid[y:y + new_height, x:x + new_width] = 255
        fitted.append(canvas)
        valid_masks.append(valid)

    return fitted, valid_masks


def detect_face_mask(image_bgr: np.ndarray) -> np.ndarray:
    # Returns mask (uint8 0/255) where faces are 255
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    cascade_dirs = []
    try:
        cascade_dirs.append(cv2.data.haarcascades)
    except Exception:
        pass
    prefix = os.environ.get("PREFIX")
    if prefix:
        cascade_dirs.append(os.path.join(prefix, "share", "opencv4", "haarcascades"))

    cascade_dir = next((path for path in cascade_dirs if os.path.isdir(path)), None)
    if cascade_dir is None:
        # No cascade available; return empty mask
        return np.zeros(gray.shape, dtype=np.uint8)

    frontal = cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_frontalface_default.xml"))
    profile = cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_profileface.xml"))
    if frontal.empty() or profile.empty():
        return np.zeros(gray.shape, dtype=np.uint8)

    # Haar cascades are both faster and more reliable here on a moderate-size
    # preview. Scale detections back to the working image afterwards.
    detect_scale = min(1.0, 800.0 / max(gray.shape))
    if detect_scale < 1.0:
        detect_gray = cv2.resize(gray, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
    else:
        detect_gray = gray

    detect_args = dict(scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(18, 18))
    faces = list(frontal.detectMultiScale(detect_gray, **detect_args))
    faces.extend(profile.detectMultiScale(detect_gray, **detect_args))

    # The profile cascade detects only one orientation, so scan a mirror and
    # convert those boxes back to the original preview coordinates.
    flipped = cv2.flip(detect_gray, 1)
    flipped_faces = profile.detectMultiScale(flipped, **detect_args)
    detect_width = detect_gray.shape[1]
    faces.extend((detect_width - x - w, y, w, h) for x, y, w, h in flipped_faces)

    mask = np.zeros_like(gray, dtype=np.uint8)
    for (x, y, w, h) in faces:
        if detect_scale < 1.0:
            x, y, w, h = (int(round(value / detect_scale)) for value in (x, y, w, h))
        # Slightly dilate the face region to fully cover
        pad_w = int(0.15 * w)
        pad_h = int(0.20 * h)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(gray.shape[1], x + w + pad_w)
        y1 = min(gray.shape[0], y + h + pad_h)
        mask[y0:y1, x0:x1] = 255
    return mask


def build_feature_mask(image_bgr: np.ndarray, align_mode: str) -> Optional[np.ndarray]:
    h, w = image_bgr.shape[:2]
    if align_mode == "features":
        return None  # use all pixels
    if align_mode == "static":
        # Legacy path (kept for completeness). Actual static background handling
        # now uses motion-based masks computed across the stack in main().
        # Here, we fall back to face-masked static if motion masks are not used.
        face_mask = detect_face_mask(image_bgr)
        inv = cv2.bitwise_not(face_mask)
        if inv.max() == 0:
            return None
        return inv
    if align_mode == "face":
        # Use only face regions
        face_mask = detect_face_mask(image_bgr)
        if face_mask is None or face_mask.max() == 0:
            # No faces found; fallback to all
            return None
        return face_mask
    return None


def compute_transform(
    from_img: np.ndarray,
    to_img: np.ndarray,
    from_mask: Optional[np.ndarray],
    to_mask: Optional[np.ndarray],
    debug: bool = False,
    debug_prefix: Optional[str] = None,
    preferred_model: str = "auto",
) -> Tuple[str, np.ndarray]:
    # Returns (transform_type, matrix)
    # Detect features (prefer SIFT, fallback to ORB)
    from_gray = cv2.cvtColor(from_img, cv2.COLOR_BGR2GRAY)
    to_gray = cv2.cvtColor(to_img, cv2.COLOR_BGR2GRAY)

    # Improve contrast for low-contrast scenes
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    from_gray_eq = clahe.apply(from_gray)
    to_gray_eq = clahe.apply(to_gray)

    detector_name = "SIFT"
    try:
        sift = cv2.SIFT_create(nfeatures=4000)
        from_kp, from_desc = sift.detectAndCompute(from_gray_eq, from_mask)
        to_kp, to_desc = sift.detectAndCompute(to_gray_eq, to_mask)
        use_orb = False
    except Exception:
        detector_name = "ORB"
        orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8)
        from_kp, from_desc = orb.detectAndCompute(from_gray_eq, from_mask)
        to_kp, to_desc = orb.detectAndCompute(to_gray_eq, to_mask)
        use_orb = True

    if from_desc is None or to_desc is None or len(from_kp) < 8 or len(to_kp) < 8:
        # Fallback to ECC (enhanced correlation coefficient) affine if features insufficient
        return ecc_affine_fallback(from_gray, to_gray, from_mask)

    if not use_orb:
        # FLANN for SIFT
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = flann.knnMatch(from_desc, to_desc, k=2)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(from_desc, to_desc, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        ratio = 0.75 if not use_orb else 0.78
        if m.distance < ratio * n.distance:
            good.append(m)

    # Optionally save debug of matches
    if debug and debug_prefix and len(good) > 0:
        draw_n = min(80, len(good))
        good_sorted = sorted(good, key=lambda x: x.distance)[:draw_n]
        dbg_img = cv2.drawMatches(
            from_img,
            from_kp,
            to_img,
            to_kp,
            good_sorted,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        try:
            cv2.imwrite(f"{debug_prefix}_matches_{detector_name}.jpg", dbg_img)
        except Exception:
            pass

    if preferred_model in ("auto", "homography") and len(good) >= 6:
        src_pts = np.float32([from_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([to_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is not None:
            return ("homography", H)

    if len(good) >= 3:
        src_pts = np.float32([from_kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([to_kp[m.trainIdx].pt for m in good]).reshape(-1, 2)
        if preferred_model in ("auto", "affine"):
            A, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if A is not None:
                return ("affine", A)
        if preferred_model == "similarity":
            A, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if A is not None:
                return ("affine", A)

    # Final fallback
    return ecc_affine_fallback(from_gray, to_gray, from_mask)


def ecc_affine_fallback(from_gray: np.ndarray, to_gray: np.ndarray, from_mask: Optional[np.ndarray]) -> Tuple[str, np.ndarray]:
    # ECC requires float32
    from_f = from_gray.astype(np.float32) / 255.0
    to_f = to_gray.astype(np.float32) / 255.0
    # Start with identity affine (2x3)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        # ECC expects template (to_f) and input (from_f). Mask applies to input.
        _cc, warp_matrix = cv2.findTransformECC(
            to_f,
            from_f,
            warp_matrix,
            motionType=cv2.MOTION_AFFINE,
            inputMask=from_mask,
            criteria=criteria,
        )
        return ("affine", warp_matrix)
    except cv2.error:
        return ("identity", np.eye(3, dtype=np.float32))


def compute_static_region_masks(images_bgr: List[np.ndarray]) -> List[np.ndarray]:
    # Build motion-based masks: 255 where static, 0 where moving
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images_bgr]
    # Use median to suppress the moving subject
    stack = np.stack(grays, axis=0).astype(np.float32)
    median_img = np.median(stack, axis=0).astype(np.uint8)
    static_masks: List[np.ndarray] = []
    for g in grays:
        diff = cv2.absdiff(g, median_img)
        # Blur a bit to merge small differences
        diff_blur = cv2.GaussianBlur(diff, (7, 7), 0)
        # Otsu threshold
        _t, th = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Moving = th; Static = not moving
        moving = th
        # Dilate moving to be conservative
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        moving_dil = cv2.dilate(moving, kernel, iterations=1)
        static = cv2.bitwise_not(moving_dil)
        # Remove tiny isolated static islands
        static = cv2.erode(static, kernel, iterations=1)
        static_masks.append(static)
    return static_masks


def warp_with_transform(img: np.ndarray, transform_type: str, M: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    if transform_type == "homography":
        return cv2.warpPerspective(img, M, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if transform_type == "affine":
        return cv2.warpAffine(img, M, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # identity
    return img.copy()


def warp_mask(transform_type: str, M: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    if transform_type == "homography":
        warped = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    elif transform_type == "affine":
        warped = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    else:
        warped = mask
    return warped


def intersection_bounding_rect(masks: List[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    inter = masks[0].copy()
    for m in masks[1:]:
        inter = cv2.bitwise_and(inter, m)
    _, thresh = cv2.threshold(inter, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def save_gif_imageio(frames_bgr: List[np.ndarray], out_path: str, fps: float) -> None:
    if imageio is None:
        raise RuntimeError("imageio is not available")
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    duration = 1.0 / max(1e-6, fps)
    imageio.mimsave(out_path, frames_rgb, duration=duration, loop=0)


def save_gif_magick(frame_paths: List[str], out_path: str, fps: float) -> None:
    # Use ImageMagick: magick -delay <cs> -loop 0 frames out.gif
    delay_cs = max(1, int(round(100.0 / max(1e-6, fps))))
    magick_bin = shutil.which("magick") or shutil.which("convert")
    if magick_bin is None:
        raise RuntimeError("ImageMagick not found. Install it (e.g., 'pkg install imagemagick').")
    cmd = [magick_bin, "-delay", str(delay_cs), "-loop", "0"] + frame_paths + [out_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ImageMagick failed: {e.stderr.decode(errors='ignore')}")


def save_gif_auto(frames_bgr: List[np.ndarray], frame_paths: List[str], out_path: str, fps: float, tool: str) -> None:
    if tool == "imageio":
        return save_gif_imageio(frames_bgr, out_path, fps)
    if tool == "magick":
        return save_gif_magick(frame_paths, out_path, fps)
    # auto: prefer imageio, fallback to magick, then raise
    try:
        return save_gif_imageio(frames_bgr, out_path, fps)
    except Exception:
        return save_gif_magick(frame_paths, out_path, fps)


def save_ffmpeg_animation(
    frames_bgr: List[np.ndarray], out_path: str, fps: float, output_format: str
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for MP4, WebM, and WebP output")

    temp_root = os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(prefix="imgstack-frames-", dir=temp_root) as sequence_dir:
        for index, frame in enumerate(frames_bgr):
            frame_path = os.path.join(sequence_dir, f"frame-{index:06d}.png")
            if not cv2.imwrite(frame_path, frame):
                raise RuntimeError(f"Failed to prepare animation frame {index}")

        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(sequence_dir, "frame-%06d.png"),
        ]
        even_scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        if output_format == "mp4":
            command += [
                "-vf", even_scale, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
        elif output_format == "webm":
            command += [
                "-vf", even_scale, "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p",
                "-crf", "30", "-b:v", "0",
            ]
        elif output_format == "webp":
            command += [
                "-c:v", "libwebp_anim", "-lossless", "0", "-q:v", "80", "-loop", "0",
            ]
        else:
            raise RuntimeError(f"Unsupported animation format: {output_format}")
        command.append(out_path)
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            message = error.stderr.decode(errors="ignore").strip()
            raise RuntimeError(f"ffmpeg failed to write {output_format}: {message}") from error


def build_crossfade_frames(
    frames: List[np.ndarray], still_fps: float, crossfade_seconds: float, animation_fps: float
) -> Tuple[List[np.ndarray], float]:
    if crossfade_seconds <= 0:
        return frames, still_fps
    still_duration = 1.0 / max(still_fps, 1e-6)
    if crossfade_seconds >= still_duration:
        raise RuntimeError(
            f"Crossfade ({crossfade_seconds:.3g}s) must be shorter than frame duration "
            f"({still_duration:.3g}s at {still_fps:.3g} FPS)"
        )
    transition_count = max(1, int(round(crossfade_seconds * animation_fps)))
    hold_count = max(1, int(round((still_duration - crossfade_seconds) * animation_fps)))
    rendered: List[np.ndarray] = []
    for index, current in enumerate(frames):
        rendered.extend([current] * hold_count)
        # Wrap the final transition to frame zero so animated outputs loop
        # without a visible jump.
        following = frames[(index + 1) % len(frames)]
        for step in range(1, transition_count + 1):
            alpha = step / (transition_count + 1)
            rendered.append(cv2.addWeighted(current, 1.0 - alpha, following, alpha, 0.0))
    return rendered, animation_fps


def blend_preview(images: List[np.ndarray]) -> np.ndarray:
    acc = np.zeros_like(images[0], dtype=np.float32)
    for im in images:
        acc += im.astype(np.float32) / 255.0
    acc /= max(1, len(images))
    return np.clip(acc * 255.0, 0, 255).astype(np.uint8)


def write_outputs(args: argparse.Namespace, paths: List[str], frames: List[np.ndarray]) -> None:
    aligned_paths: List[str] = []
    for index, (source_path, image) in enumerate(zip(paths, frames)):
        base = os.path.splitext(os.path.basename(source_path))[0]
        output_path = os.path.join(args.output_dir, f"{index:02d}_{base}_aligned.png")
        if not cv2.imwrite(output_path, image):
            raise RuntimeError(f"Failed to write aligned frame: {output_path}")
        aligned_paths.append(output_path)

    if args.save_overlay:
        overlay = blend_preview(frames)
        cv2.imwrite(os.path.join(args.output_dir, "overlay_preview.png"), overlay)

    requested_outputs = []
    if args.gif:
        requested_outputs.append(("gif", args.gif_name))
    if args.mp4:
        requested_outputs.append(("mp4", args.mp4_name))
    if args.webm:
        requested_outputs.append(("webm", args.webm_name))
    if args.webp:
        requested_outputs.append(("webp", args.webp_name))

    if args.crossfade > 0 and not requested_outputs:
        raise RuntimeError("--crossfade applies only when an animated output is requested")
    animation_frames, output_fps = build_crossfade_frames(
        frames, args.fps, args.crossfade, args.animation_fps
    )

    errors = []
    for output_format, filename in requested_outputs:
        output_path = os.path.join(args.output_dir, filename)
        try:
            if output_format == "gif":
                if args.crossfade > 0 and args.gif_tool == "magick":
                    raise RuntimeError("crossfade GIF output requires --gif-tool imageio or auto")
                if args.crossfade > 0:
                    save_gif_imageio(animation_frames, output_path, output_fps)
                else:
                    save_gif_auto(frames, aligned_paths, output_path, fps=args.fps, tool=args.gif_tool)
            else:
                save_ffmpeg_animation(animation_frames, output_path, output_fps, output_format)
        except Exception as error:
            errors.append(f"{output_format}: {error}")

    if errors:
        raise RuntimeError("; ".join(errors))
    if args.delete_frames:
        if not requested_outputs:
            raise RuntimeError("--delete-frames requires at least one animation output")
        for frame_path in aligned_paths:
            os.unlink(frame_path)


def main():
    parser = argparse.ArgumentParser(
        description="Align or subject-lock an image stack and export animated image/video formats.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input", nargs="?", default=".", help="Input directory containing images (default: current directory)")
    parser.add_argument(
        "-a",
        "--align-mode",
        choices=["static", "face", "subject", "features"],
        default="static",
        help=(
            "Alignment focus.\n"
            "  static    motion-masked background (default)\n"
            "  face      YuNet multi-face subject lock (legacy alias)\n"
            "  subject   constant-size detector/tracker subject lock\n"
            "  features  use all features (no masking)"
        ),
    )
    parser.add_argument(
        "-r",
        "--reference-index",
        type=int,
        default=None,
        help=(
            "Reference frame index after sorting (0-based).\n"
            "  0..N-1    default: middle frame"
        ),
    )
    parser.add_argument("-o", "--output-dir", default="aligned_out", help="Directory to write outputs")
    parser.add_argument("-g", "--gif", action="store_true", help="Also write an animated GIF")
    parser.add_argument("-n", "--gif-name", default="stack.gif", help="Filename of the output GIF (within output-dir)")
    parser.add_argument("-f", "--fps", type=float, default=2.0, help="Source frames per second (default 2; each still lasts 0.5s)")
    parser.add_argument("--mp4", action="store_true", help="Also write an H.264 MP4")
    parser.add_argument("--mp4-name", default="stack.mp4", help="MP4 filename within output-dir")
    parser.add_argument("--webm", action="store_true", help="Also write a VP9 WebM")
    parser.add_argument("--webm-name", default="stack.webm", help="WebM filename within output-dir")
    parser.add_argument("--webp", action="store_true", help="Also write an animated WebP")
    parser.add_argument("--webp-name", default="stack.webp", help="WebP filename within output-dir")
    parser.add_argument(
        "--crossfade", type=float, default=0.0, metavar="SECONDS",
        help="Crossfade animated outputs, including last-to-first looping (default: disabled)",
    )
    parser.add_argument(
        "--animation-fps", type=float, default=20.0,
        help="Render FPS used when crossfading (default 20)",
    )
    parser.add_argument(
        "--delete-frames", action="store_true",
        help="Delete aligned PNG frames after every requested animation succeeds",
    )
    parser.add_argument("-s", "--save-overlay", action="store_true", help="Save a blended overlay preview for quick alignment check")
    parser.add_argument("-d", "--debug", action="store_true", help="Save debug visuals (matches/masks)")
    parser.add_argument(
        "-m",
        "--model",
        choices=["auto", "homography", "affine", "similarity"],
        default="auto",
        help=(
            "Geometric model to fit.\n"
            "  auto         try homography then affine (default)\n"
            "  homography   full perspective warp\n"
            "  affine       rotate/scale/shear\n"
            "  similarity   rotate/scale only (stable for handheld)"
        ),
    )
    parser.add_argument(
        "-t",
        "--gif-tool",
        choices=["auto", "imageio", "magick"],
        default="auto",
        help=(
            "How to write GIFs.\n"
            "  auto     prefer imageio, fallback to magick/convert\n"
            "  imageio  pure Python (no external binary)\n"
            "  magick   requires ImageMagick's magick/convert in PATH"
        ),
    )
    parser.add_argument(
        "--subject-detector",
        choices=["yunet", "person", "pose", "nanodet", "yolox"],
        default="yolox",
        help=(
            "Detector for --align-mode subject.\n"
            "  yunet    faces and five landmarks\n"
            "  person   MediaPipe person detector\n"
            "  pose     person detector refined by body landmarks\n"
            "  nanodet  lightweight COCO object detector\n"
            "  yolox    stronger COCO object detector (default)"
        ),
    )
    parser.add_argument(
        "--subject-class", default="person",
        help="COCO class for NanoDet/YOLOX subject locking (default person)",
    )
    parser.add_argument(
        "--subject-selection",
        choices=["all", "largest", "leftmost", "rightmost"],
        default=None,
        help="Which matching detections to lock (default: largest; face alias: all)",
    )
    parser.add_argument(
        "--subject-count", type=int, default=0,
        help="Maximum highest-confidence detections for 'all'; 0 keeps all",
    )
    parser.add_argument(
        "--subject-confidence", type=float, default=None,
        help="Override the selected detector's confidence threshold",
    )
    parser.add_argument(
        "--subject-model", default=None,
        help="Override model path for YuNet, person, NanoDet, or YOLOX",
    )
    parser.add_argument(
        "--subject-size",
        choices=["diagonal", "width", "height", "area"],
        default="diagonal",
        help="Measurement held constant across frames (default diagonal)",
    )
    parser.add_argument(
        "--subject-tracker", choices=["none", "vittrack"], default="none",
        help="Optionally track the selected union between ordered frames",
    )

    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.animation_fps <= 0:
        parser.error("--animation-fps must be greater than zero")
    if args.crossfade < 0:
        parser.error("--crossfade cannot be negative")
    if args.subject_count < 0:
        parser.error("--subject-count cannot be negative")
    if args.subject_class not in COCO_CLASSES:
        parser.error(
            f"Unknown --subject-class {args.subject_class!r}. "
            f"Choose one of: {', '.join(COCO_CLASSES)}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    paths = list_images(args.input)
    if len(paths) < 2:
        print("Need at least 2 images in the input directory.", file=sys.stderr)
        sys.exit(1)

    images = load_images(paths)

    ref_idx = args.reference_index
    if ref_idx is None:
        ref_idx = len(images) // 2
    if not (0 <= ref_idx < len(images)):
        print("reference-index out of range", file=sys.stderr)
        sys.exit(1)

    if args.align_mode in ("face", "subject"):
        detector_name = "yunet" if args.align_mode == "face" else args.subject_detector
        selection = args.subject_selection or ("all" if args.align_mode == "face" else "largest")
        subject_count = args.subject_count
        if args.align_mode == "face" and subject_count == 0:
            subject_count = 2
        try:
            cropped, _groups = align_subject_stack(
                images=images,
                names=[os.path.basename(path) for path in paths],
                reference_index=ref_idx,
                detector_name=detector_name,
                selection=selection,
                count=subject_count,
                class_name=args.subject_class,
                confidence=args.subject_confidence,
                explicit_model=args.subject_model,
                size_metric=args.subject_size,
                tracker_name=args.subject_tracker,
                debug_dir=args.output_dir if args.debug else None,
            )
            write_outputs(args, paths, cropped)
        except RuntimeError as error:
            print(f"Subject alignment failed: {error}", file=sys.stderr)
            sys.exit(1)
        print(
            f"Subject-locked {len(paths)} images with {detector_name}. "
            f"Outputs in: {args.output_dir}"
        )
        return

    heights = [im.shape[0] for im in images]
    widths = [im.shape[1] for im in images]

    # Fit every image to one canvas. Merely scaling by the same bounds is not
    # enough: portrait and landscape inputs would still have different shapes.
    min_h = min(heights)
    min_w = min(widths)
    resized, source_valid_masks = fit_images_to_common_canvas(images, min_w, min_h)

    ref = resized[ref_idx]

    # Prepare masks depending on mode
    per_image_masks: List[Optional[np.ndarray]]
    if args.align_mode == "static":
        # Motion-based static masks
        per_image_masks = compute_static_region_masks(resized)
        per_image_masks = [
            cv2.bitwise_and(mask, valid)
            for mask, valid in zip(per_image_masks, source_valid_masks)
        ]
        if args.debug:
            for i, m in enumerate(per_image_masks):
                try:
                    cv2.imwrite(os.path.join(args.output_dir, f"{i:02d}_static_mask.png"), m)
                except Exception:
                    pass
    else:  # features
        per_image_masks = source_valid_masks

    transforms: List[Tuple[str, np.ndarray]] = []
    for idx, img in enumerate(resized):
        if idx == ref_idx:
            transforms.append(("identity", np.eye(3, dtype=np.float32)))
            continue
        m_from = per_image_masks[idx]
        m_to = per_image_masks[ref_idx]
        dbg = args.debug
        dbg_prefix = os.path.join(args.output_dir, f"{idx:02d}_to_{ref_idx:02d}") if dbg else None
        ttype, M = compute_transform(
            img,
            ref,
            m_from,
            m_to,
            debug=dbg,
            debug_prefix=dbg_prefix,
            preferred_model=args.model,
        )
        # Normalize transform to homography shape for bookkeeping
        if ttype == "affine" and M.shape == (2, 3):
            H = np.eye(3, dtype=np.float32)
            H[:2, :3] = M
            transforms.append(("affine", M))
        elif ttype == "homography" and M.shape == (3, 3):
            transforms.append(("homography", M))
        else:
            transforms.append(("identity", np.eye(3, dtype=np.float32)))

    # Warp all to reference canvas size
    out_h, out_w = ref.shape[:2]
    warped_images: List[np.ndarray] = []
    warped_masks: List[np.ndarray] = []

    for idx, (ttype, M) in enumerate(transforms):
        img = resized[idx]
        valid = source_valid_masks[idx]
        if ttype == "homography":
            warped = cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpPerspective(valid, M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        elif ttype == "affine":
            warped = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpAffine(valid, M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        else:
            warped = img.copy()
            mask = valid.copy()
        warped_images.append(warped)
        warped_masks.append(mask)

    # Intersection crop
    bbox = intersection_bounding_rect(warped_masks)
    if bbox is None:
        # Fallback: crop to the smallest central area among images
        x = int(0.05 * out_w)
        y = int(0.05 * out_h)
        w = int(0.90 * out_w)
        h = int(0.90 * out_h)
        bbox = (x, y, w, h)
    x, y, w, h = bbox
    cropped = [im[y:y+h, x:x+w] for im in warped_images]

    try:
        write_outputs(args, paths, cropped)
    except RuntimeError as error:
        print(f"Output failed: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Aligned {len(paths)} images. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
