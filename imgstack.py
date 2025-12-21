#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import shutil
import subprocess

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


def detect_face_mask(image_bgr: np.ndarray) -> np.ndarray:
    # Returns mask (uint8 0/255) where faces are 255
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    except Exception:
        cascade_path = None
    if not cascade_path or not os.path.exists(cascade_path):
        # No cascade available; return empty mask
        return np.zeros(gray.shape, dtype=np.uint8)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(40, 40))
    mask = np.zeros_like(gray, dtype=np.uint8)
    for (x, y, w, h) in faces:
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


def blend_preview(images: List[np.ndarray]) -> np.ndarray:
    acc = np.zeros_like(images[0], dtype=np.float32)
    for im in images:
        acc += im.astype(np.float32) / 255.0
    acc /= max(1, len(images))
    return np.clip(acc * 255.0, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Align a stack of images, crop to common area, and optionally make a GIF.")
    parser.add_argument("input", nargs="?", default=".", help="Input directory containing images (default: current directory)")
    parser.add_argument("--align-mode", choices=["static", "face", "features"], default="static", help="Alignment focus: 'static' masks out faces (default), 'face' focuses on faces, 'features' uses all features.")
    parser.add_argument("--reference-index", type=int, default=None, help="Index of reference image after sorting (default: middle frame)")
    parser.add_argument("--output-dir", default="aligned_out", help="Directory to write outputs")
    parser.add_argument("--gif", action="store_true", help="Also write an animated GIF")
    parser.add_argument("--gif-name", default="stack.gif", help="Filename of the output GIF (within output-dir)")
    parser.add_argument("--fps", type=float, default=4.0, help="GIF frames per second (default 4)")
    parser.add_argument("--save-overlay", action="store_true", help="Save a blended overlay preview for quick alignment check")
    parser.add_argument("--debug", action="store_true", help="Save debug visuals (matches/masks)")
    parser.add_argument("--model", choices=["auto", "homography", "affine", "similarity"], default="auto", help="Geometric model to fit. 'similarity' is most stable for handheld shots.")
    parser.add_argument("--gif-tool", choices=["auto", "imageio", "magick"], default="auto", help="How to write GIFs: Python imageio or ImageMagick (magick/convert)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paths = list_images(args.input)
    if len(paths) < 2:
        print("Need at least 2 images in the input directory.", file=sys.stderr)
        sys.exit(1)

    images = load_images(paths)
    heights = [im.shape[0] for im in images]
    widths = [im.shape[1] for im in images]

    # Resize to the smallest size among images to avoid upscaling during matching
    min_h = min(heights)
    min_w = min(widths)
    resized = []
    resize_scales = []
    for im in images:
        h, w = im.shape[:2]
        scale = min(min_h / h, min_w / w)
        if abs(scale - 1.0) < 1e-6:
            resized.append(im)
            resize_scales.append(1.0)
        else:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            resized.append(cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA))
            resize_scales.append(scale)

    # Choose reference
    ref_idx = args.reference_index
    if ref_idx is None:
        ref_idx = len(resized) // 2
    if not (0 <= ref_idx < len(resized)):
        print("reference-index out of range", file=sys.stderr)
        sys.exit(1)

    ref = resized[ref_idx]

    # Prepare masks depending on mode
    per_image_masks: List[Optional[np.ndarray]]
    if args.align_mode == "static":
        # Motion-based static masks
        per_image_masks = compute_static_region_masks(resized)
        if args.debug:
            for i, m in enumerate(per_image_masks):
                try:
                    cv2.imwrite(os.path.join(args.output_dir, f"{i:02d}_static_mask.png"), m)
                except Exception:
                    pass
    elif args.align_mode == "face":
        per_image_masks = [detect_face_mask(im) for im in resized]
        if args.debug:
            for i, m in enumerate(per_image_masks):
                cv2.imwrite(os.path.join(args.output_dir, f"{i:02d}_face_mask.png"), m)
    else:  # features
        per_image_masks = [None for _ in resized]

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
        if ttype == "homography":
            warped = cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpPerspective(np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8), M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        elif ttype == "affine":
            warped = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpAffine(np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8), M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        else:
            warped = img.copy()
            mask = np.full((out_h, out_w), 255, dtype=np.uint8)
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

    # Write aligned frames
    aligned_paths: List[str] = []
    for i, (src_path, im) in enumerate(zip(paths, cropped)):
        base = os.path.splitext(os.path.basename(src_path))[0]
        out_path = os.path.join(args.output_dir, f"{i:02d}_{base}_aligned.png")
        cv2.imwrite(out_path, im)
        aligned_paths.append(out_path)

    if args.save_overlay:
        overlay = blend_preview(cropped)
        cv2.imwrite(os.path.join(args.output_dir, "overlay_preview.png"), overlay)

    if args.gif:
        gif_path = os.path.join(args.output_dir, args.gif_name)
        try:
            save_gif_auto(cropped, aligned_paths, gif_path, fps=args.fps, tool=args.gif_tool)
        except Exception as e:
            print(f"Failed to save GIF: {e}", file=sys.stderr)

    print(f"Aligned {len(paths)} images. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
