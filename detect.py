"""Experiment 1: compare SIFT and FAST keypoint detection."""
import cv2
import numpy as np

from common import ERROR_PX, IMAGES, RESULTS, affine, benchmark, gray, load, ms, project, save, warp, write_csv


def positions(keypoints):
    return np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(-1, 2)


def occupied_cells(keypoints, shape):
    height, width = shape
    cells = (positions(keypoints) * [4 / width, 4 / height]).astype(int)
    return len(np.unique(cells, axis=0))


def inside(points, width, height, margin=12):
    return ((points >= margin) & (points < [width - margin, height - margin])).all(axis=1)


def redetection(detector, keypoints, image, matrix):
    """Directed coverage of unique original locations in the overlap."""
    height, width = image.shape
    original = np.unique(positions(keypoints), axis=0)
    projected = project(original[inside(original, width, height)], matrix)
    projected = projected[inside(projected, width, height)]
    target = positions(detector.detect(warp(image, matrix), None))
    if len(target):
        original_frame = project(target, np.linalg.inv(matrix))
        target = target[inside(original_frame, width, height, margin=8)]
    if not len(projected) or not len(target):
        return 0.0
    distances, _ = cv2.batchDistance(projected, target, cv2.CV_32F, normType=cv2.NORM_L2, K=1)
    return round(100 * float(np.mean(distances <= ERROR_PX)), 2)


def main():
    detectors = {
        "sift": cv2.SIFT_create(),
        "fast": cv2.FastFeatureDetector_create(
            threshold=10, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
        ),
    }
    rows = []
    for path in IMAGES:
        image = load(path)
        intensity = gray(image)
        height, width = intensity.shape
        row = {"image": path.stem, "width": width, "height": height}
        for name, detector in detectors.items():
            keypoints, duration = benchmark(lambda: detector.detect(intensity, None))
            row.update({
                f"{name}_count": len(keypoints),
                f"{name}_ms": ms(duration),
                f"{name}_grid_cells_of_16": occupied_cells(keypoints, intensity.shape),
                f"{name}_rotation_pct": redetection(detector, keypoints, intensity, affine(image, degrees=30)),
                f"{name}_scale_pct": redetection(detector, keypoints, intensity, affine(image, scale=0.70)),
            })
            overlay = cv2.drawKeypoints(image, keypoints, None, color=(35, 227, 255))
            save(RESULTS / "detection" / f"{path.stem}_{name}.png", overlay)
        rows.append(row)
    write_csv("detection.csv", rows)


if __name__ == "__main__":
    main()
