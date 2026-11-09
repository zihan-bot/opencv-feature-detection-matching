"""Shared image, timing, and geometry utilities."""
import csv
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINALS = ROOT / "inputs" / "originals"
PAIRS = ROOT / "inputs" / "pairs"
RESULTS = ROOT / "results"
IMAGES = sorted(path for path in ORIGINALS.iterdir() if path.suffix.lower() in {".png", ".jpg"})
REPEATS = 11
ERROR_PX = 3.0

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


def load(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def benchmark(operation):
    result = operation()  # Warm up before timing.
    times = []
    for _ in range(REPEATS):
        start = perf_counter_ns()
        result = operation()
        times.append((perf_counter_ns() - start) / 1_000_000)
    return result, median(times)


def write_csv(name, rows):
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def affine(image, degrees=0, scale=1):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), degrees, scale)
    return np.vstack((matrix, [0, 0, 1]))


def warp(image, matrix):
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


def project(points, matrix):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        return points
    return cv2.perspectiveTransform(points[None], matrix)[0]


def ms(value):
    return round(value, 4)
