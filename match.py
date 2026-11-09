"""Experiment 2: compare SIFT and ORB descriptors on shared SIFT keypoints."""
import json

import cv2
import numpy as np

from common import ERROR_PX, IMAGES, PAIRS, RESULTS, affine, benchmark, gray, load, ms, project, save, warp, write_csv

RATIO = 0.75
PAIR_SETTINGS = {
    "astronaut": ("Rotation +30 deg", 30, 1, 1, 0, False),
    "brick": ("Perspective A", 0, 1, 1, 0, True),
    "camera": ("Rotation -45 deg", -45, 1, 1, 0, False),
    "chelsea": ("Scale 0.70x", 0, 0.70, 1, 0, False),
    "coffee": ("Dimmer", 0, 1, 0.60, 10, False),
    "coins": ("Scale 0.80x", 0, 0.80, 1, 0, False),
    "gravel": ("Perspective B", 0, 1, 1, 0, True),
    "moon": ("Brighter", 0, 1, 1.25, 18, False),
    "page": ("Rotation + scale", 22, 0.83, 1, 0, False),
    "rocket": ("Rotation + dimming", -28, 1, 0.72, 8, False),
}


def make_pair(image, name):
    label, degrees, scale, gain, offset, perspective = PAIR_SETTINGS[name]
    matrix = affine(image, degrees, scale)
    if perspective:
        height, width = image.shape[:2]
        corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
        if name == "brick":
            target = [[.10, .06], [.96, .01], [.87, .96], [.04, .83]]
        else:
            target = [[.03, .12], [.85, .03], [.97, .83], [.14, .96]]
        bounds = [width - 1, height - 1]
        source_xy = np.float32(np.asarray(corners) * bounds)
        target_xy = np.float32(np.asarray(target) * bounds)
        matrix = cv2.getPerspectiveTransform(source_xy, target_xy)
    transformed = warp(image, matrix)
    if gain != 1 or offset:
        transformed = np.clip(transformed * gain + offset, 0, 255).astype(np.uint8)
    return transformed, matrix, label


def orb_keypoints(keypoints):
    # SIFT stores a packed octave code. ORB expects a plain pyramid index.
    # Use ORB level zero and keep the SIFT center, orientation, and identity.
    return [
        cv2.KeyPoint(*point.pt, point.size, point.angle, point.response, 0, point.class_id)
        for point in keypoints
    ]


def shared_keypoints(sift, orb, intensity):
    keypoints = sift.detect(intensity, None)
    for index, point in enumerate(keypoints):
        point.class_id = index
    sift_kept, _ = sift.compute(intensity, keypoints)
    orb_kept, _ = orb.compute(intensity, orb_keypoints(keypoints))
    shared = {point.class_id for point in sift_kept} & {point.class_id for point in orb_kept}
    common = [point for point in keypoints if point.class_id in shared]
    return common, len(keypoints), len(orb_kept)


def filtered_matches(matcher, first, second):
    if first is None or second is None or len(second) < 2:
        return []
    neighbors = matcher.knnMatch(first, second, k=2)
    accepted = []
    for pair in neighbors:
        if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
            accepted.append((pair[0].distance / pair[1].distance, pair[0]))
    return accepted


def show_matches(image, target, first, second, matches, matrix):
    if matches:
        source_xy = [first[match.queryIdx].pt for _, match in matches]
        target_xy = [second[match.trainIdx].pt for _, match in matches]
        errors = np.linalg.norm(project(source_xy, matrix) - target_xy, axis=1)
    else:
        errors = np.empty(0)
    correct = errors <= ERROR_PX
    canvas = cv2.drawMatches(image, first, target, second, [], None, flags=2)
    # Draw a readable subset ranked by descriptor ratio. Score all accepted matches.
    for index in sorted(range(len(matches)), key=lambda i: matches[i][0])[:24]:
        cv2.drawMatches(
            image, first, target, second, [matches[index][1]], canvas,
            matchColor=(70, 240, 65) if correct[index] else (75, 65, 255),
            flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    return canvas, int(correct.sum())


def main():
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nlevels=1, edgeThreshold=31, patchSize=31, WTA_K=2)
    algorithms = {"sift": (sift, cv2.NORM_L2), "orb": (orb, cv2.NORM_HAMMING)}
    rows, manifest = [], []
    for path in IMAGES:
        image = load(path)
        name = path.stem
        target, matrix, label = make_pair(image, name)
        target_path = PAIRS / f"{name}_transformed.png"
        save(target_path, target)
        manifest.append({
            "source": f"originals/{path.name}", "target": f"pairs/{target_path.name}",
            "transform": label, "source_to_target_homography": matrix.tolist(),
        })
        images = (gray(image), gray(target))
        detected = [shared_keypoints(sift, orb, intensity) for intensity in images]
        row = {"pair": name, "transform": label}
        for side, (common, sift_count, orb_count) in zip(("source", "target"), detected):
            row.update({
                f"sift_detected_{side}": sift_count, f"orb_retained_{side}": orb_count,
                f"shared_{side}": len(common),
            })
        for method, (extractor, norm) in algorithms.items():
            points = [entry[0] for entry in detected]
            if method == "orb":
                points = [orb_keypoints(items) for items in points]
            (first, desc1), time1 = benchmark(lambda: extractor.compute(images[0], points[0]))
            (second, desc2), time2 = benchmark(lambda: extractor.compute(images[1], points[1]))
            for expected, actual in zip(points, (first, second)):
                assert sorted(k.class_id for k in expected) == sorted(k.class_id for k in actual)
            matcher = cv2.BFMatcher(norm)
            matches, match_time = benchmark(lambda: filtered_matches(matcher, desc1, desc2))
            canvas, correct = show_matches(image, target, first, second, matches, matrix)
            save(RESULTS / "matching" / f"{name}_{method}.png", canvas)
            row.update({
                f"{method}_accepted": len(matches), f"{method}_correct": correct,
                f"{method}_precision_pct": round(100 * correct / len(matches), 2) if matches else "",
                f"{method}_descriptor_ms": ms(time1 + time2), f"{method}_match_ms": ms(match_time),
            })
        rows.append(row)
    write_csv("matching.csv", rows)
    (PAIRS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
