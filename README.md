# OpenCV Feature Detection and Matching

A computer vision lab with two experiments.

1. **Detection** compares SIFT and FAST on ten images. It measures keypoint count, detector-only runtime, spatial coverage, and redetection after rotation or resizing.
2. **Matching** compares SIFT and ORB descriptors on ten image pairs. Both descriptors use the same retained SIFT keypoints. Brute-force matching uses L2 for SIFT and Hamming for ORB. Known transformations provide the expected positions for scoring.

## Run the experiments

Use Python 3.9 or later. Run these commands from the repository root.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_inputs.py
python detect.py
python match.py
```

The input script downloads ten versioned scikit-image sample images and checks their SHA-256 hashes. See [image credits](IMAGE_CREDITS.md). The matching script generates the ten target images and the exact source-to-target transforms in `inputs/pairs/`.

## Output

| Location | Contents |
| --- | --- |
| `results/detection/` | Twenty overlays, with yellow markers at detected keypoints. |
| `results/matching/` | Twenty side-by-side match views. Green lines show geometrically correct matches and red lines show larger errors. |
| `results/detection.csv` | Keypoint counts, detector runtimes, grid coverage, and redetection rates. |
| `results/matching.csv` | Keypoint retention, accepted and correct counts, precision, and per-stage timing. |

Each stage uses one OpenCV thread, one warm-up call, and 11 timed calls. Runtime depends on the machine. Matching views show up to 24 matches ranked by descriptor ratio. The CSV metrics use all accepted matches and a 3-pixel threshold.
