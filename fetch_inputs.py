"""Fetch and verify the exact scikit-image sample files used in the lab."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

ORIGINALS = Path(__file__).resolve().parent / "inputs" / "originals"
DATA_URL = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/"
SHA256 = {
    "astronaut.png": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    "brick.png": "7966caf324f6ba843118d98f7a07746d22f6a343430add0233eca5f6eaaa8fcf",
    "camera.png": "b0793d2adda0fa6ae899c03989482bff9a42d3d5690fc7e3648f2795d730c23a",
    "chelsea.png": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
    "coffee.png": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    "coins.png": "f8d773fc9cfa6f4d8e5942dc34d0a0788fcaed2a4fefbbed0aef5398d7ef4cba",
    "gravel.png": "c48615b451bf1e606fbd72c0aa9f8cc0f068ab7111ef7d93bb9b0f2586440c12",
    "moon.png": "78739619d11f7eb9c165bb5d2efd4772cee557812ec847532dbb1d92ef71f577",
    "page.png": "341a6f0a61557662b02734a9b6e56ec33a915b2c41886b97509dedf2a43b47a3",
    "rocket.jpg": "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c",
}


def main():
    ORIGINALS.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in SHA256.items():
        path = ORIGINALS / filename
        if path.is_file() and sha256(path.read_bytes()).hexdigest() == expected_hash:
            continue
        with urlopen(DATA_URL + filename, timeout=30) as response:
            data = response.read()
        if sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"Unexpected SHA-256 for {filename}")
        path.write_bytes(data)
        print(f"Downloaded {filename}")


if __name__ == "__main__":
    main()
