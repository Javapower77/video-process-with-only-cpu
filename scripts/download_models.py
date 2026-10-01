"""Download official assets atomically; reject archive path traversal."""
import argparse
import hashlib
import platform
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
ASSETS = {
    "realesr-animevideov3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
}
FACE_ASSETS = {
    "detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing_parsenet.pth": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
}
RIFE_URL = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip"


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        print(f"Already installed: {destination.name}")
        return
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS downloads are allowed")
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        print(f"Downloading {destination.name}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "CPU-Video-Studio/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not temporary.stat().st_size:
            raise RuntimeError("Empty download")
        temporary.replace(destination)
        with destination.open("rb") as downloaded:
            digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
        print(f"SHA256 {destination.name}: {digest}")
    finally:
        temporary.unlink(missing_ok=True)


def extract_safe(archive, target):
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            path = (target / entry.filename).resolve()
            if not path.is_relative_to(target.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive member")
        zipped.extractall(target)


def install_rife():
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("Bundled RIFE targets Linux x86-64. Build from source on this platform.")
    archive = ROOT / ".cache" / "rife-20221029.zip"
    download(RIFE_URL, archive)
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary)
        extract_safe(archive, target)
        binary = next(target.rglob("rife-ncnn-vulkan"), None)
        if binary is None:
            raise RuntimeError("RIFE archive has no executable")
        destination = ROOT / "bin" / binary.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(binary, destination)
        destination.chmod(0o755)
        models = [p for p in target.rglob("rife-v4*") if p.is_dir() and list(p.glob("*.param"))]
        if not models:
            raise RuntimeError("Archive contains no RIFE v4 NCNN models")
        for model in models:
            shutil.copytree(model, WEIGHTS / model.name, dirs_exist_ok=True)
        print("Installed RIFE models:", ", ".join(p.name for p in models))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rife", action="store_true", help="Skip the approximately 431 MB RIFE bundle")
    args = parser.parse_args()
    for name, url in ASSETS.items():
        download(url, WEIGHTS / name)
    for name, url in FACE_ASSETS.items():
        download(url, ROOT / "gfpgan" / "weights" / name)
    if not args.skip_rife:
        install_rife()
    print("Assets installed. Only load trusted model checkpoints; upstream licenses apply.")


if __name__ == "__main__":
    main()