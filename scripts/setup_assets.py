"""Download the Franka Panda model from MuJoCo Menagerie.

Only the ``franka_emika_panda`` directory is downloaded. Existing asset files
are never overwritten unless ``--force`` is supplied.
"""
from pathlib import Path
import argparse
import json
import shutil
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents"
RAW_LICENSE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/LICENSE"
USER_AGENT = "mujoco-saycan-asset-installer/1.0"


def request_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def content_listing(path, ref):
    url = f"{API_ROOT}/{path}?ref={ref}"
    return json.loads(request_bytes(url).decode("utf-8"))


def download_directory(repository_path, destination, ref):
    downloaded = []
    for item in content_listing(repository_path, ref):
        target = destination / item["name"]
        if item["type"] == "dir":
            target.mkdir(parents=True, exist_ok=True)
            downloaded.extend(download_directory(item["path"], target, ref))
        elif item["type"] == "file":
            target.write_bytes(request_bytes(item["download_url"]))
            downloaded.append(target)
            print(f"downloaded {item['path']}")
    return downloaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        default=str(ROOT / "assets" / "franka_emika_panda"),
    )
    parser.add_argument("--ref", default="main")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = Path(args.destination).expanduser().resolve()
    if (destination / "scene.xml").is_file() and not args.force:
        print(f"Panda assets already installed: {destination}")
        return
    if destination.exists() and any(destination.iterdir()) and not args.force:
        raise SystemExit(
            f"Refusing to overwrite non-empty directory: {destination}\n"
            "Use --force only if you intentionally want to replace its files."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mujoco_saycan_assets_") as temp:
        staged = Path(temp) / "franka_emika_panda"
        staged.mkdir()
        files = download_directory("franka_emika_panda", staged, args.ref)
        required = [staged / "scene.xml", staged / "panda.xml"]
        missing = [str(path.name) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Downloaded model is incomplete; missing {missing}")
        if destination.exists() and args.force:
            shutil.rmtree(destination)
        shutil.copytree(staged, destination)

    license_path = destination.parent / "MUJOCO_MENAGERIE_LICENSE"
    license_path.write_bytes(request_bytes(RAW_LICENSE))
    print(f"\nInstalled {len(files)} files to: {destination}")
    print(f"License copied to: {license_path}")


if __name__ == "__main__":
    main()

