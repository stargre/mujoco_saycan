"""Install the Franka Panda model from MuJoCo Menagerie.

The installer uses Git sparse checkout instead of the GitHub Contents API, so
anonymous API rate limits cannot break setup. Existing asset files are never
overwritten unless ``--force`` is supplied.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"


def sparse_checkout(destination, ref):
    clone = destination / "mujoco_menagerie"
    command = [
        "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
        "--branch", ref, MENAGERIE_REPOSITORY, str(clone),
    ]
    try:
        subprocess.run(command, check=True)
        subprocess.run(
            ["git", "-C", str(clone), "sparse-checkout", "set", "franka_emika_panda"],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Unable to sparse-clone MuJoCo Menagerie. Git must be installed "
            "and github.com must be reachable. If Menagerie already exists, "
            "skip this script and set PANDA_MODEL_DIR to its "
            "franka_emika_panda directory."
        ) from exc
    return clone


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
    configured = os.getenv("PANDA_MODEL_DIR", "").strip()
    if configured and (Path(configured).expanduser() / "scene.xml").is_file():
        print(f"Using existing PANDA_MODEL_DIR: {Path(configured).expanduser().resolve()}")
        return
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
        clone = sparse_checkout(Path(temp), args.ref)
        staged = clone / "franka_emika_panda"
        files = [path for path in staged.rglob("*") if path.is_file()]
        required = [staged / "scene.xml", staged / "panda.xml"]
        missing = [str(path.name) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Downloaded model is incomplete; missing {missing}")
        if destination.exists() and args.force:
            shutil.rmtree(destination)
        shutil.copytree(staged, destination)
        license_path = destination.parent / "MUJOCO_MENAGERIE_LICENSE"
        shutil.copy2(clone / "LICENSE", license_path)
    print(f"\nInstalled {len(files)} files to: {destination}")
    print(f"License copied to: {license_path}")


if __name__ == "__main__":
    main()

