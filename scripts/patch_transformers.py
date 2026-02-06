#!/usr/bin/env python3
"""
Patch transformers installation with HunyuanVL fixes.

This is a FALLBACK script - normally `uv sync` should install transformers
from the GitHub repo with the patch already applied.

Use this script only if:
1. uv sync fails to fetch from GitHub
2. You installed transformers from PyPI instead of the patched repo

The patch fixes:
1. Attention implementation check for None values
2. Custom from_pretrained to bypass meta device initialization
   (which corrupts weights and causes garbage output)

Run this after `uv sync` if needed:
    python scripts/patch_transformers.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_site_packages() -> Path:
    """Find the site-packages directory."""
    for path in sys.path:
        if "site-packages" in path:
            return Path(path)
    raise RuntimeError("Could not find site-packages directory")


def get_patch_source() -> Path:
    """Get the patched file, cloning from GitHub if needed."""
    # Try local path first (for development)
    local_source = Path("/tmp/transformers_merge/src/transformers/models/hunyuan_vl/modeling_hunyuan_vl.py")
    if local_source.exists():
        return local_source

    # Clone from GitHub
    print("Fetching patch from GitHub...")
    repo_url = "https://github.com/JuneLeGency/transformers_hy.git"
    branch = "hunyuan-vl-patch"

    tmp_dir = Path(tempfile.mkdtemp())
    clone_dir = tmp_dir / "transformers_hy"

    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", branch, repo_url, str(clone_dir)],
        check=True,
        capture_output=True,
    )

    source = clone_dir / "src" / "transformers" / "models" / "hunyuan_vl" / "modeling_hunyuan_vl.py"
    if not source.exists():
        raise RuntimeError(f"Patch file not found in cloned repo: {source}")

    return source


def patch_transformers():
    """Copy patched HunyuanVL modeling file to site-packages."""
    source = get_patch_source()

    if not source.exists():
        print(f"Error: Source file not found: {source}")
        print("Please ensure /tmp/transformers_merge exists with the patched transformers.")
        return False

    site_packages = find_site_packages()
    dest = site_packages / "transformers" / "models" / "hunyuan_vl" / "modeling_hunyuan_vl.py"

    if not dest.parent.exists():
        print(f"Error: Destination directory not found: {dest.parent}")
        print("Please ensure transformers is installed.")
        return False

    # Backup original
    backup = dest.with_suffix(".py.bak")
    if dest.exists() and not backup.exists():
        shutil.copy(dest, backup)
        print(f"Backed up original to: {backup}")

    # Copy patched file
    shutil.copy(source, dest)
    print(f"Patched: {dest}")

    # Verify
    content = dest.read_text()
    if "bypasses meta device" in content:
        print("Verification: OK - Custom from_pretrained found")
        return True
    else:
        print("Verification: FAILED - Patch not applied correctly")
        return False


if __name__ == "__main__":
    success = patch_transformers()
    sys.exit(0 if success else 1)
