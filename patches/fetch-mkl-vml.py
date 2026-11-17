#!/usr/bin/env python3
"""Fetch pinned Linux x86-64 host preparation libraries, including licenses.

Usage: python3 patches/fetch-mkl-vml.py [destination]
No Python package installation or Torch dependency is required.
"""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import shutil
import sys
import zipfile

PACKAGES = [
    {
        "version": "2021.1.1",
        "url": "https://files.pythonhosted.org/packages/4c/a0/01a55415e1d85e10ce46f1a90887fe97e5ccfb7340c0c4023b43d683b542/mkl-2021.1.1-py2.py3-none-manylinux1_x86_64.whl",
        "sha256": "9b59d5a78e9d4c9da6c8ef053549c0a44966b8fc318c0f4886c11e3ffed6f769",
        "filename": "mkl-2021.1.1-py2.py3-none-manylinux1_x86_64.whl"
    },
    {
        "version": "2021.1.1",
        "url": "https://files.pythonhosted.org/packages/7d/70/86d0db59598c34a5c1d334ba996271dddad89108127b743c84beb6354afd/mkl_include-2021.1.1-py2.py3-none-manylinux1_x86_64.whl",
        "sha256": "865c884473b0a76da201fe972e68c3b2591e6580753485548acc32169db3ffe7",
        "filename": "mkl_include-2021.1.1-py2.py3-none-manylinux1_x86_64.whl"
    }
]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".deps/oneMKL-2021.1.1")
    cache = root / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for package in PACKAGES:
        archive = cache / package["filename"]
        if not archive.exists():
            pending = archive.with_suffix(".partial")
            with urlopen(package["url"]) as source, pending.open("wb") as target:
                shutil.copyfileobj(source, target)
            pending.replace(archive)
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != package["sha256"]:
            raise RuntimeError("SHA256 mismatch: " + str(archive))
        with zipfile.ZipFile(archive) as wheel:
            for name in wheel.namelist():
                if name.endswith("/"):
                    continue
                if "/data/lib/" in name and ".so" in Path(name).name:
                    relative = Path("lib") / Path(name).name
                elif "/data/include/" in name:
                    relative = Path("include") / name.split("/data/include/", 1)[1]
                elif "LICENSE" in name.upper():
                    relative = Path("licenses") / package["filename"] / Path(name).name
                else:
                    continue
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Unsafe archive path: " + name)
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with wheel.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    (root / "sources.json").write_text(json.dumps(PACKAGES, indent=2) + "\n")
    print("Prepared oneMKL 2021.1.1 in " + str(root))


if __name__ == "__main__":
    main()
