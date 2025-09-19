"""cpmcu package initialization.

This module tries to import the compiled CUDA extension `cpmcu.C` from the
installed package first. If that fails (e.g., in local development without
installation), it falls back to locating the built shared library under the
repository's `build/` directory and loads it dynamically.

Long-term structure guideline:
  - Keep Python sources in `cpmcu/` (this package)
  - Keep compiled artifacts in top-level `build/` (e.g., build/lib.*/*)
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
import sysconfig
from types import ModuleType
import os


def _load_extension_from_build() -> ModuleType | None:
    package_name = __name__  # e.g., "cpmcu"
    so_module_qualified = f"{package_name}.C"

    # Repo root candidates: current package dir -> parent is repo root in dev
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    build_dir = repo_root / "build"
    if not build_dir.exists():
        return None

    cache_tag = getattr(sys.implementation, "cache_tag", None)  # e.g., cpython-311
    soabi = sysconfig.get_config_var("SOABI")  # e.g., cpython-311-x86_64-linux-gnu

    # Collect candidate lib.* directories that match current interpreter
    matched_lib_dirs: list[Path] = []
    for child in build_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name  # e.g., lib.linux-x86_64-cpython-311
        if not name.startswith("lib."):
            continue
        if (soabi and soabi in name) or (cache_tag and cache_tag in name):
            matched_lib_dirs.append(child)
    # Only accept directories that match current interpreter; otherwise, skip fallback
    lib_dirs: list[Path] = matched_lib_dirs
    if not lib_dirs:
        return None

    for lib_dir in lib_dirs:
        pkg_in_build = lib_dir / package_name
        if not pkg_in_build.is_dir():
            continue
        # Look for extension files like C.cpython-311-x86_64-linux-gnu.so (matching current ABI)
        pattern = "C*.so"
        so_candidates_all = list(pkg_in_build.glob(pattern))
        if not so_candidates_all:
            continue
        so_candidates = [p for p in so_candidates_all if (soabi and soabi in p.name) or (cache_tag and cache_tag in p.name)]
        if not so_candidates:
            continue
        # Prefer exact SOABI match
        def score(p: Path) -> int:
            return 0 if (soabi and soabi in p.name) else 1

        for candidate in sorted(so_candidates, key=score):
            try:
                loader = importlib.machinery.ExtensionFileLoader(so_module_qualified, str(candidate))
                spec = importlib.util.spec_from_loader(so_module_qualified, loader)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[so_module_qualified] = module
                return module
            except Exception:
                # Try next candidate if loading fails (ABI mismatch, etc.)
                continue

    return None


def _ensure_torch_loaded() -> None:
    # Ensure PyTorch shared libraries (e.g., libc10.so) are loaded before our extension
    import torch  # noqa: F401


def _load_extension_from_installed_distribution() -> ModuleType | None:
    """Load cpmcu.C from an installed wheel distribution even if the local source
    tree shadows site-packages. Uses importlib.metadata to locate files robustly."""
    try:
        import importlib.metadata as importlib_metadata  # Python 3.8+
    except Exception:
        return None

    try:
        dist = importlib_metadata.distribution("cpmcu")
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None

    soabi = sysconfig.get_config_var("SOABI")
    cache_tag = getattr(sys.implementation, "cache_tag", None)

    # Find candidate so files recorded in the wheel metadata
    candidates: list[Path] = []
    for f in (dist.files or []):  # type: ignore[attr-defined]
        try:
            if not str(f).startswith("cpmcu/"):
                continue
            name = os.path.basename(str(f))
            if not name.startswith("C") or not name.endswith(".so"):
                continue
            p = Path(dist.locate_file(f))
            candidates.append(p)
        except Exception:
            continue

    if not candidates:
        return None

    # Filter for current interpreter ABI
    filtered = [p for p in candidates if (soabi and soabi in p.name) or (cache_tag and cache_tag in p.name)]
    if not filtered:
        return None

    # Prefer exact SOABI match
    filtered.sort(key=lambda p: 0 if (soabi and soabi in p.name) else 1)

    package_name = __name__
    so_module_qualified = f"{package_name}.C"
    for candidate in filtered:
        try:
            loader = importlib.machinery.ExtensionFileLoader(so_module_qualified, str(candidate))
            spec = importlib.util.spec_from_loader(so_module_qualified, loader)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[so_module_qualified] = module
            return module
        except Exception:
            continue
    return None


_ensure_torch_loaded()
_C = _load_extension_from_installed_distribution()
if _C is None:
    disable_build_fallback = os.getenv("CPMCU_DISABLE_BUILD_FALLBACK", "0").lower() in ("1", "true", "yes")
    if not disable_build_fallback:
        _C = _load_extension_from_build()
if _C is None:
    raise ImportError(
        "Failed to import extension 'cpmcu.C'. Please install via 'pip install .' "
        "or build locally (python setup.py build_ext)."
    )


# Maintain backward compatibility: allow `from cpmcu import C`
C = _C  # type: ignore

__all__ = ["_C", "C"]


