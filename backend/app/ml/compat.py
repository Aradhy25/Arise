"""Runtime compatibility shims for incomplete Python builds (e.g. pyenv without xz)."""

from __future__ import annotations

import sys
import types


def ensure_lzma() -> None:
    """Stub stdlib lzma if _lzma is missing so torchvision can import.

    Proper fix on macOS:
        brew install xz
        pyenv install --force 3.11.9
        # then recreate the venv
    """
    try:
        import lzma  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType("lzma")
    stub.LZMAError = OSError  # type: ignore[attr-defined]
    stub.open = None  # type: ignore[attr-defined]
    stub.compress = lambda data, *a, **k: data  # type: ignore[attr-defined]
    stub.decompress = lambda data, *a, **k: data  # type: ignore[attr-defined]
    sys.modules["lzma"] = stub


ensure_lzma()
