"""URI-aware path helpers — local filesystem **or** ``s3://`` (and any
other fsspec-known scheme).

Why this exists
---------------

For the SageMaker / S3 deployment we don't know upfront whether the
workshop data will be:

1. mounted as a local filesystem (FSx for Lustre, Mountpoint-for-S3),
   in which case the bundle root is e.g. ``/home/sagemaker-user/data``;
2. pre-downloaded to local EBS at instance startup, same as (1);
3. accessed directly via ``s3://bucket/key`` URIs.

zarr and polars already accept ``s3://`` strings when ``s3fs`` is
installed. The remaining surfaces in this package use ``pathlib.Path``
for ``.exists()``, ``.read_text()`` and parent/child math. This module
provides minimal fsspec-aware equivalents so the loader works for all
three patterns without forking the codebase.

Local paths stay ``pathlib.Path`` objects (so all existing tests keep
working byte-for-byte); URIs are kept as plain strings.

Convention: every helper accepts ``str | Path`` and returns either a
``Path`` (for local roots) or a ``str`` (for URIs). Downstream code
should call ``as_str(...)`` before passing to zarr / polars / fsspec.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any


# Anything matching this prefix is treated as a remote URI handled by
# fsspec. ``file://`` is allowed but coerced to a local Path.
_URI_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://")


def is_uri(p: Any) -> bool:
    """Return ``True`` if ``p`` is a string with a scheme:// prefix."""
    if not isinstance(p, str):
        return False
    if p.startswith("file://"):
        return False
    return bool(_URI_RE.match(p))


def as_str(p: str | Path) -> str:
    """Coerce a Path/URI to the string form zarr/polars expect."""
    return p if isinstance(p, str) else str(p)


def child(root: str | Path, *parts: str) -> str | Path:
    """Join ``root`` with one or more path parts.

    Returns ``Path`` for local roots, ``str`` for URIs.
    """
    if is_uri(root):
        joined = str(root).rstrip("/")
        for part in parts:
            joined = f"{joined}/{str(part).lstrip('/')}"
        return joined
    return Path(root).joinpath(*parts)


def parent(p: str | Path) -> str | Path:
    """Parent of ``p`` (Path for local, str for URIs)."""
    if is_uri(p):
        s = str(p).rstrip("/")
        return s.rsplit("/", 1)[0] if "/" in s.split("://", 1)[1] else s
    return Path(p).parent


def exists(p: str | Path) -> bool:
    """``True`` if ``p`` points at an existing file/dir/object."""
    if is_uri(p):
        try:
            import fsspec
        except ImportError as e:  # pragma: no cover — only when s3fs missing
            raise RuntimeError(
                f"fsspec is required to resolve URI {p!r}. Install s3fs "
                f"for s3:// access (`pip install s3fs`)."
            ) from e
        fs, path = fsspec.core.url_to_fs(str(p))
        return fs.exists(path)
    return Path(p).exists()


def read_text(p: str | Path) -> str:
    """Read a small text file (e.g. ``classes.json``) from local or URI."""
    if is_uri(p):
        import fsspec
        with fsspec.open(str(p), "r") as f:
            return f.read()
    return Path(p).read_text()


def read_bytes(p: str | Path) -> bytes:
    """Read a small binary file (e.g. an ``.npz``) from local or URI."""
    if is_uri(p):
        import fsspec
        with fsspec.open(str(p), "rb") as f:
            return f.read()
    return Path(p).read_bytes()


def load_json(p: str | Path) -> Any:
    """Convenience: ``json.loads(read_text(p))``."""
    return json.loads(read_text(p))


def open_buffered(p: str | Path) -> io.BytesIO:
    """Return an in-memory ``BytesIO`` for ``p`` — usable with
    ``numpy.load`` and other seekable-stream consumers."""
    return io.BytesIO(read_bytes(p))
