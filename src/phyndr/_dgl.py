"""Central DGL import used by the project."""
from __future__ import annotations

import os

os.environ.setdefault("DGLBACKEND", "pytorch")

# DGL core heterographs are the only required component.  Some Windows wheels
# expose this switch to avoid loading an incompatible optional GraphBolt DLL.
if os.getenv("PHY_NDR_DISABLE_GRAPHBOLT") == "1":
    os.environ.setdefault("DGL_DISABLE_GRAPHBOLT", "1")

import dgl  # noqa: E402

__all__ = ["dgl"]

