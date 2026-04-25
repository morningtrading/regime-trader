"""
Asset Groups Registry — single source of truth.

Reads / writes ``config/asset_groups.yaml`` and exposes a typed API for
listing, inspecting and mutating asset groups. All writes are atomic
(tmp file + os.replace) and leave a ``.bak`` snapshot of the previous
version for safety.

Public API (used by main.py, menu, and tools):
    AssetGroup                — immutable dataclass
    AssetGroupRegistry        — load/list/get/add/remove/edit/...
    load_default_registry()   — singleton-ish helper

CLI is wired into ``main.py groups`` (see main.py cmd_groups).
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "config" / "asset_groups.yaml"
SCHEMA_VERSION = 1


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AssetGroup:
    name: str
    symbols: Tuple[str, ...]
    description: str = ""
    asset_class: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)
    warning: str = ""   # non-empty => loudly surfaced on backtest/trade commands

    def to_dict(self) -> Dict:
        out = {
            "description": self.description,
            "asset_class": self.asset_class,
            "tags": list(self.tags),
            "symbols": list(self.symbols),
        }
        if self.warning:
            out["warning"] = self.warning
        return out

    @classmethod
    def from_dict(cls, name: str, data: Mapping) -> "AssetGroup":
        symbols = data.get("symbols") or []
        if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
            raise ValueError(f"Group '{name}': symbols must be a list of strings")
        tags = data.get("tags") or []
        return cls(
            name=name,
            symbols=tuple(s.strip() for s in symbols if s.strip()),
            description=str(data.get("description", "") or ""),
            asset_class=str(data.get("asset_class", "") or ""),
            tags=tuple(str(t) for t in tags),
            warning=str(data.get("warning", "") or ""),
        )


# ── Registry ──────────────────────────────────────────────────────────────────


class AssetGroupRegistry:
    """Loads, mutates and persists asset_groups.yaml.

    Instances are cheap — re-instantiate after external mutations instead of
    mutating across long-lived references.
    """

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self._version: int = SCHEMA_VERSION
        self._default: str = ""
        self._groups: Dict[str, AssetGroup] = {}
        self.load()

    # ── IO ─────────────────────────────────────────────────────────────────
    def load(self) -> None:
        if not self.path.exists():
            logger.warning("Asset groups file missing: %s (empty registry)", self.path)
            self._groups = {}
            return
        with self.path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{self.path}: top-level must be a mapping")
        self._version = int(raw.get("version", SCHEMA_VERSION))
        self._default = str(raw.get("default", "") or "")
        groups_raw = raw.get("groups", {}) or {}
        if not isinstance(groups_raw, dict):
            raise ValueError(f"{self.path}: 'groups' must be a mapping")
        self._groups = {
            name: AssetGroup.from_dict(name, data)
            for name, data in groups_raw.items()
        }
        # default fallback: first group if default unset or invalid
        if self._default not in self._groups and self._groups:
            self._default = next(iter(self._groups))

    def _dump(self) -> Dict:
        return {
            "version": self._version,
            "default": self._default,
            "groups": {name: g.to_dict() for name, g in self._groups.items()},
        }

    def save(self) -> None:
        """Atomic write with .bak snapshot."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            shutil.copy2(self.path, self.path.with_suffix(self.path.suffix + ".bak"))
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self._dump(), fh, sort_keys=False, default_flow_style=None)
        os.replace(tmp, self.path)

    # ── Read ───────────────────────────────────────────────────────────────
    def list(self) -> List[str]:
        return list(self._groups.keys())

    def all(self) -> Dict[str, AssetGroup]:
        return dict(self._groups)

    def get(self, name: str) -> AssetGroup:
        if name not in self._groups:
            raise KeyError(f"Asset group not found: {name!r}. Available: {self.list()}")
        return self._groups[name]

    def has(self, name: str) -> bool:
        return name in self._groups

    def default(self) -> str:
        return self._default

    def filter(self, *, asset_class: Optional[str] = None, tag: Optional[str] = None) -> List[AssetGroup]:
        out = []
        for g in self._groups.values():
            if asset_class and g.asset_class != asset_class:
                continue
            if tag and tag not in g.tags:
                continue
            out.append(g)
        return out

    # ── Mutate ─────────────────────────────────────────────────────────────
    def add(self, group: AssetGroup, *, overwrite: bool = False) -> None:
        if group.name in self._groups and not overwrite:
            raise ValueError(f"Group '{group.name}' already exists (use overwrite=True)")
        self._groups[group.name] = group
        if not self._default:
            self._default = group.name
        self.save()

    def remove(self, name: str) -> None:
        if name not in self._groups:
            raise KeyError(f"Group not found: {name!r}")
        del self._groups[name]
        if self._default == name:
            self._default = next(iter(self._groups), "")
        self.save()

    def update(
        self,
        name: str,
        *,
        symbols: Optional[Sequence[str]] = None,
        add_symbols: Optional[Iterable[str]] = None,
        remove_symbols: Optional[Iterable[str]] = None,
        description: Optional[str] = None,
        asset_class: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> AssetGroup:
        g = self.get(name)
        new_syms = list(g.symbols)
        if symbols is not None:
            new_syms = [s.strip() for s in symbols if s.strip()]
        if add_symbols:
            for s in add_symbols:
                s = s.strip()
                if s and s not in new_syms:
                    new_syms.append(s)
        if remove_symbols:
            to_drop = {s.strip() for s in remove_symbols}
            new_syms = [s for s in new_syms if s not in to_drop]
        updated = replace(
            g,
            symbols=tuple(new_syms),
            description=g.description if description is None else description,
            asset_class=g.asset_class if asset_class is None else asset_class,
            tags=g.tags if tags is None else tuple(tags),
        )
        self._groups[name] = updated
        self.save()
        return updated

    def rename(self, old: str, new: str) -> None:
        if new in self._groups:
            raise ValueError(f"Target name '{new}' already exists")
        g = self.get(old)
        self._groups[new] = replace(g, name=new)
        del self._groups[old]
        if self._default == old:
            self._default = new
        self.save()

    def set_default(self, name: str) -> None:
        if name not in self._groups:
            raise KeyError(f"Group not found: {name!r}")
        self._default = name
        self.save()

    # ── Validation ─────────────────────────────────────────────────────────
    def validate(self) -> List[str]:
        errs: List[str] = []
        if not self._groups:
            errs.append("no groups defined")
        if self._default and self._default not in self._groups:
            errs.append(f"default '{self._default}' points to unknown group")
        for name, g in self._groups.items():
            if not g.symbols:
                errs.append(f"group '{name}': empty symbols list")
            if len(set(g.symbols)) != len(g.symbols):
                errs.append(f"group '{name}': duplicate symbols")
            for s in g.symbols:
                if not s or any(c.isspace() for c in s):
                    errs.append(f"group '{name}': invalid symbol {s!r}")
        return errs


# ── Module helpers ────────────────────────────────────────────────────────────

_cached: Optional[AssetGroupRegistry] = None


def load_default_registry(path: Path = DEFAULT_PATH, *, reload: bool = False) -> AssetGroupRegistry:
    """Return a process-wide registry instance. Pass reload=True after mutations."""
    global _cached
    if _cached is None or reload or _cached.path != path:
        _cached = AssetGroupRegistry(path)
    return _cached
