"""Unit tests for core.asset_groups registry."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from core.asset_groups import AssetGroup, AssetGroupRegistry


@pytest.fixture
def sample_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "asset_groups.yaml"
    p.write_text(
        textwrap.dedent(
            """
            version: 1
            default: stocks
            groups:
              stocks:
                description: US equities
                asset_class: equity
                tags: [us, liquid]
                symbols: [SPY, QQQ, AAPL]
              crypto:
                description: Crypto pairs
                asset_class: crypto
                tags: [24x7]
                symbols: [BTC/USD, ETH/USD]
            """
        ).strip(),
        encoding="utf-8",
    )
    return p


def test_load_and_list(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    assert reg.list() == ["stocks", "crypto"]
    assert reg.default() == "stocks"
    assert reg.has("crypto")
    assert not reg.has("forex")


def test_get_returns_immutable(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    g = reg.get("stocks")
    assert g.symbols == ("SPY", "QQQ", "AAPL")
    assert g.asset_class == "equity"
    assert "us" in g.tags
    with pytest.raises(Exception):
        g.symbols = ("XYZ",)  # frozen dataclass


def test_get_missing_raises(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    with pytest.raises(KeyError):
        reg.get("unknown")


def test_add_and_persist(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    reg.add(AssetGroup(name="tech", symbols=("NVDA", "AMD"), description="Semis"))
    reg2 = AssetGroupRegistry(sample_yaml)  # reload from disk
    assert reg2.has("tech")
    assert reg2.get("tech").symbols == ("NVDA", "AMD")


def test_add_duplicate_requires_overwrite(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    with pytest.raises(ValueError):
        reg.add(AssetGroup(name="stocks", symbols=("X",)))
    reg.add(AssetGroup(name="stocks", symbols=("X",)), overwrite=True)
    assert reg.get("stocks").symbols == ("X",)


def test_remove_and_default_fallback(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    reg.remove("stocks")
    assert not reg.has("stocks")
    # default was 'stocks' → fallback to remaining group
    assert reg.default() == "crypto"


def test_update_add_remove_symbols(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    reg.update("stocks", add_symbols=["NVDA"], remove_symbols=["AAPL"])
    g = reg.get("stocks")
    assert "NVDA" in g.symbols
    assert "AAPL" not in g.symbols
    assert "SPY" in g.symbols  # unchanged


def test_update_replace_all_symbols(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    reg.update("stocks", symbols=["A", "B"])
    assert reg.get("stocks").symbols == ("A", "B")


def test_rename(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    reg.rename("stocks", "us_stocks")
    assert not reg.has("stocks")
    assert reg.has("us_stocks")
    assert reg.default() == "us_stocks"


def test_rename_collision(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    with pytest.raises(ValueError):
        reg.rename("stocks", "crypto")


def test_set_default(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    reg.set_default("crypto")
    assert reg.default() == "crypto"
    with pytest.raises(KeyError):
        reg.set_default("nope")


def test_validate_clean(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    assert reg.validate() == []


def test_validate_detects_duplicates(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "version: 1\ndefault: x\ngroups:\n  x:\n    symbols: [A, A, B]\n",
        encoding="utf-8",
    )
    reg = AssetGroupRegistry(p)
    errs = reg.validate()
    assert any("duplicate" in e for e in errs)


def test_validate_detects_empty(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("version: 1\ndefault: x\ngroups:\n  x:\n    symbols: []\n", encoding="utf-8")
    errs = AssetGroupRegistry(p).validate()
    assert any("empty" in e for e in errs)


def test_filter_by_tag(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    assert [g.name for g in reg.filter(tag="24x7")] == ["crypto"]
    assert [g.name for g in reg.filter(asset_class="equity")] == ["stocks"]


def test_save_creates_backup(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    reg.add(AssetGroup(name="new", symbols=("X",)))
    assert sample_yaml.with_suffix(".yaml.bak").exists()


def test_missing_file_is_empty(tmp_path):
    reg = AssetGroupRegistry(tmp_path / "nope.yaml")
    assert reg.list() == []
    assert reg.default() == ""


def test_atomic_write_no_tmp_leftover(sample_yaml):
    reg = AssetGroupRegistry(sample_yaml)
    reg.add(AssetGroup(name="new", symbols=("X",)))
    assert not sample_yaml.with_suffix(".yaml.tmp").exists()
