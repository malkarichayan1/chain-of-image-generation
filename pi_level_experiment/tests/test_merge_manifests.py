"""
Coverage for merge_manifests.py -- pure dict/JSON merge, no models, no GPU.
"""
import pytest

from merge_manifests import merge_manifests


def _manifest(seeds, chain_keys, img_size=512):
    return {
        "seeds": seeds,
        "img_size": img_size,
        "chains": [{"chain_key": k, "detected": True} for k in chain_keys],
    }


def test_merge_combines_chains_and_unions_seeds():
    v1 = _manifest([42, 7, 1234], ["p0_s42", "p0_s7"])
    v2 = _manifest([2024], ["p0_s2024"])

    merged = merge_manifests([v1, v2])

    assert merged["seeds"] == [7, 42, 1234, 2024]
    assert {c["chain_key"] for c in merged["chains"]} == {"p0_s42", "p0_s7", "p0_s2024"}
    assert merged["img_size"] == 512


def test_merge_rejects_duplicate_chain_keys():
    v1 = _manifest([42], ["p0_s42"])
    v2 = _manifest([42], ["p0_s42"])

    with pytest.raises(ValueError, match="duplicate chain_key"):
        merge_manifests([v1, v2])


def test_merge_rejects_mismatched_img_size():
    v1 = _manifest([42], ["p0_s42"], img_size=512)
    v2 = _manifest([2024], ["p0_s2024"], img_size=256)

    with pytest.raises(ValueError, match="img_size mismatch"):
        merge_manifests([v1, v2])


def test_merge_single_manifest_is_identity():
    v1 = _manifest([42, 7], ["p0_s42", "p0_s7"])

    merged = merge_manifests([v1])

    assert merged["seeds"] == [7, 42]
    assert len(merged["chains"]) == 2
