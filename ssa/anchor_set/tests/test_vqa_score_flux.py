"""Local-run support for vqa_score_flux.py (experiment #31).

The kernel was written Kaggle-only: `find_input_dir` searched `/kaggle/input` and nothing
else, so the whole experiment was gated behind a GPU queue it never needed --
Salesforce/blip-vqa-base is ~385M parameters and scores an anchor set on CPU in minutes.
These tests cover the local-directory path and the resume behaviour the module docstring
already promised ("saved incrementally ... so a Kaggle timeout doesn't lose completed
work") but did not actually implement: the previous main() started from an empty dict every
run, so a timeout lost everything anyway.

The model-calling functions (load_model, p_yes) stay untested here, same split as
recompute_boxes.py / exp10_clipscore_discriminant.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from vqa_score_flux import find_input_dir, load_existing_results, pending_vqa_images


# --------------------------------------------------------------------------- find_input_dir

def test_find_input_dir_returns_explicit_local_dir_containing_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    assert find_input_dir(local_dir=tmp_path) == tmp_path


def test_find_input_dir_raises_when_local_dir_has_no_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        find_input_dir(local_dir=tmp_path)


def test_find_input_dir_raises_when_local_dir_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_input_dir(local_dir=tmp_path / "nope")


# --------------------------------------------------------------------------- resume

def test_load_existing_results_returns_empty_dict_when_absent(tmp_path):
    assert load_existing_results(tmp_path / "vqa_scores.json") == {}


def test_load_existing_results_reads_prior_run(tmp_path):
    out = tmp_path / "vqa_scores.json"
    out.write_text(json.dumps({"7": {"red apron": {"barista": 0.9}}}))
    assert load_existing_results(out) == {"7": {"red apron": {"barista": 0.9}}}


def _img(prompt_id, detected=True):
    return {"prompt_id": prompt_id, "detected": detected,
            "attributes": [{"attribute": "red apron", "intended_subject": "barista"}]}


def test_pending_vqa_images_skips_undetected_and_boxless_and_cached():
    manifest = {"images": [_img(1), _img(2), _img(3, detected=False), _img(4)]}
    boxes = {"1": {"barista": [0, 0, 10, 10]}, "2": {"barista": [0, 0, 10, 10]},
             "3": {"barista": [0, 0, 10, 10]}}
    results = {"1": {"red apron": {"barista": 0.9}}}
    # 1 is cached, 3 is undetected, 4 has no box entry -> only 2 remains
    assert [i["prompt_id"] for i in pending_vqa_images(manifest, boxes, results)] == [2]


def test_pending_vqa_images_returns_all_on_a_fresh_run():
    manifest = {"images": [_img(1), _img(2)]}
    boxes = {"1": {"barista": [0, 0, 1, 1]}, "2": {"barista": [0, 0, 1, 1]}}
    assert [i["prompt_id"] for i in pending_vqa_images(manifest, boxes, {})] == [1, 2]
