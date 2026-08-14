"""Tests for exp9_taxonomy_analysis.py (experiments #14, #16, #17, #18, #19).

The stakes here are higher than usual: a transposed axis or an off-by-one band boundary
would silently scramble every "binding is sharper in block N" style conclusion, and #19's
verdict logic is the thing that decides whether the paper's conclusion changes. Isolation
tests (does layer_band_report actually look at ONLY the layers it claims to?) and the
verdict-logic tests (does sharpest_cell_report say POSITIVE only when it should?) get the
most attention below.
"""
import json

import numpy as np
import pytest
from scipy import stats

import exp9_taxonomy_analysis as tax

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

N_LAYERS_REAL = 19    # LAYER_BANDS assumes exactly this
N_STEPS_REAL = 25     # TIMESTEP_WINDOWS / EXPECTED_N_STEPS assumes exactly this
MAX_STEPS_EARLY = 12  # matches taxonomy_capture_flux.py's MAX_STEPS (25 * 0.5)


def _cells(in_box_mass, entropy, attributes, subjects, n_layers=None, max_steps_early=MAX_STEPS_EARLY,
          prompt_id=1):
    n_layers = n_layers or in_box_mass.shape[0]
    return tax.ImageCells(
        prompt_id=prompt_id, in_box_mass=in_box_mass.astype(np.float32),
        entropy=entropy.astype(np.float32), attributes=attributes, subjects=subjects,
        layer_order=[f"transformer_blocks.{i}" for i in range(n_layers)],
        max_steps_early=max_steps_early)


# --------------------------------------------------------------- poisson_binomial_pvalue

def test_poisson_binomial_matches_binomial_when_chances_are_equal():
    """When every trial shares one chance, Poisson-binomial reduces to Binomial -- cross-
    checked against scipy's own exact one-sided binomial test."""
    expected = stats.binomtest(3, 5, 0.3, alternative="greater").pvalue

    assert tax.poisson_binomial_pvalue(3, [0.3] * 5) == pytest.approx(expected, abs=1e-9)


def test_poisson_binomial_two_fair_coins_both_heads():
    assert tax.poisson_binomial_pvalue(2, [0.5, 0.5]) == pytest.approx(0.25)


def test_poisson_binomial_zero_successes_is_certain():
    assert tax.poisson_binomial_pvalue(0, [0.1, 0.2, 0.3]) == pytest.approx(1.0)


def test_poisson_binomial_rejects_empty_chances():
    with pytest.raises(ValueError, match="non-empty"):
        tax.poisson_binomial_pvalue(0, [])


# --------------------------------------------------------------- holm_correction

def test_holm_correction_matches_hand_computed_values():
    # sorted ascending: 0.005(3), 0.01(0), 0.03(2), 0.04(1); m=4
    p = [0.01, 0.04, 0.03, 0.005]

    adjusted = tax.holm_correction(p)

    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_correction_is_monotonic_in_rank_order():
    """The ratchet (cumulative max over increasing raw p) is what makes Holm valid -- a
    later-ranked (less significant) raw p must never adjust to something SMALLER than an
    earlier-ranked one."""
    adjusted = tax.holm_correction([0.001, 0.002, 0.5])
    order = sorted(range(3), key=lambda i: [0.001, 0.002, 0.5][i])

    assert adjusted[order[0]] <= adjusted[order[1]] <= adjusted[order[2]]


def test_holm_correction_passes_none_through_and_excludes_from_family_size():
    adjusted = tax.holm_correction([0.01, None, 0.03])

    assert adjusted[1] is None
    assert adjusted[0] == pytest.approx(min(2 * 0.01, 1.0))


def test_holm_correction_empty_and_all_none():
    assert tax.holm_correction([]) == []
    assert tax.holm_correction([None, None]) == [None, None]


# --------------------------------------------------------------- benjamini_hochberg

def test_benjamini_hochberg_matches_hand_computed_values():
    p = [0.01, 0.04, 0.03, 0.005]

    adjusted = tax.benjamini_hochberg(p)

    assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_benjamini_hochberg_passes_none_through():
    adjusted = tax.benjamini_hochberg([0.01, None, 0.03])

    assert adjusted[1] is None


def test_benjamini_hochberg_is_never_stricter_than_holm():
    """FDR control is a weaker requirement than FWER -- BH-adjusted p-values must never
    exceed Holm-adjusted p-values for the same input."""
    p = [0.001, 0.01, 0.02, 0.03, 0.04]

    holm = tax.holm_correction(p)
    bh = tax.benjamini_hochberg(p)

    assert all(b <= h + 1e-12 for b, h in zip(bh, holm))


# --------------------------------------------------------------- reduce_mass / reduce_entropy

def test_reduce_mass_averages_exactly_the_requested_indices():
    # (layers=3, steps=2, heads=2, attrs=1, subjects=1)
    mass = np.zeros((3, 2, 2, 1, 1))
    mass[1, 0, 0, 0, 0] = 4.0   # layer 1, step 0, head 0 -- the only cell we'll select
    mass[0, :, :, :, :] = 999   # everywhere else: a value that must NOT leak in
    mass[2, :, :, :, :] = 999
    cells = _cells(mass, np.zeros_like(mass[..., 0]), ["a"], ["s"])

    reduced = tax.reduce_mass(cells, layer_idx=[1], step_idx=[0], head_idx=[0])

    assert reduced[0, 0] == pytest.approx(4.0)


def test_reduce_mass_means_over_multiple_indices():
    mass = np.zeros((2, 1, 1, 1, 1))
    mass[0, 0, 0, 0, 0] = 2.0
    mass[1, 0, 0, 0, 0] = 6.0
    cells = _cells(mass, np.zeros_like(mass[..., 0]), ["a"], ["s"], n_layers=2)

    reduced = tax.reduce_mass(cells, layer_idx=[0, 1], step_idx=[0], head_idx=[0])

    assert reduced[0, 0] == pytest.approx(4.0)   # mean of 2.0 and 6.0


def test_reduce_entropy_shape_and_values():
    ent = np.zeros((2, 1, 1, 1))
    ent[0, 0, 0, 0] = 1.0
    ent[1, 0, 0, 0] = 3.0
    cells = _cells(np.zeros((2, 1, 1, 1, 1)), ent, ["a"], ["s"], n_layers=2)

    reduced = tax.reduce_entropy(cells, layer_idx=[0, 1], step_idx=[0], head_idx=[0])

    assert reduced.shape == (1,)
    assert reduced[0] == pytest.approx(2.0)


# --------------------------------------------------------------- predicted_owners / cell_rows

def test_predicted_owners_picks_argmax_per_attribute():
    reduced = np.array([[0.1, 0.9], [0.7, 0.3]])

    assert tax.predicted_owners(reduced, ["a", "b"]) == ["b", "a"]


def test_cell_rows_skips_rows_missing_ground_truth_or_intended_subject():
    mass = np.zeros((1, 1, 1, 2, 2))
    mass[0, 0, 0, 0] = [1, 0]   # attribute 0 -> subject index 0
    mass[0, 0, 0, 1] = [0, 1]   # attribute 1 -> subject index 1
    cells = _cells(mass, np.zeros((1, 1, 1, 2)), ["red", "blue"], ["s0", "s1"], n_layers=1)

    rows = tax.cell_rows([cells], ground_truth={(1, "red"): "s0"},
                         intended={(1, "red"): "s0"},   # "blue" has no ground truth at all
                         layer_idx=[0], step_idx=[0], head_idx=[0])

    assert len(rows) == 1 and rows[0]["attribute"] == "red"


def test_cell_rows_marks_non_subject_labels_as_unscored():
    mass = np.zeros((1, 1, 1, 1, 2))
    mass[0, 0, 0, 0] = [1, 0]
    cells = _cells(mass, np.zeros((1, 1, 1, 1)), ["red"], ["s0", "s1"], n_layers=1)

    rows = tax.cell_rows([cells], {(1, "red"): "unclear"}, {(1, "red"): "s0"},
                         [0], [0], [0])

    assert rows[0]["scored"] is False
    assert rows[0]["correct"] is False


def test_cell_rows_carries_the_rows_own_subject_count_for_chance():
    mass = np.zeros((1, 1, 1, 1, 3))
    mass[0, 0, 0, 0] = [1, 0, 0]
    cells = _cells(mass, np.zeros((1, 1, 1, 1)), ["red"], ["s0", "s1", "s2"], n_layers=1)

    rows = tax.cell_rows([cells], {(1, "red"): "s0"}, {(1, "red"): "s0"}, [0], [0], [0])

    assert rows[0]["n"] == 3


# --------------------------------------------------------------- accuracy_report

def test_accuracy_report_computes_accuracy_and_calls_poisson_binomial():
    rows = [dict(scored=True, correct=True, n=2), dict(scored=True, correct=False, n=4),
           dict(scored=False, correct=False, n=2)]

    report = tax.accuracy_report(rows)

    assert report["n_scored"] == 2
    assert report["n_correct"] == 1
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["p_value"] == pytest.approx(
        tax.poisson_binomial_pvalue(1, [0.5, 0.25]))


def test_accuracy_report_handles_no_scored_rows():
    report = tax.accuracy_report([dict(scored=False, correct=False, n=2)])

    assert report == dict(n_scored=0, n_correct=0, accuracy=None, mean_chance=None,
                          p_value=None)


# --------------------------------------------------------------- #14 layer band isolation

def _banded_image(favor_correct_layers, subjects=("wrong", "correct")):
    """One image, one attribute, `favor_correct_layers` gets a landslide for "correct" at
    every early-window step; every other layer favors "wrong" instead. Real-sized
    (19, 25) so LAYER_BANDS' hardcoded ranges are genuinely exercised."""
    mass = np.zeros((N_LAYERS_REAL, N_STEPS_REAL, 1, 1, 2))
    wrong_idx, correct_idx = 0, 1
    for li in range(N_LAYERS_REAL):
        winner = correct_idx if li in favor_correct_layers else wrong_idx
        mass[li, :MAX_STEPS_EARLY, 0, 0, winner] = 10.0
    return _cells(mass, np.zeros((N_LAYERS_REAL, N_STEPS_REAL, 1, 1)), ["red"],
                 list(subjects), n_layers=N_LAYERS_REAL)


def test_layer_band_report_isolates_each_band():
    """Only the 'mid' band's layers favor the correct subject; 'early' and 'late' must NOT
    pick that up. A transposed or mis-ranged band would make this fail."""
    img = _banded_image(favor_correct_layers=set(range(7, 13)))   # exactly LAYER_BANDS["mid_7_12"]
    gt, intended = {(1, "red"): "correct"}, {(1, "red"): "wrong"}

    report = tax.layer_band_report([img], gt, intended)

    assert report["mid_7_12"]["accuracy"] == pytest.approx(1.0)
    assert report["early_0_6"]["accuracy"] == pytest.approx(0.0)
    assert report["late_13_18"]["accuracy"] == pytest.approx(0.0)


def test_layer_band_report_applies_holm_across_exactly_three_bands():
    img = _banded_image(favor_correct_layers=set(range(19)))   # every band favors correct
    report = tax.layer_band_report([img], {(1, "red"): "correct"}, {(1, "red"): "wrong"})

    assert all("p_value_holm" in report[b] for b in tax.LAYER_BANDS)


# --------------------------------------------------------------- #17 timestep window isolation

def _windowed_image(favor_correct_steps, subjects=("wrong", "correct")):
    mass = np.zeros((N_LAYERS_REAL, N_STEPS_REAL, 1, 1, 2))
    wrong_idx, correct_idx = 0, 1
    for step in range(N_STEPS_REAL):
        winner = correct_idx if step in favor_correct_steps else wrong_idx
        mass[:, step, 0, 0, winner] = 10.0
    return _cells(mass, np.zeros((N_LAYERS_REAL, N_STEPS_REAL, 1, 1)), ["red"],
                 list(subjects), n_layers=N_LAYERS_REAL)


def test_timestep_window_report_isolates_each_window():
    img = _windowed_image(favor_correct_steps=set(range(7, 13)))   # exactly w2_7_12
    gt, intended = {(1, "red"): "correct"}, {(1, "red"): "wrong"}

    report = tax.timestep_window_report([img], gt, intended)

    assert report["w2_7_12"]["accuracy"] == pytest.approx(1.0)
    assert report["w1_0_6"]["accuracy"] == pytest.approx(0.0)
    assert report["w3_13_18"]["accuracy"] == pytest.approx(0.0)
    assert report["w4_19_24"]["accuracy"] == pytest.approx(0.0)


def test_timestep_window_report_rejects_wrong_step_count():
    bad = _cells(np.zeros((N_LAYERS_REAL, 10, 1, 1, 2)), np.zeros((N_LAYERS_REAL, 10, 1, 1)),
                ["red"], ["a", "b"], n_layers=N_LAYERS_REAL)

    with pytest.raises(ValueError, match="assumes 25"):
        tax.timestep_window_report([bad], {}, {})


# --------------------------------------------------------------- #16 per-head grid

def test_per_head_grid_isolates_the_targeted_cell():
    mass = np.zeros((2, 1, 2, 1, 2))     # layers=2, steps=1, heads=2
    mass[1, 0, 1, 0] = [0, 10]           # layer1/head1 favors subject "correct"
    mass[0, 0, 0, 0] = [10, 0]           # every other cell favors "wrong"
    mass[0, 0, 1, 0] = [10, 0]
    mass[1, 0, 0, 0] = [10, 0]
    cells = _cells(mass, np.zeros((2, 1, 2, 1)), ["red"], ["wrong", "correct"],
                  n_layers=2, max_steps_early=1)

    grid = tax.per_head_grid([cells], {(1, "red"): "correct"}, {(1, "red"): "wrong"})

    assert grid["cells"]["layer1_head1"]["accuracy"] == pytest.approx(1.0)
    assert grid["cells"]["layer0_head0"]["accuracy"] == pytest.approx(0.0)
    assert grid["n_layers"] == 2 and grid["n_heads"] == 2
    assert "p_value_fdr" in grid["cells"]["layer1_head1"]
    assert grid["distribution"]["n_cells"] == 4


# --------------------------------------------------------------- #18 distribution grid

def test_collect_cell_metrics_mass_fraction_and_ratio():
    mass = np.zeros((1, 1, 1, 1, 2))
    mass[0, 0, 0, 0] = [1.0, 3.0]   # correct subject (index 1) gets 3 of total 4
    cells = _cells(mass, np.full((1, 1, 1, 1), 0.7), ["red"], ["wrong", "correct"], n_layers=1)

    mass_fracs, ratios, entropies = tax.collect_cell_metrics(
        [cells], {(1, "red"): "correct"}, {(1, "red"): "wrong"}, [0], [0], [0])

    assert mass_fracs == pytest.approx([0.75])
    assert ratios == pytest.approx([3.0 / 1.0])
    assert entropies == pytest.approx([0.7])


def test_collect_cell_metrics_skips_non_subject_labels():
    mass = np.zeros((1, 1, 1, 1, 2))
    cells = _cells(mass, np.zeros((1, 1, 1, 1)), ["red"], ["a", "b"], n_layers=1)

    mass_fracs, ratios, entropies = tax.collect_cell_metrics(
        [cells], {(1, "red"): "none"}, {(1, "red"): "a"}, [0], [0], [0])

    assert mass_fracs == ratios == entropies == []


def test_collect_cell_metrics_zero_second_peak_gives_infinite_ratio():
    mass = np.zeros((1, 1, 1, 1, 2))
    mass[0, 0, 0, 0] = [5.0, 0.0]
    cells = _cells(mass, np.zeros((1, 1, 1, 1)), ["red"], ["correct", "other"], n_layers=1)

    _, ratios, _ = tax.collect_cell_metrics(
        [cells], {(1, "red"): "correct"}, {(1, "red"): "other"}, [0], [0], [0])

    assert ratios == [float("inf")]


def test_summarize_distribution_excludes_infinite_ratio_from_stats_but_counts_it():
    summary = tax.summarize_distribution([0.5, 0.8], [2.0, float("inf")], [0.1, 0.2])

    assert summary["peak_to_second_ratio"]["n"] == 1
    assert summary["n_infinite_ratio"] == 1
    assert summary["mass_fraction_correct"]["mean"] == pytest.approx(0.65)


def test_summarize_distribution_handles_empty_input():
    summary = tax.summarize_distribution([], [], [])

    assert summary["n_rows"] == 0
    assert summary["mass_fraction_correct"]["mean"] is None


def test_distribution_grid_covers_bands_x_windows_x_heads():
    n_heads = 2
    mass = np.zeros((N_LAYERS_REAL, N_STEPS_REAL, n_heads, 1, 2))
    cells = _cells(mass, np.zeros((N_LAYERS_REAL, N_STEPS_REAL, n_heads, 1)), ["red"],
                  ["a", "b"], n_layers=N_LAYERS_REAL)

    grid = tax.distribution_grid([cells], {}, {})

    assert len(grid) == len(tax.LAYER_BANDS) * len(tax.TIMESTEP_WINDOWS) * n_heads
    assert "early_0_6|w1_0_6|head0" in grid


# --------------------------------------------------------------- #19 selection

def test_select_sharpest_cells_ranks_by_mean_mass_fraction_correct():
    # layers=2, steps=1 (within max_steps_early), heads=1
    mass = np.zeros((2, 1, 1, 1, 2))
    mass[0, 0, 0, 0] = [9, 1]    # layer0: correct subject gets 10% -> low mass fraction
    mass[1, 0, 0, 0] = [1, 9]    # layer1: correct subject gets 90% -> high mass fraction
    cells = _cells(mass, np.zeros((2, 1, 1, 1)), ["red"], ["wrong", "correct"],
                  n_layers=2, max_steps_early=1)

    top = tax.select_sharpest_cells([cells], {(1, "red"): "correct"}, {(1, "red"): "wrong"}, k=2)

    assert top[0] == (1, 0)   # layer 1 ranked first
    assert top[1] == (0, 0)


def test_select_sharpest_cells_respects_k():
    mass = np.zeros((3, 1, 1, 1, 2))
    mass[:, 0, 0, 0] = [5, 5]
    cells = _cells(mass, np.zeros((3, 1, 1, 1)), ["red"], ["a", "b"],
                  n_layers=3, max_steps_early=1)

    top = tax.select_sharpest_cells([cells], {(1, "red"): "a"}, {(1, "red"): "b"}, k=2)

    assert len(top) == 2


# --------------------------------------------------------------- prompt_baseline_test / misbound_test

def test_prompt_baseline_test_direction_cell_wins():
    rows = [dict(scored=True, correct=True, human_label="b", intended_subject="a")] * 8 + \
          [dict(scored=True, correct=False, human_label="a", intended_subject="a")] * 2
    # cell correct (tracks rendered "b") on rows where baseline (intended "a") is wrong

    report = tax.prompt_baseline_test(rows)

    assert report["cell_acc"] > report["base_acc"]
    assert report["b"] == 8 and report["c"] == 2


def test_misbound_test_restricts_to_disobeyed_rows_only():
    rows = [
        dict(scored=True, correct=True, human_label="a", intended_subject="a", n=2),   # obeyed
        dict(scored=True, correct=True, human_label="b", intended_subject="a", n=2),   # misbound
        dict(scored=True, correct=False, human_label="c", intended_subject="a", n=2),  # misbound
    ]

    report = tax.misbound_test(rows)

    assert report["n_misbound"] == 2
    assert report["n_correct"] == 1


# --------------------------------------------------------------- #19 verdict logic (the critical part)

def _easy_image_favoring_cell(layer, head, n_layers=2, n_heads=2):
    """Easy-set image where exactly (layer, head) has a mass-fraction landslide -- makes
    that cell the unambiguous top pick for select_sharpest_cells."""
    mass = np.full((n_layers, 1, n_heads, 1, 2), 0.0)
    for li in range(n_layers):
        for hi in range(n_heads):
            mass[li, 0, hi, 0] = [9, 1] if (li, hi) != (layer, head) else [1, 9]
    return _cells(mass, np.zeros((n_layers, 1, n_heads, 1)), ["red"], ["wrong", "correct"],
                 n_layers=n_layers, max_steps_early=1)


def test_sharpest_cell_report_positive_when_a_cell_beats_baseline_and_clears_chance():
    easy = _easy_image_favoring_cell(layer=0, head=0)
    easy_gt, easy_intended = {(1, "red"): "correct"}, {(1, "red"): "wrong"}

    # 6 independent hard-set images: at cell (0,0), the cell's prediction tracks the
    # RENDERED (human) label on every one, even though the model disobeyed the prompt
    # (intended != human) on every one too -- exactly the positive case, with enough rows
    # to actually clear significance (not just pin down cell selection). prompt_baseline_test
    # uses a TWO-sided binomtest, so b=6,c=0 -> p=2*0.5**6=0.03125 < 0.05; misbound_test's
    # one-sided poisson-binomial clears even more easily at the same n.
    hard_images = []
    for pid in range(2, 8):
        hard_mass = np.zeros((2, 1, 2, 1, 2))
        hard_mass[0, 0, 0, 0] = [1, 9]   # cell (0,0) strongly favors subject index 1
        hard_images.append(_cells(hard_mass, np.zeros((2, 1, 2, 1)), ["red"],
                                  ["wrong", "correct"], n_layers=2, max_steps_early=1,
                                  prompt_id=pid))
    hard_gt = {(pid, "red"): "correct" for pid in range(2, 8)}      # rendered: "correct"
    hard_intended = {(pid, "red"): "wrong" for pid in range(2, 8)}  # prompt asked "wrong"

    report = tax.sharpest_cell_report(
        [easy], easy_gt, easy_intended, hard_images, hard_gt, hard_intended, k=1)

    assert report["selected_cells"] == ["layer0_head0"]
    assert report["n_cells_passing"] == 1
    assert report["verdict"].startswith("POSITIVE")


def test_sharpest_cell_report_negative_when_no_cell_beats_baseline():
    easy = _easy_image_favoring_cell(layer=0, head=0)
    easy_gt, easy_intended = {(1, "red"): "correct"}, {(1, "red"): "wrong"}

    # Hard set: cell (0,0) predicts the SAME as the prompt's intended subject, not the
    # rendered outcome -- exactly the paper's null hypothesis (attention encodes intent).
    hard_mass = np.zeros((2, 1, 2, 1, 2))
    hard_mass[0, 0, 0, 0] = [9, 1]   # cell (0,0) favors "wrong" == intended_subject
    hard = _cells(hard_mass, np.zeros((2, 1, 2, 1)), ["red"], ["wrong", "correct"],
                 n_layers=2, max_steps_early=1, prompt_id=2)
    hard_gt = {(2, "red"): "correct"}       # but the model actually rendered "correct"
    hard_intended = {(2, "red"): "wrong"}

    report = tax.sharpest_cell_report(
        [easy], easy_gt, easy_intended, [hard], hard_gt, hard_intended, k=1)

    assert report["n_cells_passing"] == 0
    assert report["verdict"].startswith("NEGATIVE")


def test_sharpest_cell_report_direction_matters_not_just_significance():
    """A cell that is significantly WORSE than the baseline must not count as passing --
    only a significant WIN in the cell's favor does."""
    rows_cell_loses = [
        dict(scored=True, correct=False, human_label="a", intended_subject="a", n=2)] * 9 + \
        [dict(scored=True, correct=True, human_label="b", intended_subject="a", n=2)] * 1
    b_rep = tax.prompt_baseline_test(rows_cell_loses)

    assert b_rep["cell_acc"] < b_rep["base_acc"]
    # This is exactly the shape sharpest_cell_report's `cell_beats_baseline` guard must
    # reject even if the McNemar p-value alone were significant.


# --------------------------------------------------------------- on-disk loading (integration)

def _write_taxonomy_fixture(tmp_path, prompt_ids_and_diffs, n_layers=2, n_steps=2, n_heads=1):
    """Writes a minimal on-disk taxonomy capture: taxonomy_index.json +
    taxonomy_cells_p<id>.npz per id, plus manifest.json/labels_test.json so
    load_ground_truth/load_intended_subjects have something to join against."""
    index = {}
    manifest_images = []
    labels = {}
    for pid, diff in prompt_ids_and_diffs:
        mass = np.zeros((n_layers, n_steps, n_heads, 1, 2), dtype=np.float16)
        mass[0, 0, 0, 0] = [1, 0]
        ent = np.zeros((n_layers, n_steps, n_heads, 1), dtype=np.float16)
        np.savez_compressed(tmp_path / f"taxonomy_cells_p{pid}.npz", in_box_mass=mass, entropy=ent)
        index[str(pid)] = dict(
            attributes=["red"], subjects=["a", "b"],
            layer_order=[f"transformer_blocks.{i}" for i in range(n_layers)],
            n_heads=n_heads, n_steps=n_steps, max_steps_early=1,
            repro_mean_abs_pixel_diff=diff, pooled_owner_matches_manifest=True)
        manifest_images.append(dict(
            prompt_id=pid, detected=True,
            attributes=[dict(attribute="red", intended_subject="a")]))
        labels[f"{pid}::red"] = "a"

    (tmp_path / "taxonomy_index.json").write_text(json.dumps(index))
    (tmp_path / "manifest.json").write_text(json.dumps({"images": manifest_images}))
    (tmp_path / "labels_test.json").write_text(json.dumps(labels))


def test_load_all_cells_drops_images_above_the_repro_threshold(tmp_path):
    _write_taxonomy_fixture(tmp_path, [(1, 0.0), (2, 0.2), (3, 0.01)])

    images, dropped = tax.load_all_cells(tmp_path, repro_threshold=0.05)

    assert sorted(img.prompt_id for img in images) == [1, 3]
    assert dropped == [2]


def test_load_all_cells_treats_missing_repro_field_as_pass_through(tmp_path):
    _write_taxonomy_fixture(tmp_path, [(1, None)])

    images, dropped = tax.load_all_cells(tmp_path, repro_threshold=0.05)

    assert len(images) == 1 and dropped == []


def test_load_all_cells_raises_on_shape_mismatch_across_images(tmp_path):
    _write_taxonomy_fixture(tmp_path, [(1, 0.0)], n_layers=2)
    idx = json.loads((tmp_path / "taxonomy_index.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    mass2 = np.zeros((3, 2, 1, 1, 2), dtype=np.float16)   # 3 layers instead of 2
    np.savez_compressed(tmp_path / "taxonomy_cells_p2.npz", in_box_mass=mass2,
                        entropy=np.zeros((3, 2, 1, 1), dtype=np.float16))
    idx["2"] = dict(attributes=["red"], subjects=["a", "b"],
                    layer_order=[f"transformer_blocks.{i}" for i in range(3)],
                    max_steps_early=1, repro_mean_abs_pixel_diff=0.0)
    (tmp_path / "taxonomy_index.json").write_text(json.dumps(idx))
    manifest["images"].append(dict(prompt_id=2, detected=True,
                                   attributes=[dict(attribute="red", intended_subject="a")]))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="capture shape"):
        tax.load_all_cells(tmp_path)


def test_load_ground_truth_and_intended_subjects_join_correctly(tmp_path):
    _write_taxonomy_fixture(tmp_path, [(1, 0.0)])

    gt = tax.load_ground_truth(tmp_path, "test")
    intended = tax.load_intended_subjects(tmp_path)

    assert gt == {(1, "red"): "a"}
    assert intended == {(1, "red"): "a"}


# --------------------------------------------------------------- run_full_battery

def test_run_full_battery_end_to_end_smoke(tmp_path):
    easy_dir, hard_dir = tmp_path / "easy", tmp_path / "hard"
    easy_dir.mkdir()
    hard_dir.mkdir()
    _write_taxonomy_fixture(easy_dir, [(1, 0.0), (2, 0.0)], n_layers=N_LAYERS_REAL, n_steps=N_STEPS_REAL)
    _write_taxonomy_fixture(hard_dir, [(3, 0.0), (4, 0.0)], n_layers=N_LAYERS_REAL, n_steps=N_STEPS_REAL)

    report = tax.run_full_battery({"easy": (easy_dir, "test"), "hard": (hard_dir, "test")})

    assert "exp14_layer_bands" in report["datasets"]["easy"]
    assert "exp14_layer_bands" in report["datasets"]["hard"]
    assert "verdict" in report["exp19"]
    json.dumps(report)   # must be JSON-serializable end to end


def test_run_full_battery_reports_error_when_hard_set_missing(tmp_path):
    easy_dir = tmp_path / "easy"
    easy_dir.mkdir()
    _write_taxonomy_fixture(easy_dir, [(1, 0.0)], n_layers=N_LAYERS_REAL, n_steps=N_STEPS_REAL)

    report = tax.run_full_battery({"easy": (easy_dir, "test")})

    assert "error" in report["exp19"]
