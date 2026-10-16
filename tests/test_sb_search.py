"""recipes/moabb/sb_search.py: cells and packs, seeds per trial, the objective's standard error and selection."""

import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("moabb")
pytest.importorskip("optuna")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipes" / "moabb"))
import sb_protocol as protocol  # noqa: E402
import sb_search  # noqa: E402
from train import packs  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)


def fake_data(n_subjects=3, sessions=("0train", "1test"), n_runs=6, per_run=8):
    rng = np.random.default_rng(0)
    n = n_subjects * len(sessions) * n_runs * per_run
    subjects = np.repeat(np.arange(1, n_subjects + 1), n // n_subjects)
    session = np.tile(np.repeat(sessions, n_runs * per_run), n_subjects)
    runs = np.tile(np.repeat([str(r) for r in range(n_runs)], per_run), n_subjects * len(sessions))
    y = rng.integers(0, 4, n)
    folds = protocol.leave_one_session_out(y, subjects, session, 0.2)
    return dict(folds=folds, y_np=y, sessions=session, runs=runs)


def test_default_cells_and_packs_are_speechbrain_moabbs():
    """With the default protocol, the cells and packs are the (seed, fold) runs and packs sb_search made before."""
    data = fake_data()
    seeds = [1986, 1987]
    cells = sb_search.make_cells(data, seeds, sb_search.PROTOCOL)
    runs = [(name, seed, train, test, valid) for seed in seeds for name, train, valid, test in data["folds"]]
    old = [[r for r in pack if len(r[4]) == n_valid] for pack in packs(runs, 4)
           for n_valid in dict.fromkeys(len(r[4]) for r in pack)]
    new = list(sb_search.pack_cells(cells, 4))
    assert [[(c["fold"], c["seed"]) for c in pack] for pack in new] == [[(r[0], r[1]) for r in pack] for pack in old]
    for pack_new, pack_old in zip(new, old):
        for c, r in zip(pack_new, pack_old):
            assert c["inner"] == 0
            assert c["train"].tolist() == list(r[2]) and c["valid"].tolist() == list(r[4])
            assert c["test"].tolist() == list(r[3])


def test_cells_with_inner_folds():
    data = fake_data()
    settings = dict(sb_search.PROTOCOL, inner_folds=3, group_by_run=True)
    cells = sb_search.make_cells(data, [1, 2], settings)
    assert len(cells) == 2 * len(data["folds"]) * 3
    assert len({(c["fold"], c["inner"], c["seed"]) for c in cells}) == len(cells)
    for c in cells:
        name, train, valid, test = next(f for f in data["folds"] if f[0] == c["fold"])
        pool = np.union1d(train, valid)
        assert sorted(c["train"].tolist() + c["valid"].tolist()) == pool.tolist()
        assert len(set(data["runs"][c["valid"].numpy()])) == 2  # two of the six runs
    assert all(len({len(c["train"]) for c in pack}) == 1 for pack in sb_search.pack_cells(cells, 10))
    per_fold = sb_search.make_cells(data, [1, 2], dict(settings, seed_per_inner_fold=True))
    assert [c["seed"] for c in per_fold if c["inner"] == 0] == [c["seed"] for c in cells if c["inner"] == 0]
    seeds = {(c["seed"], c["inner"]) for c in per_fold}
    assert len({s for s, _ in seeds}) == 6 and len(seeds) == 6  # 2 seeds x 3 inner folds, each its own
    final = sb_search.make_cells(data, [1], dict(settings, final_train="all"), final=True)
    assert len(final) == len(data["folds"]) and all(c["valid"] is None for c in final)
    assert len(final[0]["train"]) == len(np.union1d(data["folds"][0][1], data["folds"][0][2]))


def test_trial_seeds():
    assert sb_search.trial_seeds(1986, 1, 7, False) == [1986]
    assert sb_search.trial_seeds(1986, 3, 7, False) == [1986, 1987, 1988]
    a, b = sb_search.trial_seeds(1986, 3, 7, True), sb_search.trial_seeds(1986, 3, 8, True)
    assert a == sb_search.trial_seeds(1986, 3, 7, True) and len(set(a + b)) == 6
    assert a != sb_search.trial_seeds(1, 3, 7, True)


def test_summarize():
    rows = [dict(seed=s, inner=j, fold=f, valid_acc=0.5 + 0.1 * s + 0.01 * j + 0.001 * f)
            for s in range(2) for j in range(3) for f in range(4)]
    out = sb_search.summarize(rows, "acc")
    replicates = [0.5 + 0.1 * s + 0.01 * j + 0.0015 for s in range(2) for j in range(3)]
    assert out["se"] == pytest.approx(np.std(replicates, ddof=1) / math.sqrt(6))
    assert out["per_seed"] == pytest.approx({0: 0.5 + 0.01 + 0.0015, 1: 0.6 + 0.01 + 0.0015})
    assert out["per_inner"][2] == pytest.approx(0.55 + 0.02 + 0.0015)
    assert math.isnan(sb_search.summarize([dict(seed=0, inner=0, fold=0, valid_acc=0.5)], "acc")["se"])


def test_select():
    trial = lambda number, value, se, params: SimpleNamespace(number=number, values=[value],
                                                              user_attrs=dict(se=se, params=params, flops=2 * params))
    trials = [trial(0, 0.80, 0.02, 100_000), trial(1, 0.79, 0.01, 10_000), trial(2, 0.70, 0.01, 1_000),
              trial(3, 0.785, 0.01, 20_000)]
    assert sb_search.select(trials, "best", "params")[0].number == 0
    assert sb_search.select(trials, "one_se", "params")[0].number == 1
    assert sb_search.select(trials, "one_se", "flops")[0].number == 1
    trials[0].user_attrs["se"] = float("nan")  # one seed and one validation split: no standard error
    assert sb_search.select(trials, "one_se", "params")[0].number == 0


def test_context_before(tmp_path):
    """A step's starting values rebuilt from best-*.yaml, for studies run before trials recorded them."""
    cfg = dict(fixed=dict(lr=0.1, fmin=1.0, dropout=0.5),
               steps=dict(step0=dict(search={}, epochs="grid"), step1=dict(search={}, epochs="grid"),
                          step2=dict(search={}, augment="eeg", fixed=dict(dropout=0.0))))
    steps = sb_search.steps_of(cfg)
    (tmp_path / "best-step0.yaml").write_text("trial: 3\nvalid: 0.5\nepoch: 300\nvalues: {fmin: 2.0}\n")
    (tmp_path / "best-step1.yaml").write_text("trial: 4\nvalid: 0.6\nepoch: 500\nvalues: {lr: 0.01}\n")
    hp = sb_search.context_before(cfg, steps, "step1", tmp_path, "schedule-free")
    assert hp == dict(lr=0.1, fmin=2.0, dropout=0.5, augment="none")  # step 1 chooses its epoch again
    hp = sb_search.context_before(cfg, steps, "step2", tmp_path, "schedule-free")
    assert hp == dict(lr=0.01, fmin=2.0, dropout=0.0, augment="eeg", chosen_epochs=500)


def test_steps_of():
    cfg = dict(search=dict(step1=dict(lr="choices([0.1])"), step2=dict(amp_delta="uniform(0, 1)")))
    steps = sb_search.steps_of(cfg)
    assert list(steps) == ["step1", "step2"]
    assert [s["augment"] for s in steps.values()] == ["none", "speechbrain"]
    assert steps["step1"]["search"] is cfg["search"]["step1"]  # --set-search edits them
    cfg = dict(steps=dict(step0=dict(search={}, fixed=dict(lr=0.001)), step1=dict(search={}, augment="eeg")))
    steps = sb_search.steps_of(cfg)
    assert steps["step0"]["augment"] == "none" and steps["step1"]["fixed"] == {}
