import numpy as np
import pandas as pd
import pytest

from cmapss_health.config import load_config
from cmapss_health.data import load_subset
from cmapss_health.features import FeatureBuilder
from cmapss_health.labeling import add_rul_test, add_rul_train, assign_stage, label_frame
from cmapss_health.pipeline import HealthPipeline, _sort
from cmapss_health.synthetic import generate_subset


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["features"]["windows"] = [5, 15]
    c["model"]["name"] = "logistic_regression"
    return c


def _data(tmp_path, subset, cfg):
    generate_subset(tmp_path, subset, n_train=25, n_test=12, seed=1)
    tr, te, rul = load_subset(tmp_path, subset)
    return (_sort(label_frame(add_rul_train(tr), cfg)),
            _sort(label_frame(add_rul_test(te, rul), cfg)))


def test_rul_train_and_stage_thresholds(cfg):
    df = add_rul_train(pd.DataFrame({"unit": [1] * 6, "cycle": range(1, 7)}))
    assert df["RUL"].tolist() == [5, 4, 3, 2, 1, 0]
    c = {"labeling": {**cfg["labeling"], "rul_thresholds": [1, 3, 4]}}
    # RUL<=1 Critical(3) | <=3 Warning(2) | <=4 Degrading(1) | >4 Healthy(0)
    assert assign_stage(df, c).tolist() == [0, 1, 2, 2, 3, 3]


def test_stage_is_monotonic_per_engine(tmp_path, cfg):
    tr, _ = _data(tmp_path, "FD001", cfg)
    for _, g in tr.groupby("unit"):
        assert (g["stage"].diff().dropna() >= 0).all()


def test_test_rul_uses_rul_file(tmp_path, cfg):
    generate_subset(tmp_path, "FD001", n_train=5, n_test=5, seed=2)
    _, te, rul = load_subset(tmp_path, "FD001")
    te = add_rul_test(te, rul)
    last = te.groupby("unit").tail(1).set_index("unit")["RUL"]
    assert (last.values == rul.values).all()


def test_features_are_causal():
    df = pd.DataFrame({"unit": [1] * 10, "cycle": range(1, 11), "a": np.arange(10.0)})
    X1 = FeatureBuilder(["a"], [3]).transform(df)
    df2 = df.copy(); df2.loc[9, "a"] = 999.0  # change the future
    X2 = FeatureBuilder(["a"], [3]).transform(df2)
    pd.testing.assert_frame_equal(X1.iloc[:9], X2.iloc[:9])


@pytest.mark.parametrize("subset", ["FD001", "FD002"])
def test_end_to_end(tmp_path, cfg, subset):
    tr, te = _data(tmp_path, subset, cfg)
    pipe = HealthPipeline(cfg).fit(tr)
    assert pipe.n_conditions_ == (6 if subset == "FD002" else 1)
    pred = pipe.predict_frame(te)
    assert len(pred) == len(te)
    assert np.allclose(pred[[f"p_{n}" for n in pipe.stage_names]].sum(axis=1), 1)
    assert (pred["pred_stage_id"] == pred["true_stage_id"]).mean() > 0.5
    assert len(pipe.predict_last(te)) == te["unit"].nunique()
    path = tmp_path / "m.joblib"; pipe.save(path)
    again = HealthPipeline.load(path).predict_frame(te)
    assert (again["pred_stage_id"].values == pred["pred_stage_id"].values).all()
