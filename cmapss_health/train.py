"""Training / evaluation entry point.

    python -m cmapss_health.train --subset FD001
    python -m cmapss_health.train --subset FD002 --compare
    python -m cmapss_health.train --subset FD001 --model random_forest --stages 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import PROJECT_ROOT, load_config
from .data import SENSOR_COLS, load_subset
from .evaluate import (classification_metrics, plot_confusion_matrix, plot_engine_timeline,
                       plot_feature_importance, plot_sensor_trends, plot_stage_distribution)
from .labeling import add_rul_test, add_rul_train, label_frame
from .pipeline import HealthPipeline, _sort


def prepare_data(cfg):
    data_dir = Path(cfg["data"]["dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    train, test, rul = load_subset(data_dir, cfg["data"]["subset"])
    train = label_frame(add_rul_train(train), cfg)
    test = label_frame(add_rul_test(test, rul), cfg)
    return _sort(train), _sort(test)


def cross_validate(cfg, train, model_names):
    """GroupKFold by engine: cycles of one engine never appear in both train and validation."""
    gkf = GroupKFold(n_splits=cfg["cv"]["n_splits"])
    names = cfg["labeling"]["stage_names"]
    rows = []
    for m in model_names:
        t0 = time.time()
        scores = []
        for tr_idx, va_idx in gkf.split(train, groups=train["unit"]):
            pipe = HealthPipeline(cfg, m).fit(train.iloc[tr_idx])
            pred = pipe.predict_frame(train.iloc[va_idx])
            met = classification_metrics(pred["true_stage_id"], pred["pred_stage_id"], names)
            scores.append([met["accuracy"], met["macro_f1"], met["adjacent_accuracy"]])
        s = pd.DataFrame(scores, columns=["accuracy", "macro_f1", "adjacent_accuracy"])
        row = {"model": m, "seconds": round(time.time() - t0, 1)}
        for c in s:
            row[f"{c}_mean"], row[f"{c}_std"] = s[c].mean(), s[c].std()
        rows.append(row)
        print(f"  CV {m:24s} macro-F1 {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} "
              f"| acc {row['accuracy_mean']:.4f} | {row['seconds']}s")
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--subset", default=None, choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--compare", action="store_true", help="cross-validate candidate models first")
    ap.add_argument("--models", nargs="+", default=None, help="candidate models for --compare")
    ap.add_argument("--folds", type=int, default=None, help="CV folds for --compare")
    ap.add_argument("--synthetic", action="store_true", help="generate & use synthetic demo data")
    ap.add_argument("--out", default=None, help="output model path")
    a = ap.parse_args(argv)

    ov = {}
    if a.data_dir: ov.setdefault("data", {})["dir"] = a.data_dir
    if a.subset: ov.setdefault("data", {})["subset"] = a.subset
    if a.model: ov.setdefault("model", {})["name"] = a.model
    if a.models: ov.setdefault("model", {})["candidates"] = a.models
    if a.folds: ov.setdefault("cv", {})["n_splits"] = a.folds
    cfg = load_config(a.config, ov)
    if a.synthetic:
        from .synthetic import generate_subset
        d = PROJECT_ROOT / "data" / "synthetic"
        generate_subset(d, cfg["data"]["subset"], seed=cfg["seed"])
        cfg["data"]["dir"] = str(d)

    subset, names = cfg["data"]["subset"], cfg["labeling"]["stage_names"]
    rep = Path(cfg["output"]["reports_dir"]); rep = rep if rep.is_absolute() else PROJECT_ROOT / rep
    fig = rep / "figures"; fig.mkdir(parents=True, exist_ok=True)
    tag = subset + ("_synthetic" if a.synthetic else "")

    print(f"[1/5] Loading {subset} ...")
    train, test = prepare_data(cfg)
    print(f"      train: {len(train)} rows / {train['unit'].nunique()} engines | "
          f"test: {len(test)} rows / {test['unit'].nunique()} engines")
    dist = train["stage"].value_counts(normalize=True).sort_index()
    print("      train class balance:", {names[i]: f"{v:.1%}" for i, v in dist.items()})
    plot_stage_distribution(train, names, fig / f"{tag}_class_distribution.png")

    model_name = cfg["model"]["name"]
    if a.compare:
        print(f"[2/5] Cross-validating {cfg['model']['candidates']} ({cfg['cv']['n_splits']}-fold, grouped by engine)")
        comp = cross_validate(cfg, train, cfg["model"]["candidates"])
        comp.to_csv(rep / f"{tag}_model_comparison.csv", index=False)
        model_name = comp.iloc[0]["model"]
        print(f"      best model: {model_name}")
    else:
        print(f"[2/5] Skipping model comparison (use --compare). Model: {model_name}")

    print(f"[3/5] Fitting final model '{model_name}' on all training engines ...")
    pipe = HealthPipeline(cfg, model_name).fit(train)
    print(f"      operating conditions: {pipe.n_conditions_} | sensors kept: {len(pipe.scaler_.keep_)} | "
          f"features: {len(pipe.feature_names_)}")

    print("[4/5] Evaluating on the official test set ...")
    pred = pipe.predict_frame(test)
    last = pred.groupby("unit").tail(1)
    m_last = classification_metrics(last["true_stage_id"], last["pred_stage_id"], names)
    m_all = classification_metrics(pred["true_stage_id"], pred["pred_stage_id"], names)
    for lab, m in [("last cycle / engine (official protocol)", m_last), ("all test cycles", m_all)]:
        print(f"      {lab:40s} acc {m['accuracy']:.3f} | macro-F1 {m['macro_f1']:.3f} | "
              f"adjacent-acc {m['adjacent_accuracy']:.3f} (n={m['n_samples']})")

    print("[5/5] Saving model, metrics and figures ...")
    out = Path(a.out or cfg["output"]["model_path"].format(subset=tag))
    out = out if out.is_absolute() else PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    pipe.save(out)
    metrics = {"subset": subset, "model": model_name, "stage_names": names,
               "labeling": cfg["labeling"], "test_last_cycle": m_last, "test_all_cycles": m_all}
    (rep / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    pred.to_csv(rep / f"{tag}_test_predictions.csv", index=False)
    plot_confusion_matrix(m_last["confusion_matrix"], names, fig / f"{tag}_cm_last_cycle.png",
                          f"{subset} - last cycle per test engine")
    plot_confusion_matrix(m_all["confusion_matrix"], names, fig / f"{tag}_cm_all_cycles.png",
                          f"{subset} - all test cycles")
    imp = pipe.feature_importance(train)
    if imp is not None:
        plot_feature_importance(imp, fig / f"{tag}_feature_importance.png")
        imp.head(30).to_csv(rep / f"{tag}_feature_importance.csv", header=["importance"])
    plot_sensor_trends(pipe.scaler_.transform(train), pipe.scaler_.keep_, fig / f"{tag}_sensor_trends.png")
    plot_engine_timeline(pred, int(test["unit"].iloc[0]), names, fig / f"{tag}_engine_timeline.png")
    print(f"      model  -> {out}\n      report -> {rep}")
    return metrics


if __name__ == "__main__":
    main()
