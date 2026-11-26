"""Metrics and plots."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)


def classification_metrics(y_true, y_pred, names) -> dict:
    labels = list(range(len(names)))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        # stages are ordered, so being off by one stage is less bad than off by three
        "adjacent_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mean_stage_error": float(np.mean(np.abs(y_true - y_pred))),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion_matrix(cm, names, path, title="Confusion matrix"):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_stage_distribution(df: pd.DataFrame, names, path, title="Class distribution (training rows)"):
    counts = df["stage"].value_counts().reindex(range(len(names))).fillna(0)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(names, counts.values, color=["#4caf50", "#cddc39", "#ff9800", "#f44336"][: len(names)])
    ax.set_ylabel("rows"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_feature_importance(imp: pd.Series, path, top: int = 20):
    imp = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.barh(imp.index, imp.values, color="#1976d2")
    ax.set_title(f"Top {len(imp)} feature importances")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_sensor_trends(df: pd.DataFrame, sensors, path, n_engines: int = 5):
    units = df["unit"].unique()[:n_engines]
    sensors = list(sensors)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=False)
    for ax, s in zip(axes.ravel(), sensors):
        for u in units:
            d = df[df["unit"] == u]
            ax.plot(d["cycle"].max() - d["cycle"], d[s], lw=0.8, label=f"engine {u}")
        ax.set_title(s); ax.invert_xaxis(); ax.set_xlabel("cycles to failure")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Normalised sensor trends vs. cycles-to-failure (training engines)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_engine_timeline(pred: pd.DataFrame, unit: int, names, path):
    d = pred[pred["unit"] == unit]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cols = [f"p_{n}" for n in names]
    ax.stackplot(d["cycle"], [d[c] for c in cols], labels=names,
                 colors=["#4caf50", "#cddc39", "#ff9800", "#f44336"][: len(names)], alpha=0.85)
    if "true_stage_id" in d:
        ax.step(d["cycle"], d["true_stage_id"] / max(len(names) - 1, 1), where="post",
                color="black", lw=1.2, label="true stage (scaled)")
    ax.set_xlabel("cycle"); ax.set_ylabel("stage probability"); ax.set_ylim(0, 1)
    ax.set_title(f"Engine {unit}: predicted health-stage probabilities over time")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
