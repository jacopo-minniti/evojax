#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import jax.nn as jnn
import jax.numpy as jnp

from evojax.algo.neat_backprop import NEATBackprop
from evojax.policy.neat import NEATPolicy
from evojax.task import BinaryClassification
from evojax.task.binary_dataset import BinaryClassificationDataset
from evojax.trainer import Trainer
from evojax.util import create_logger
from scripts.benchmarks.problems import load_yaml


def _default_output_dir(config: Dict) -> Path:
    base = config.get(
        "output_base",
        Path(f"log/{config['es_name']}/{config['problem_type']}"),
    )
    run_name = str(config.get("run_name", "default"))
    return Path(base) / run_name


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for sub in ("plots", "metrics", "checkpoints"):
        (path / sub).mkdir(exist_ok=True)


def _copy_config(config_path: Path | None, output_dir: Path) -> None:
    if config_path is None:
        return
    target = output_dir / "config.yaml"
    if config_path.resolve() != target.resolve():
        shutil.copy2(config_path, target)


def _extract_dataset_config(config: Dict) -> Dict:
    dataset_cfg = dict(config.get("problem_config", {}))
    defaults = {
        "dataset_type": "circle",
        "train_size": 200,
        "test_size": 200,
        "dataset_noise": 0.5,
        "dataset_seed": config.get("seed", 0),
        "batch_size": 32,
        "grid_range": [-6.0, 6.0],
        "grid_resolution": 200,
    }
    for key, value in defaults.items():
        dataset_cfg.setdefault(key, value)
    return dataset_cfg


def _save_metrics(history: Dict[str, List[Dict[str, float]]], output_dir: Path) -> None:
    metrics_dir = output_dir / "metrics"

    def _write_csv(entries: List[Dict[str, float]], name: str) -> None:
        if not entries:
            return
        path = metrics_dir / name
        fieldnames = list(entries[0].keys())
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)

    _write_csv(history["train"], "train_metrics.csv")
    _write_csv(history["test"], "test_metrics.csv")
    summary = {
        "train_rows": len(history["train"]),
        "test_rows": len(history["test"]),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _plot_metrics(history: Dict[str, List[Dict[str, float]]], output_dir: Path) -> None:
    if not history["train"]:
        return
    train_history = history["train"]
    train_iters = [entry["iteration"] for entry in train_history]

    fig, (ax_fit, ax_acc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_fit.plot(train_iters, [entry["max_fitness"] for entry in train_history], label="Train max fitness")
    ax_fit.plot(train_iters, [entry["mean_fitness"] for entry in train_history], label="Train mean fitness")
    ax_fit.plot(train_iters, [entry["std_fitness"] for entry in train_history], label="Train std fitness")

    if history["test"]:
        test_history = history["test"]
        test_iters = [entry["iteration"] for entry in test_history]
        ax_fit.plot(test_iters, [entry["mean_fitness"] for entry in test_history], label="Test mean fitness")

    ax_fit.set_ylabel("Fitness")
    ax_fit.legend()
    ax_fit.grid(True, linestyle="--", alpha=0.3)

    ax_acc.plot(train_iters, [entry["train_accuracy"] for entry in train_history], label="Train accuracy")
    ax_acc.plot(train_iters, [entry["test_accuracy"] for entry in train_history], label="Eval accuracy")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.grid(True, linestyle="--", alpha=0.3)

    ax_loss = ax_acc.twinx()
    ax_loss.plot(train_iters, [entry["training_loss"] for entry in train_history],
                 label="Training loss", color="tab:red", alpha=0.6)
    ax_loss.set_ylabel("Loss", color="tab:red")
    ax_loss.tick_params(axis="y", labelcolor="tab:red")

    ax_acc.set_xlabel("Iteration")
    ax_acc.legend(loc="upper left")
    ax_loss.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "training_metrics.png", dpi=200)
    plt.close(fig)


def _plot_decision_boundary(
    dataset_cfg: Dict,
    solver: NEATBackprop,
    policy: NEATPolicy,
    output_dir: Path,
) -> None:
    dataset = BinaryClassificationDataset(
        dataset_type=dataset_cfg["dataset_type"],
        train_size=int(dataset_cfg["train_size"]),
        test_size=int(dataset_cfg["test_size"]),
        noise=float(dataset_cfg["dataset_noise"]),
        seed=int(dataset_cfg["dataset_seed"]),
    )
    grid_low, grid_high = dataset_cfg["grid_range"]
    resolution = int(dataset_cfg["grid_resolution"])

    grid_points = np.linspace(grid_low, grid_high, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    params = jnp.asarray(solver.best_params, dtype=jnp.float32)
    params_batch = jnp.broadcast_to(params, (grid.shape[0], params.shape[0]))
    logits = policy._forward_fn(params_batch, jnp.asarray(grid))
    probs = np.array(jnn.sigmoid(logits).reshape(-1))

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(
        xx,
        yy,
        probs.reshape(xx.shape),
        levels=200,
        cmap="coolwarm",
        alpha=0.7,
    )
    plt.colorbar(contour, ax=ax, label="P(class=1)")
    ax.scatter(
        dataset.train_inputs[:, 0],
        dataset.train_inputs[:, 1],
        c=dataset.train_labels.reshape(-1),
        cmap="coolwarm",
        edgecolors="k",
        alpha=0.6,
        label="Train",
    )
    ax.scatter(
        dataset.test_inputs[:, 0],
        dataset.test_inputs[:, 1],
        c=dataset.test_labels.reshape(-1),
        cmap="coolwarm",
        marker="x",
        alpha=0.6,
        label="Test",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Decision boundary")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.savefig(output_dir / "plots" / "decision_boundary.png", dpi=200)
    plt.close(fig)


def _build_solver_and_policy(
    config: Dict,
    dataset_cfg: Dict,
    input_dim: int,
    output_dim: int,
    logger,
) -> Tuple[NEATBackprop, NEATPolicy]:
    es_cfg = dict(config.get("es_config", {}))
    max_hidden_nodes = int(config.get("max_hidden_nodes", es_cfg.get("max_hidden_nodes", 16)))
    propagation_steps = config.get("propagation_steps", None)

    policy = NEATPolicy(
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        max_hidden_nodes=max_hidden_nodes,
        propagation_steps=propagation_steps,
    )

    es_cfg.setdefault("pop_size", config.get("pop_size", 64))
    es_cfg.setdefault("n_input", policy.input_dim)
    es_cfg.setdefault("n_output", policy.output_dim)
    es_cfg.setdefault("max_hidden_nodes", max_hidden_nodes)
    es_cfg.setdefault("activation_choices", [1, 5, 9])
    es_cfg.setdefault("batch_size", dataset_cfg["batch_size"])
    es_cfg.setdefault("grad_steps", 4)
    es_cfg.setdefault("learning_rate", 1e-2)
    es_cfg.setdefault("dataset_type", dataset_cfg["dataset_type"])
    es_cfg.setdefault("train_size", dataset_cfg["train_size"])
    es_cfg.setdefault("test_size", dataset_cfg["test_size"])
    es_cfg.setdefault("dataset_noise", dataset_cfg["dataset_noise"])
    es_cfg.setdefault("dataset_seed", dataset_cfg["dataset_seed"])

    solver = NEATBackprop(
        param_size=policy.num_params,
        seed=config.get("seed", 0),
        logger=logger,
        **es_cfg,
    )
    return solver, policy


def _train_from_config(config: Dict, config_path: Path | None) -> Tuple[Dict, NEATBackprop, NEATPolicy, Path, Dict]:
    if config["es_name"] != "NEATBackprop":
        raise ValueError("This trainer expects es_name to be 'NEATBackprop'.")
    if config["problem_type"] != "binary_classification":
        raise ValueError("This trainer expects problem_type to be 'binary_classification'.")

    output_dir = _default_output_dir(config)
    _ensure_dir(output_dir)
    _copy_config(config_path, output_dir)

    logger = create_logger(
        name="BinaryClassification",
        log_dir=str(output_dir),
        debug=bool(config.get("debug", False)),
    )
    logger.info("EvoJAX Binary Classification with NEATBackprop")
    logger.info("=" * 40)

    dataset_cfg = _extract_dataset_config(config)

    train_task = BinaryClassification(
        batch_size=int(dataset_cfg["batch_size"]),
        test=False,
        dataset_type=dataset_cfg["dataset_type"],
        train_size=int(dataset_cfg["train_size"]),
        test_size=int(dataset_cfg["test_size"]),
        noise=float(dataset_cfg["dataset_noise"]),
        dataset_seed=int(dataset_cfg["dataset_seed"]),
    )
    test_task = BinaryClassification(
        batch_size=int(dataset_cfg["test_size"]),
        test=True,
        dataset_type=dataset_cfg["dataset_type"],
        train_size=int(dataset_cfg["train_size"]),
        test_size=int(dataset_cfg["test_size"]),
        noise=float(dataset_cfg["dataset_noise"]),
        dataset_seed=int(dataset_cfg["dataset_seed"]),
    )

    solver, policy = _build_solver_and_policy(
        config,
        dataset_cfg,
        train_task.obs_shape[0],
        train_task.act_shape[0],
        logger,
    )

    history: Dict[str, List[Dict[str, float]]] = {"train": [], "test": []}

    def log_scores(iter_idx: int, scores: jnp.ndarray, stage: str) -> None:
        scores_np = np.asarray(scores, dtype=np.float32)
        entry = {
            "iteration": int(iter_idx),
            "stage": stage,
            "max_fitness": float(scores_np.max()),
            "mean_fitness": float(scores_np.mean()),
            "min_fitness": float(scores_np.min()),
            "std_fitness": float(scores_np.std()),
            "training_loss": float(solver.training_loss) if not np.isnan(solver.training_loss) else np.nan,
            "train_accuracy": float(solver.train_accuracy) if not np.isnan(solver.train_accuracy) else np.nan,
            "test_accuracy": float(solver.test_accuracy) if not np.isnan(solver.test_accuracy) else np.nan,
        }
        history.setdefault(stage, [])
        history[stage].append(entry)

    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=int(config.get("max_iter", 200)),
        log_interval=int(config.get("log_interval", 1)),
        test_interval=int(config.get("test_interval", 20)),
        n_repeats=int(config.get("n_repeats", 1)),
        n_evaluations=int(config.get("num_tests", 32)),
        seed=int(config.get("seed", 0)),
        log_dir=str(output_dir),
        logger=logger,
        normalize_obs=bool(config.get("normalize", False)),
        log_scores_fn=log_scores,
    )

    logger.info("Starting training for %d iterations.", config.get("max_iter", 200))
    trainer.run()
    best_seen = max((entry["max_fitness"] for entry in history["train"]), default=float("-inf"))
    logger.info("Training complete. Best observed fitness %.4f", best_seen)

    return history, solver, policy, output_dir, dataset_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NEATBackprop classifier from YAML config.")
    parser.add_argument(
        "--config_fname",
        type=Path,
        default=Path("scripts/benchmarks/configs/NEAT_backprop/binary_classification.yaml"),
        help="Path to YAML configuration.",
    )
    args = parser.parse_args()
    config = load_yaml(str(args.config_fname))

    history, solver, policy, output_dir, dataset_cfg = _train_from_config(config, args.config_fname)
    _save_metrics(history, output_dir)
    _plot_metrics(history, output_dir)
    _plot_decision_boundary(dataset_cfg, solver, policy, output_dir)

    print(f"Training artifacts saved under {output_dir}")


if __name__ == "__main__":
    main()
