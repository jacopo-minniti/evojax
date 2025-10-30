#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import matplotlib.pyplot as plt

from evojax.util import create_logger
from evojax.algo.neat_backprop import NEATBackprop
from evojax.policy.neat import NEATPolicy
from evojax.task import BinaryClassification
from evojax.task.binary_dataset import BinaryClassificationDataset
from evojax.trainer import Trainer


def parse_activation_choices(raw: str) -> Sequence[int]:
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("activation choices cannot be empty.")
    try:
        return [int(tok) for tok in tokens]
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"invalid activation choice list '{raw}'") from err


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train NEATBackprop on EvoJAX binary classification datasets."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("log/NEAT_backprop/binary_classification"),
                        help="Directory to store checkpoints, metrics, and plots.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--pop-size", type=int, default=64, help="Population size.")
    parser.add_argument("--max-hidden-nodes", type=int, default=16,
                        help="Maximum hidden nodes allowed by NEAT.")
    parser.add_argument("--max-iter", type=int, default=200, help="Number of training iterations.")
    parser.add_argument("--test-interval", type=int, default=20,
                        help="Iterations between evaluation runs.")
    parser.add_argument("--n-evaluations", type=int, default=32,
                        help="Number of evaluation rollouts for best parameters.")
    parser.add_argument("--batch-size", type=int, default=32, help="Backprop batch size.")
    parser.add_argument("--grad-steps", type=int, default=4,
                        help="Backprop gradient steps per ask call.")
    parser.add_argument("--learning-rate", type=float, default=1e-2,
                        help="Backprop optimizer step-size.")
    parser.add_argument("--dataset-type", type=str, default="circle",
                        choices=BinaryClassificationDataset.SUPPORTED,
                        help="Synthetic dataset to learn.")
    parser.add_argument("--train-size", type=int, default=200, help="Training dataset size.")
    parser.add_argument("--test-size", type=int, default=200, help="Testing dataset size.")
    parser.add_argument("--dataset-noise", type=float, default=0.5,
                        help="Noise level used when generating samples.")
    parser.add_argument("--dataset-seed", type=int, default=42,
                        help="Seed for deterministic dataset generation.")
    parser.add_argument("--activation-choices", type=parse_activation_choices, default=[1, 5, 9],
                        help="Comma-separated activation IDs available to NEAT.")
    parser.add_argument("--grid-range", type=float, nargs=2, default=(-6.0, 6.0),
                        metavar=("LOW", "HIGH"), help="Plotting range for decision boundary.")
    parser.add_argument("--grid-resolution", type=int, default=200,
                        help="Resolution for decision boundary visualization.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser


def prepare_directories(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    (path / "metrics").mkdir(exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)


def save_metrics(history: Dict[str, List[Dict[str, float]]], output_dir: Path) -> None:
    metrics_dir = output_dir / "metrics"
    train_path = metrics_dir / "train_metrics.csv"
    test_path = metrics_dir / "test_metrics.csv"

    if history["train"]:
        fieldnames = list(history["train"][0].keys())
        with train_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history["train"])

    if history["test"]:
        fieldnames = list(history["test"][0].keys())
        with test_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history["test"])

    summary = {
        "train_rows": len(history["train"]),
        "test_rows": len(history["test"]),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def plot_metrics(history: Dict[str, List[Dict[str, float]]], output_dir: Path) -> None:
    if not history["train"]:
        return
    train_history = history["train"]
    train_iters = [entry["iteration"] for entry in train_history]
    train_max = [entry["max_fitness"] for entry in train_history]
    train_mean = [entry["mean_fitness"] for entry in train_history]
    train_std = [entry["std_fitness"] for entry in train_history]
    train_loss = [entry["training_loss"] for entry in train_history]
    train_acc = [entry["train_accuracy"] for entry in train_history]
    test_acc = [entry["test_accuracy"] for entry in train_history]

    fig, (ax_fit, ax_acc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_fit.plot(train_iters, train_max, label="Train max fitness")
    ax_fit.plot(train_iters, train_mean, label="Train mean fitness")
    ax_fit.plot(train_iters, train_std, label="Train std fitness")

    if history["test"]:
        test_history = history["test"]
        test_iters = [entry["iteration"] for entry in test_history]
        test_mean = [entry["mean_fitness"] for entry in test_history]
        ax_fit.plot(test_iters, test_mean, label="Test mean fitness")

    ax_fit.set_ylabel("Fitness")
    ax_fit.legend()
    ax_fit.grid(True, linestyle="--", alpha=0.3)

    ax_acc.plot(train_iters, train_acc, label="Train accuracy")
    ax_acc.plot(train_iters, test_acc, label="Eval accuracy")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.grid(True, linestyle="--", alpha=0.3)

    ax_loss = ax_acc.twinx()
    ax_loss.plot(train_iters, train_loss, label="Training loss", color="tab:red", alpha=0.6)
    ax_loss.set_ylabel("Loss", color="tab:red")
    ax_loss.tick_params(axis="y", labelcolor="tab:red")

    ax_acc.set_xlabel("Iteration")
    ax_acc.legend(loc="upper left")
    ax_loss.legend(loc="upper right")

    fig.tight_layout()
    plot_path = output_dir / "plots" / "training_metrics.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def plot_decision_boundary(
    args: argparse.Namespace,
    solver: NEATBackprop,
    policy: NEATPolicy,
    output_dir: Path,
) -> None:
    dataset = BinaryClassificationDataset(
        dataset_type=args.dataset_type,
        train_size=args.train_size,
        test_size=args.test_size,
        noise=args.dataset_noise,
        seed=args.dataset_seed,
    )

    grid_low, grid_high = args.grid_range
    grid_points = np.linspace(grid_low, grid_high, args.grid_resolution, dtype=np.float32)
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

    plot_path = output_dir / "plots" / "decision_boundary.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def run_training(
    args: argparse.Namespace,
) -> Tuple[Dict[str, List[Dict[str, float]]], NEATBackprop, NEATPolicy]:
    prepare_directories(args.output_dir)
    logger = create_logger("NEATBackpropRunner", log_dir=str(args.output_dir), debug=args.debug)

    train_task = BinaryClassification(
        batch_size=args.batch_size,
        test=False,
        dataset_type=args.dataset_type,
        train_size=args.train_size,
        test_size=args.test_size,
        noise=args.dataset_noise,
        dataset_seed=args.dataset_seed,
    )
    test_task = BinaryClassification(
        batch_size=args.test_size,
        test=True,
        dataset_type=args.dataset_type,
        train_size=args.train_size,
        test_size=args.test_size,
        noise=args.dataset_noise,
        dataset_seed=args.dataset_seed,
    )

    policy = NEATPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        max_hidden_nodes=args.max_hidden_nodes,
        propagation_steps=None,
    )

    solver = NEATBackprop(
        param_size=policy.num_params,
        pop_size=args.pop_size,
        seed=args.seed,
        n_input=policy.input_dim,
        n_output=policy.output_dim,
        max_hidden_nodes=args.max_hidden_nodes,
        activation_choices=args.activation_choices,
        dataset_type=args.dataset_type,
        train_size=args.train_size,
        test_size=args.test_size,
        dataset_noise=args.dataset_noise,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_steps=args.grad_steps,
        dataset_seed=args.dataset_seed,
        logger=logger,
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
        history[stage].append(entry)

    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=args.max_iter,
        log_interval=1,
        test_interval=args.test_interval,
        n_repeats=1,
        test_n_repeats=1,
        n_evaluations=args.n_evaluations,
        seed=args.seed,
        debug=args.debug,
        normalize_obs=False,
        model_dir=str(args.output_dir / "checkpoints"),
        log_dir=str(args.output_dir),
        logger=logger,
        log_scores_fn=log_scores,
    )

    logger.info("Starting training for %d iterations.", args.max_iter)
    trainer.run()
    best_seen = max((entry["max_fitness"] for entry in history["train"]), default=float("nan"))
    logger.info("Training complete. Best observed fitness %.4f", best_seen)

    return history, solver, policy


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    history, solver, policy = run_training(args)
    save_metrics(history, args.output_dir)
    plot_metrics(history, args.output_dir)
    plot_decision_boundary(args, solver, policy, args.output_dir)

    print(f"Artifacts saved under {args.output_dir}")


if __name__ == "__main__":
    main()
