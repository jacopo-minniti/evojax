#!/usr/bin/env python
"""
Visualize a trained SlimeVolley policy by creating a GIF replay.

Example:
    python scripts/visualize_slimevolley.py \
        --model-path log/NEAT/slimevolley/default/best.npz
"""

import argparse
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from evojax.obs_norm import ObsNormalizer
from scripts.benchmarks.problems import load_yaml, setup_problem
from evojax.task.slimevolley import SlimeVolley


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a trained SlimeVolley NEAT policy to a GIF."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model .npz file (e.g. .../best.npz).",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional path to the YAML config (defaults to <model_dir>/config.yaml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output GIF path (defaults to <model_dir>/slimevolley.gif).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of rollout steps (defaults to environment max).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for the rollout.",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=40,
        help="GIF frame duration in milliseconds.",
    )
    return parser.parse_args()


def _load_model(model_path: Path) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    with np.load(model_path) as data:
        params = data["params"]
        obs_params = data["obs_params"] if "obs_params" in data.files else None
    params = jnp.asarray(params).reshape(1, -1)
    if obs_params is None:
        return params, None
    obs_params = np.asarray(obs_params)
    if obs_params.ndim == 0 and obs_params.item() is None:
        return params, None
    return params, jnp.asarray(obs_params).reshape(-1)


def _select_state(state, index: int = 0):
    """Extract a single environment from a batched task state."""
    return jax.tree_map(lambda x: np.asarray(jax.device_get(x[index])), state)


def main() -> None:
    args = _parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    config_path = Path(args.config).expanduser().resolve() if args.config else model_path.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Provide --config explicitly if it lives elsewhere."
        )

    output_path = Path(args.output).expanduser().resolve() if args.output else model_path.parent / "slimevolley.gif"

    config = load_yaml(str(config_path))
    _, test_task, policy = setup_problem(config, logger=None)

    params, obs_params = _load_model(model_path)

    obs_normalizer = ObsNormalizer(
        obs_shape=test_task.obs_shape,
        dummy=not bool(config.get("normalize", False)),
    )
    if obs_params is None:
        obs_params = obs_normalizer.get_init_params()
    else:
        obs_params = jnp.asarray(obs_params)

    rollout_steps = args.max_steps or getattr(test_task, "max_steps", 3000)

    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)

    rng_key = jax.random.PRNGKey(args.seed)
    task_state = task_reset_fn(jax.random.split(rng_key, 1))
    policy_state = policy_reset_fn(task_state)

    frames: list[Image.Image] = []
    frames.append(SlimeVolley.render(_select_state(task_state)))

    for _ in range(rollout_steps):
        normalized_obs = obs_normalizer.normalize_obs(task_state.obs, obs_params)
        normed_state = task_state.replace(obs=normalized_obs)
        actions, policy_state = action_fn(normed_state, params, policy_state)
        task_state, _, done = step_fn(task_state, actions)
        frames.append(SlimeVolley.render(_select_state(task_state)))
        if bool(jax.device_get(done[0])):
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.frame_duration,
        loop=0,
    )
    print(f"Saved GIF to {output_path}")


if __name__ == "__main__":
    main()
