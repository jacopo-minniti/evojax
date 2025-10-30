from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import random
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask
from evojax.task.binary_dataset import BinaryClassificationDataset


@dataclass
class BinaryState(TaskState):
    obs: jnp.ndarray
    label: jnp.float32
    samples: jnp.ndarray
    labels: jnp.ndarray
    step: jnp.int32


def _sample_batch(
    key: jnp.ndarray,
    data: jnp.ndarray,
    labels: jnp.ndarray,
    batch_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a minibatch using JAX primitives."""
    idx = random.choice(key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return jnp.take(data, idx, axis=0), jnp.take(labels, idx, axis=0)


class BinaryClassification(VectorizedTask):
    """Binary classification playground task producing accuracy rewards."""

    def __init__(
        self,
        batch_size: int = 32,
        test: bool = False,
        dataset_type: str = "circle",
        train_size: int = 200,
        test_size: int = 200,
        noise: float = 0.5,
        dataset_seed: int = 0,
        dataset: Optional[BinaryClassificationDataset] = None,
    ):
        dataset_type = dataset_type.lower()
        if dataset is None:
            dataset = BinaryClassificationDataset(
                dataset_type=dataset_type,
                train_size=train_size,
                test_size=test_size,
                noise=noise,
                seed=dataset_seed,
            )
        else:
            if dataset.dataset_type != dataset_type:
                raise ValueError(
                    f"Provided dataset_type '{dataset_type}' does not match "
                    f"dataset.dataset_type '{dataset.dataset_type}'."
                )
            if dataset.train_size != train_size or dataset.test_size != test_size:
                raise ValueError(
                    "Provided dataset does not match requested train/test sizes."
                )
        self._dataset = dataset
        self._train_inputs = jnp.asarray(self._dataset.train_inputs, dtype=jnp.float32)
        self._train_labels = jnp.asarray(
            self._dataset.train_labels.reshape(-1), dtype=jnp.float32
        )
        self._test_inputs = jnp.asarray(self._dataset.test_inputs, dtype=jnp.float32)
        self._test_labels = jnp.asarray(
            self._dataset.test_labels.reshape(-1), dtype=jnp.float32
        )

        self._test = bool(test)
        self._batch_size = int(batch_size)
        if self._batch_size <= 0 and not self._test:
            raise ValueError("batch_size must be positive for training.")

        self.obs_shape = (2,)
        self.act_shape = (1,)
        self.multi_agent_training = False

        self.max_steps = (
            int(self._test_inputs.shape[0])
            if self._test
            else int(self._batch_size)
        )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._reward_scale = 1.0 / float(self.max_steps)

        def reset_fn(key: jnp.ndarray) -> BinaryState:
            if self._test:
                samples = self._test_inputs
                labels = self._test_labels
            else:
                subkey = key
                samples, labels = _sample_batch(
                    subkey, self._train_inputs, self._train_labels, self._batch_size
                )
            first_obs = samples[0]
            first_label = labels[0]
            return BinaryState(
                obs=first_obs,
                label=first_label,
                samples=samples,
                labels=labels,
                step=jnp.array(0, dtype=jnp.int32),
            )

        def step_fn(state: BinaryState, action: jnp.ndarray):
            logit = action[0]
            pred = jnp.where(jnn.sigmoid(logit) >= 0.5, 1.0, 0.0)
            reward = jnp.where(pred == state.label, 1.0, 0.0) * self._reward_scale

            next_step = state.step + 1
            done = next_step >= self.max_steps
            next_index = jnp.minimum(next_step, self.max_steps - 1)

            next_obs = jnp.where(done, state.obs, state.samples[next_index])
            next_label = jnp.where(done, state.label, state.labels[next_index])
            next_step_state = jnp.where(done, state.step, next_step)

            new_state = BinaryState(
                obs=next_obs,
                label=next_label,
                samples=state.samples,
                labels=state.labels,
                step=next_step_state,
            )
            return new_state, reward, jnp.asarray(done, dtype=jnp.float32)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        self._step_fn = jax.jit(jax.vmap(step_fn))

    @property
    def dataset(self) -> BinaryClassificationDataset:
        return self._dataset

    def reset(self, key: jnp.ndarray) -> BinaryState:
        return self._reset_fn(key)

    def step(
        self,
        state: TaskState,
        action: jnp.ndarray,
    ):
        return self._step_fn(state, action)
