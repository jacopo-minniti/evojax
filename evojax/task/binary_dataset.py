from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple


class BinaryClassificationDataset:
    """Utility dataset mirroring the TensorFlow Playground binary generators."""

    SUPPORTED = ("circle", "xor", "gaussian", "spiral")

    def __init__(
        self,
        dataset_type: str,
        train_size: int,
        test_size: int,
        noise: float,
        seed: int,
    ) -> None:
        dataset_type = dataset_type.lower()
        if dataset_type not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported dataset_type '{dataset_type}'. "
                f"Valid options are: {', '.join(self.SUPPORTED)}."
            )

        self.dataset_type = dataset_type
        self.train_size = int(train_size)
        self.test_size = int(test_size)
        self.noise = float(noise)
        self._rng = np.random.default_rng(seed)
        train_inputs, train_labels = self._generate(self.train_size)
        test_inputs, test_labels = self._generate(self.test_size)
        self.train_inputs, self.train_labels = self._shuffle(train_inputs, train_labels)
        self.test_inputs, self.test_labels = self._shuffle(test_inputs, test_labels)
        self._sampler = np.random.default_rng(seed ^ 0xABCDEF)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.train_inputs.size == 0:
            raise ValueError("Training set is empty; cannot sample a batch.")
        idx = self._sampler.integers(0, self.train_inputs.shape[0], size=int(batch_size))
        return (
            self.train_inputs[idx].astype(np.float32, copy=False),
            self.train_labels[idx].astype(np.float32, copy=False),
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dataset_type": self.dataset_type,
            "train_inputs": self.train_inputs,
            "train_labels": self.train_labels,
            "test_inputs": self.test_inputs,
            "test_labels": self.test_labels,
            "sampler_state": self._sampler.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.dataset_type = state.get("dataset_type", self.dataset_type)
        self.train_inputs = np.asarray(state["train_inputs"], dtype=np.float32)
        self.train_labels = np.asarray(state["train_labels"], dtype=np.float32)
        self.test_inputs = np.asarray(state["test_inputs"], dtype=np.float32)
        self.test_labels = np.asarray(state["test_labels"], dtype=np.float32)
        self.train_size = self.train_inputs.shape[0]
        self.test_size = self.test_inputs.shape[0]
        sampler_state = state.get("sampler_state")
        self._sampler = np.random.default_rng()
        if sampler_state is not None:
            self._sampler.bit_generator.state = sampler_state

    def train_length(self) -> int:
        return int(self.train_inputs.shape[0])

    def test_length(self) -> int:
        return int(self.test_inputs.shape[0])

    def _generate(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        if size <= 0:
            empty_inputs = np.zeros((0, 2), dtype=np.float32)
            empty_labels = np.zeros((0, 1), dtype=np.float32)
            return empty_inputs, empty_labels
        if self.dataset_type == "xor":
            return self._generate_xor(size)
        if self.dataset_type == "gaussian":
            return self._generate_gaussian(size)
        if self.dataset_type == "spiral":
            return self._generate_spiral(size)
        return self._generate_circle(size)

    def _generate_xor(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        inputs = self._rng.uniform(-5.0, 5.0, size=(size, 2))
        noise = self._rng.normal(0.0, self.noise, size=(size, 2))
        inputs = inputs + noise
        labels = np.logical_or(
            np.logical_and(inputs[:, 0] > 0.0, inputs[:, 1] > 0.0),
            np.logical_and(inputs[:, 0] < 0.0, inputs[:, 1] < 0.0),
        ).astype(np.float32)
        return inputs.astype(np.float32), labels.reshape(-1, 1)

    def _generate_gaussian(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        n_pos = size // 2
        n_neg = size - n_pos
        scale = self.noise + 1.0
        pos = self._rng.normal(loc=2.0, scale=scale, size=(n_pos, 2))
        neg = self._rng.normal(loc=-2.0, scale=scale, size=(n_neg, 2))
        inputs = np.vstack([pos, neg]).astype(np.float32)
        labels = np.concatenate(
            [np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)]
        ).reshape(-1, 1)
        return inputs, labels

    def _generate_circle(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        inside = size // 2
        outside = size - inside
        radius = 5.0
        inputs = np.zeros((size, 2), dtype=np.float32)
        labels = np.zeros((size, 1), dtype=np.float32)
        for i in range(inside):
            r = self._rng.uniform(0.0, radius * 0.5)
            angle = self._rng.uniform(0.0, 2.0 * np.pi)
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            noise_x = self._rng.uniform(-radius, radius) * (self.noise / 3.0)
            noise_y = self._rng.uniform(-radius, radius) * (self.noise / 3.0)
            inputs[i] = (x + noise_x, y + noise_y)
            labels[i, 0] = 1.0
        for i in range(outside):
            r = self._rng.uniform(radius * 0.75, radius)
            angle = self._rng.uniform(0.0, 2.0 * np.pi)
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            noise_x = self._rng.uniform(-radius, radius) * (self.noise / 3.0)
            noise_y = self._rng.uniform(-radius, radius) * (self.noise / 3.0)
            idx = inside + i
            inputs[idx] = (x + noise_x, y + noise_y)
            labels[idx, 0] = 0.0
        return inputs, labels

    def _generate_spiral(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        half = size // 2
        remainder = size - 2 * half
        data: List[Tuple[float, float]] = []
        labels: List[float] = []

        def gen_spiral(delta: float, label: float, count: int) -> None:
            if count <= 0:
                return
            for i in range(count):
                frac = i / max(count - 1, 1)
                r = frac * 6.0
                t = 1.75 * frac * 2.0 * np.pi + delta
                x = r * np.sin(t) + self._rng.uniform(-1.0, 1.0) * self.noise
                y = r * np.cos(t) + self._rng.uniform(-1.0, 1.0) * self.noise
                data.append((x, y))
                labels.append(label)

        gen_spiral(0.0, 1.0, half + remainder)
        gen_spiral(np.pi, 0.0, half)
        inputs = np.asarray(data, dtype=np.float32)
        label_arr = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        return inputs, label_arr

    def _shuffle(
        self, inputs: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        perm = self._rng.permutation(inputs.shape[0])
        return inputs[perm], labels[perm]
