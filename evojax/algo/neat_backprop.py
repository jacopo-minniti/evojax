# neat_evojax.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import random
import numpy as np

from evojax.util import create_logger
from evojax.algo.base import NEAlgorithm
from evojax.policy.neat import NEATPolicy
from evojax.task.binary_dataset import BinaryClassificationDataset


# ---------------------------------------------------------------------------
# Utility helpers (ported from prettyNEAT/utils and ann modules).
# ---------------------------------------------------------------------------

def rank_array(values: np.ndarray) -> np.ndarray:
    idx = np.argsort(values)
    rank = np.empty_like(idx)
    rank[idx] = np.arange(len(values))
    return rank


def tied_rank(values: np.ndarray) -> np.ndarray:
    pairs = sorted([(val, i) for i, val in enumerate(values)], reverse=True)
    ranks = np.zeros(len(values), dtype=float)
    start = 0
    for i in range(1, len(pairs)):
        if pairs[i][0] != pairs[i - 1][0]:
            for j in range(start, i):
                ranks[pairs[j][1]] = (start + i + 1) / 2.0
            start = i
    for j in range(start, len(pairs)):
        ranks[pairs[j][1]] = (start + len(pairs) + 1) / 2.0
    return ranks


def best_int_split(ratio: np.ndarray, total: int) -> np.ndarray:
    ratio = np.asarray(ratio, dtype=float).flatten()
    if ratio.sum() <= 0:
        if ratio.size == 0:
            return np.zeros(0, dtype=int)
        ints = np.full(ratio.size, total // ratio.size, dtype=int)
        remainder = total - ints.sum()
        if remainder > 0:
            ints[:remainder] += 1
        return ints
    normalized = ratio / ratio.sum()
    floats = normalized * total
    ints = np.floor(floats).astype(int)
    remainder = total - ints.sum()
    if remainder > 0:
        deserving = np.argsort(-(floats - ints))
        ints[deserving[:remainder]] += 1
    return ints


def quick_intersect(a: Iterable[int], b: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    a_arr = np.asarray(list(a), dtype=int)
    b_arr = np.asarray(list(b), dtype=int)
    if a_arr.size == 0 or b_arr.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    lookup = np.zeros(max(a_arr.max(), b_arr.max()) + 1, dtype=bool)
    lookup[a_arr] = True
    mask_b = lookup[b_arr]
    lookup[a_arr] = False
    lookup[b_arr] = True
    mask_a = lookup[a_arr]
    return mask_a, mask_b


def get_node_order(node_genes: np.ndarray, conn_genes: np.ndarray):
    nodes = np.copy(node_genes)
    conns = np.copy(conn_genes)
    if conns.size > 0:
        conns[3, conns[4, :] == 0] = np.nan
    src = conns[1, :].astype(int) if conns.size > 0 else np.array([], dtype=int)
    dst = conns[2, :].astype(int) if conns.size > 0 else np.array([], dtype=int)
    lookup = nodes[0, :].astype(int)
    src_idx = src.copy()
    dst_idx = dst.copy()
    for i, node_id in enumerate(lookup):
        src_idx[src == node_id] = i
        dst_idx[dst == node_id] = i
    total_nodes = nodes.shape[1]
    w_mat = np.zeros((total_nodes, total_nodes))
    if conns.size > 0:
        w_mat[src_idx, dst_idx] = conns[3, :]
    n_input = np.sum(nodes[1, :] == 1) + np.sum(nodes[1, :] == 4)
    n_output = np.sum(nodes[1, :] == 2)
    core = w_mat[n_input:nodes.shape[1] - n_output, n_input:nodes.shape[1] - n_output]
    core_mask = np.array(core != 0.0, dtype=float)
    in_degree = np.sum(core_mask, axis=0)
    queue = list(np.where(in_degree == 0)[0])
    order = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        outgoing = core_mask[current, :]
        in_degree -= outgoing
        new_nodes = np.where((in_degree == 0) & (~np.isin(np.arange(len(in_degree)), order)))[0]
        for node in new_nodes:
            if node not in queue:
                queue.append(int(node))
    if len(order) != core.shape[0]:
        return False, False
    order = [n_input + idx for idx in order]
    full_order = np.concatenate(
        [lookup[nodes[1, :] == 4], lookup[(nodes[1, :] == 1)], np.array(order), lookup[(nodes[1, :] == 2)]]
    ).astype(int)
    w_mat = w_mat[np.ix_(full_order, full_order)]
    return full_order, w_mat


def get_layers(weight_matrix: np.ndarray) -> np.ndarray:
    matrix = np.array(weight_matrix, dtype=float)
    matrix[np.isnan(matrix)] = 0.0
    matrix[matrix != 0] = 1.0
    n = matrix.shape[0]
    layers = np.zeros(n)
    stable = False
    while not stable:
        prev = layers.copy()
        for i in range(n):
            parents = layers * matrix[:, i]
            layers[i] = np.max(parents) + 1.0
        stable = np.allclose(prev, layers)
    return layers - 1.0




# ---------------------------------------------------------------------------
# Genome and Species data structures.
# ---------------------------------------------------------------------------

@dataclass
class NodeGene:
    node_id: int
    node_type: int  # 1=input, 2=output, 3=hidden, 4=bias
    activation: int


@dataclass
class ConnectionGene:
    innovation: int
    source: int
    target: int
    weight: float
    enabled: bool = True

    def clone(self) -> "ConnectionGene":
        return ConnectionGene(
            innovation=self.innovation,
            source=self.source,
            target=self.target,
            weight=self.weight,
            enabled=self.enabled,
        )


@dataclass
class Genome:
    nodes: Dict[int, NodeGene]
    connections: Dict[int, ConnectionGene]
    n_input: int
    n_output: int
    bias_id: int = 0
    fitness: float = np.nan
    rank: int = 0
    birth: int = 0
    species: int = -1
    n_conn: int = 0

    def clone(self) -> "Genome":
        return Genome(
            nodes={nid: NodeGene(g.node_id, g.node_type, g.activation) for nid, g in self.nodes.items()},
            connections={iid: gene.clone() for iid, gene in self.connections.items()},
            n_input=self.n_input,
            n_output=self.n_output,
            bias_id=self.bias_id,
            fitness=self.fitness,
            rank=self.rank,
            birth=self.birth,
            species=self.species,
            n_conn=self.n_conn,
        )

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        node_ids = sorted(self.nodes.keys())
        node_arr = np.zeros((3, len(node_ids)))
        for idx, node_id in enumerate(node_ids):
            gene = self.nodes[node_id]
            node_arr[0, idx] = gene.node_id
            node_arr[1, idx] = gene.node_type
            node_arr[2, idx] = gene.activation
        conn_genes = sorted(self.connections.values(), key=lambda g: g.innovation)
        conn_arr = np.zeros((5, len(conn_genes)))
        for idx, gene in enumerate(conn_genes):
            conn_arr[0, idx] = gene.innovation
            conn_arr[1, idx] = gene.source
            conn_arr[2, idx] = gene.target
            conn_arr[3, idx] = gene.weight
            conn_arr[4, idx] = 1.0 if gene.enabled else 0.0
        return node_arr, conn_arr

    def express(self) -> bool:
        node_arr, conn_arr = self.to_arrays()
        if node_arr.size == 0:
            return False
        order, _ = get_node_order(node_arr, conn_arr)
        if order is False:
            return False
        self.n_conn = sum(1 for conn in self.connections.values() if conn.enabled)
        return True

    def mutate(
        self,
        cfg: Dict[str, float],
        rng: np.random.Generator,
        innov: "InnovationManager",
        generation: int,
        node_limit: int,
        activation_choices: Sequence[int],
    ) -> None:
        # 1. re-enable disabled connections
        disabled = [conn for conn in self.connections.values() if not conn.enabled]
        for conn in disabled:
            if rng.random() < cfg["prob_enable"]:
                conn.enabled = True
        
        # 2. weights are trained via backpropagation in NEATBackprop; skip stochastic weight jitters.
        
        # 3. pick random activation function
        hidden_nodes = [node for node in self.nodes.values() if node.node_type == 3]
        if cfg["prob_mut_act"] > 0.0:
            for node in hidden_nodes:
                if rng.random() < cfg["prob_mut_act"]:
                    node.activation = int(activation_choices[int(rng.integers(len(activation_choices)))])
        
        # 4. add random node with connection: in node--1.0-->new node--old weight-->out node
        if rng.random() < cfg["prob_add_node"]:
            self._mutate_add_node(cfg, rng, innov, generation, node_limit, activation_choices)
        
        # 5. add feed-forward, valid connection
        if rng.random() < cfg["prob_add_conn"]:
            self._mutate_add_connection(cfg, rng, innov, generation, node_limit)

        self.n_conn = sum(1 for conn in self.connections.values() if conn.enabled)

    def _mutate_add_node(
        self,
        cfg: Dict[str, float],
        rng: np.random.Generator,
        innov: "InnovationManager",
        generation: int,
        node_limit: int,
        activation_choices: Sequence[int],
    ) -> None:
        active = [conn for conn in self.connections.values() if conn.enabled]
        if not active:
            return
        conn = rng.choice(active)
        split = innov.ensure_split(conn.source, conn.target, node_limit)
        if split is None:
            return
        new_node_id, inn1, inn2 = split
        if new_node_id not in self.nodes:
            activation = int(activation_choices[int(rng.integers(len(activation_choices)))])
            self.nodes[new_node_id] = NodeGene(new_node_id, 3, activation)
        conn.enabled = False
        conn_to = ConnectionGene(inn1, conn.source, new_node_id, 1.0, enabled=True)
        conn_from = ConnectionGene(inn2, new_node_id, conn.target, conn.weight, enabled=True)
        self.connections[inn1] = conn_to
        self.connections[inn2] = conn_from
        self.n_conn = sum(1 for gene in self.connections.values() if gene.enabled)

    def _mutate_add_connection(
        self,
        cfg: Dict[str, float],
        rng: np.random.Generator,
        innov: "InnovationManager",
        generation: int,
        node_limit: int,
    ) -> None:
        node_arr, conn_arr = self.to_arrays()
        order, w_mat = get_node_order(node_arr, conn_arr)
        if order is False:
            return
        n_inputs = self.n_input + 1
        n_outputs = self.n_output
        hidden_matrix = w_mat[n_inputs:-n_outputs or None, n_inputs:-n_outputs or None]
        if hidden_matrix.size == 0:
            layers = np.zeros(0)
        else:
            layers = get_layers(hidden_matrix) + 1.0
        last_layer = (np.max(layers) + 1.0) if layers.size > 0 else 1.0
        node_layers = np.concatenate(
            [
                np.zeros(n_inputs),
                layers,
                np.full(n_outputs, last_layer),
            ]
        )
        order_nodes = np.concatenate(
            [
                node_arr[0, node_arr[1, :] == 4],
                node_arr[0, node_arr[1, :] == 1],
                order[n_inputs:-n_outputs or None],
                node_arr[0, node_arr[1, :] == 2],
            ]
        )
        existing_pairs = {(conn.source, conn.target) for conn in self.connections.values()}
        perm = rng.permutation(len(order_nodes))
        for idx in perm:
            src_id = int(order_nodes[idx])
            src_layer = node_layers[idx]
            candidates = [
                int(order_nodes[j])
                for j, layer in enumerate(node_layers)
                if layer > src_layer and (src_id, int(order_nodes[j])) not in existing_pairs
            ]
            if not candidates:
                continue
            dst_id = int(rng.choice(candidates))
            innov_id = innov.ensure_connection(src_id, dst_id)
            weight = float(rng.uniform(-cfg["ann_abs_w_cap"], cfg["ann_abs_w_cap"]))
            self.connections[innov_id] = ConnectionGene(innov_id, src_id, dst_id, weight, enabled=True)
            return

    @staticmethod
    def from_state(state: Dict[str, Any], n_input: int, n_output: int) -> "Genome":
        nodes = {
            int(node_id): NodeGene(int(node_id), int(gene["type"]), int(gene["act"]))
            for node_id, gene in state["nodes"].items()
        }
        connections = {
            int(innov): ConnectionGene(
                innovation=int(innov),
                source=int(gene["src"]),
                target=int(gene["dst"]),
                weight=float(gene["w"]),
                enabled=bool(gene["enabled"]),
            )
            for innov, gene in state["connections"].items()
        }
        genome = Genome(
            nodes=nodes,
            connections=connections,
            n_input=n_input,
            n_output=n_output,
            bias_id=state.get("bias_id", 0),
            fitness=state.get("fitness", np.nan),
            rank=state.get("rank", 0),
            birth=state.get("birth", 0),
            species=state.get("species", -1),
            n_conn=state.get("n_conn", 0),
        )
        return genome

    def to_state(self) -> Dict[str, Any]:
        return {
            "nodes": {
                int(node_id): {"type": gene.node_type, "act": gene.activation}
                for node_id, gene in self.nodes.items()
            },
            "connections": {
                int(innov): {
                    "src": gene.source,
                    "dst": gene.target,
                    "w": gene.weight,
                    "enabled": gene.enabled,
                }
                for innov, gene in self.connections.items()
            },
            "bias_id": self.bias_id,
            "fitness": self.fitness,
            "rank": self.rank,
            "birth": self.birth,
            "species": self.species,
            "n_conn": self.n_conn,
        }


class Species:
    def __init__(self, seed: Genome):
        self.seed = seed.clone()
        self.members: List[Genome] = [seed]
        self.best_ind = seed.clone()
        self.best_fit = seed.fitness if not np.isnan(seed.fitness) else -np.inf
        self.last_improved = 0
        self.n_offspring = 0


class InnovationManager:
    def __init__(self, next_conn_id: int, next_node_id: int):
        self._conn_innov: Dict[Tuple[int, int], int] = {}
        self._node_innov: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        self._next_conn_id = next_conn_id
        self._next_node_id = next_node_id

    def ensure_connection(self, src: int, dst: int) -> int:
        key = (int(src), int(dst))
        if key not in self._conn_innov:
            self._conn_innov[key] = self._next_conn_id
            self._next_conn_id += 1
        return self._conn_innov[key]

    def ensure_split(self, src: int, dst: int, node_limit: int) -> Optional[Tuple[int, int, int]]:
        key = (int(src), int(dst))
        if key in self._node_innov:
            return self._node_innov[key]
        if self._next_node_id >= node_limit:
            return None
        new_node_id = self._next_node_id
        self._next_node_id += 1
        inn1 = self.ensure_connection(src, new_node_id)
        inn2 = self.ensure_connection(new_node_id, dst)
        self._node_innov[key] = (new_node_id, inn1, inn2)
        return self._node_innov[key]

    def update_next_node(self, node_id: int) -> None:
        self._next_node_id = max(self._next_node_id, node_id)

    def to_state(self) -> Dict[str, Any]:
        return {
            "conn": {f"{src}:{dst}": innov for (src, dst), innov in self._conn_innov.items()},
            "node": {f"{src}:{dst}": {"node": node, "in": inn1, "out": inn2} for (src, dst), (node, inn1, inn2) in
self._node_innov.items()},
            "next_conn": self._next_conn_id,
            "next_node": self._next_node_id,
        }

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "InnovationManager":
        manager = InnovationManager(state.get("next_conn", 0), state.get("next_node", 0))
        for key, innov in state.get("conn", {}).items():
            src, dst = map(int, key.split(":"))
            manager._conn_innov[(src, dst)] = int(innov)
        for key, payload in state.get("node", {}).items():
            src, dst = map(int, key.split(":"))
            manager._node_innov[(src, dst)] = (int(payload["node"]), int(payload["in"]), int(payload["out"]))
        return manager


# ---------------------------------------------------------------------------
# Evolutionary NEAT solver interfacing with EvoJAX.
# ---------------------------------------------------------------------------

class NEATBackprop(NEAlgorithm):
    """NEAT variant that trains connection weights via backpropagation."""

    def __init__(
        self,
        param_size: int,
        pop_size: int,
        seed: int,
        n_input: int,
        n_output: int,
        max_hidden_nodes: int,
        activation_choices: Sequence[int],
        alg_speciate: str = "neat",
        alg_prob_moo: float = 0.0,
        prob_add_conn: float = 0.05,
        prob_add_node: float = 0.03,
        prob_crossover: float = 0.8,
        prob_enable: float = 0.01,
        prob_mut_conn: float = 0.8,
        prob_mut_act: float = 0.0,
        prob_init_enable: float = 1.0,
        select_cull_ratio: float = 0.1,
        select_elite_ratio: float = 0.1,
        select_tourn_size: int = 2,
        select_rank_weight: str = "exp",
        spec_compat_mod: float = 0.25,
        spec_drop_off_age: int = 64,
        spec_target: int = 4,
        spec_thresh: float = 2.0,
        spec_thresh_min: float = 2.0,
        spec_gene_coef: float = 1.0,
        spec_weight_coef: float = 0.5,
        ann_abs_w_cap: float = 5.0,
        ann_mut_sigma: Optional[float] = None,
        init_activation: Optional[int] = None,
        dataset_type: str = "circle",
        train_size: int = 200,
        test_size: int = 200,
        dataset_noise: float = 0.5,
        batch_size: int = 32,
        learning_rate: float = 1e-2,
        grad_steps: int = 1,
        propagation_steps: Optional[int] = None,
        dataset_seed: Optional[int] = None,
        dataset: Optional[BinaryClassificationDataset] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if logger is None:
            self._logger = create_logger("NEATBackprop")
        else:
            self._logger = logger

        self.pop_size = int(pop_size)
        self.param_size = int(param_size)
        self._key = random.PRNGKey(seed)
        self._n_input = int(n_input)
        self._n_output = int(n_output)
        self._bias_count = 1
        self._hidden_start = 1 + self._n_input + self._n_output
        self._max_nodes = self._hidden_start + int(max_hidden_nodes)

        self._activation_choices = tuple(int(a) for a in activation_choices)
        if not self._activation_choices:
            raise ValueError("activation_choices must contain at least one activation id.")
        self._ann_init_act = int(init_activation if init_activation is not None else self._activation_choices[0])
        ann_mut_sigma = ann_abs_w_cap * 0.2 if ann_mut_sigma is None else ann_mut_sigma
        dataset_type = dataset_type.lower()
        if dataset is not None:
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
        else:
            ds_seed = int(dataset_seed if dataset_seed is not None else (seed + 1))
            self._dataset = BinaryClassificationDataset(
                dataset_type=dataset_type,
                train_size=train_size,
                test_size=test_size,
                noise=dataset_noise,
                seed=ds_seed,
            )
        self._batch_size = int(batch_size)
        if self._batch_size <= 0:
            raise ValueError("batch_size must be greater than 0 for backprop training.")
        self._learning_rate = float(learning_rate)
        if self._learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative.")
        self._grad_steps = int(grad_steps)
        if self._grad_steps < 0:
            raise ValueError("grad_steps must be non-negative.")
        self._propagation_steps = propagation_steps
        self._refresh_dataset_arrays()

        self._cfg: Dict[str, Any] = {
            "alg_speciate": alg_speciate,
            "alg_prob_moo": float(alg_prob_moo),
            "prob_add_conn": float(prob_add_conn),
            "prob_add_node": float(prob_add_node),
            "prob_crossover": float(prob_crossover),
            "prob_enable": float(prob_enable),
            "prob_mut_conn": float(prob_mut_conn),
            "prob_mut_act": float(prob_mut_act),
            "prob_init_enable": float(prob_init_enable),
            "select_cull_ratio": float(select_cull_ratio),
            "select_elite_ratio": float(select_elite_ratio),
            "select_tourn_size": int(select_tourn_size),
            "select_rank_weight": select_rank_weight,
            "spec_compat_mod": float(spec_compat_mod),
            "spec_drop_off_age": int(spec_drop_off_age),
            "spec_target": int(spec_target),
            "spec_thresh": float(spec_thresh),
            "spec_thresh_min": float(spec_thresh_min),
            "spec_gene_coef": float(spec_gene_coef),
            "spec_weight_coef": float(spec_weight_coef),
            "ann_abs_w_cap": float(ann_abs_w_cap),
            "ann_mut_sigma": float(ann_mut_sigma),
        }
        self._hidden_limit = self._max_nodes
        self._weight_size = self._max_nodes * self._max_nodes
        self._weights_slice = slice(0, self._weight_size)
        self._gene_weights_slice = slice(self._weight_size, 2 * self._weight_size)
        self._state_slice = slice(2 * self._weight_size, 3 * self._weight_size)
        node_type_start = 3 * self._weight_size
        self._node_type_slice = slice(node_type_start, node_type_start + self._max_nodes)
        self._node_act_slice = slice(
            self._node_type_slice.stop, self._node_type_slice.stop + self._max_nodes
        )
        computed_param_size = self._node_act_slice.stop
        if self.param_size != computed_param_size:
            raise ValueError(
                f"param_size={self.param_size} does not match required size {computed_param_size} "
                f"for max_nodes={self._max_nodes}."
            )
        if self._n_output != 1:
            raise ValueError("NEATBackprop currently supports binary classification with a single output node.")
        self._policy = NEATPolicy(
            input_dim=self._n_input,
            output_dim=self._n_output,
            max_hidden_nodes=max_hidden_nodes,
            propagation_steps=self._propagation_steps,
        )
        if self._policy.num_params != self.param_size:
            raise ValueError(
                f"NEATPolicy expects param_size={self._policy.num_params}, but received {self.param_size}."
            )
        mask = np.zeros(self.param_size, dtype=np.float32)
        mask[self._weights_slice] = 1.0
        self._trainable_mask = jnp.asarray(mask)
        self._forward_fn = self._policy._forward_fn
        self._last_training_loss: float = float("nan")
        self._train_accuracy: float = float("nan")
        self._test_accuracy: float = float("nan")
        self._population: List[Genome] = []
        self._species: List[Species] = []
        self._generation: int = 0
        self._encoded_genomes = jnp.zeros((self.pop_size, self.param_size), dtype=jnp.float32)
        self._decoded_params = jnp.zeros_like(self._encoded_genomes)
        self._best_params = jnp.zeros((self.param_size,), dtype=jnp.float32)
        self._best_fitness: float = float("-inf")
        self._best_genome: Optional[Genome] = None
        self._innovation = InnovationManager(next_conn_id=0, next_node_id=self._hidden_start)
        self._init_population()

    # --------------------------------------------------------------------- #
    # EvoJAX interface
    # --------------------------------------------------------------------- #

    def ask(self) -> jnp.ndarray:
        if self._population and not np.isnan(self._population[0].fitness):
            self._prob_moo()
            self._speciate()
            self._population = self._reproduce()
            self._generation += 1
        self._train_population()
        encoded = np.stack([self._encode_genome(genome) for genome in self._population], axis=0)
        self._encoded_genomes = jnp.asarray(encoded, dtype=jnp.float32)
        return self._encoded_genomes

    def tell(self, fitness: np.ndarray) -> None:
        scores = jnp.asarray(fitness)

        # ensure the shape is consistent with population
        if scores.shape[0] != self.pop_size:
            raise ValueError(f"Expected fitness length {self.pop_size}, received {scores.shape[0]}.")
        
        fitness_np = np.array(scores, dtype=float)

        # iterate through each genome to assign fitness
        for idx, fit in enumerate(fitness_np):
            genome = self._population[idx]
            genome.fitness = float(fit)
            genome.n_conn = sum(1 for conn in genome.connections.values() if conn.enabled)

            if float(fit) > self._best_fitness or self._best_genome is None:
                self._best_fitness = float(fit)
                self._best_genome = genome.clone()
                self._best_params = jnp.asarray(self._encode_genome(self._best_genome), dtype=jnp.float32)
        if self._best_genome is not None:
            params = jnp.asarray(self._encode_genome(self._best_genome), dtype=jnp.float32)
            self._train_accuracy = self._compute_accuracy(params, self._train_inputs, self._train_labels)
            self._test_accuracy = self._compute_accuracy(params, self._test_inputs, self._test_labels)

    def _train_population(self) -> None:
        if (
            self._grad_steps == 0
            or self._learning_rate == 0.0
            or self._dataset.train_length() == 0
            or not self._population
        ):
            return
        losses: List[float] = []
        cap = float(self._cfg["ann_abs_w_cap"])
        for idx, genome in enumerate(self._population):
            params = jnp.asarray(self._encode_genome(genome), dtype=jnp.float32)
            last_loss = jnp.nan
            for _ in range(self._grad_steps):
                batch_x, batch_y = self._sample_batch()
                loss_fn = lambda p: self._loss_fn(p, batch_x, batch_y)
                loss, grad = jax.value_and_grad(loss_fn)(params)
                params = params - self._learning_rate * grad * self._trainable_mask
                weight_matrix = params[self._weights_slice].reshape(self._max_nodes, self._max_nodes)
                clipped_weights = jnp.clip(weight_matrix, -cap, cap)
                params = params.at[self._weights_slice].set(clipped_weights.reshape(-1))
                stored_matrix = params[self._gene_weights_slice].reshape(self._max_nodes, self._max_nodes)
                stored_matrix = jnp.clip(stored_matrix, -cap, cap)
                state_matrix = params[self._state_slice].reshape(self._max_nodes, self._max_nodes)
                updated_stored = jnp.where(state_matrix > 0.0, clipped_weights, stored_matrix)
                params = params.at[self._gene_weights_slice].set(updated_stored.reshape(-1))
                last_loss = loss
            if jnp.isnan(last_loss):
                loss_value = float("nan")
            else:
                loss_value = float(last_loss)
            losses.append(loss_value)
            updated = self._decode_params(np.asarray(params, dtype=np.float32))
            updated.fitness = genome.fitness
            updated.rank = genome.rank
            updated.birth = genome.birth
            updated.species = genome.species
            self._population[idx] = updated
        if losses:
            finite_losses = [val for val in losses if not np.isnan(val)]
            self._last_training_loss = float(np.mean(finite_losses)) if finite_losses else float("nan")
        else:
            self._last_training_loss = float("nan")

    def _sample_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_x_np, batch_y_np = self._dataset.sample_batch(self._batch_size)
        return jnp.asarray(batch_x_np, dtype=jnp.float32), jnp.asarray(batch_y_np, dtype=jnp.float32)

    def _forward_batch(self, params: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        if inputs.size == 0:
            return jnp.zeros((0, self._n_output), dtype=jnp.float32)
        params_batch = jnp.broadcast_to(params, (inputs.shape[0], params.shape[0]))
        return self._forward_fn(params_batch, inputs.astype(jnp.float32))

    def _loss_fn(self, params: jnp.ndarray, inputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        logits = self._forward_batch(params, inputs).reshape(-1)
        targets = targets.reshape(-1)
        positive_term = jnp.maximum(logits, 0.0) - logits * targets
        loss = positive_term + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.mean(loss)

    def _compute_accuracy(self, params: jnp.ndarray, inputs: jnp.ndarray, targets: jnp.ndarray) -> float:
        if inputs.size == 0:
            return float("nan")
        logits = self._forward_batch(params, inputs).reshape(-1)
        probs = jnn.sigmoid(logits)
        preds = jnp.where(probs >= 0.5, 1.0, 0.0)
        targets = targets.reshape(-1)
        accuracy = jnp.mean((preds == targets).astype(jnp.float32))
        return float(accuracy)

    def _refresh_dataset_arrays(self) -> None:
        self._train_inputs_np = self._dataset.train_inputs.astype(np.float32, copy=True)
        self._train_labels_np = self._dataset.train_labels.astype(np.float32, copy=True)
        self._test_inputs_np = self._dataset.test_inputs.astype(np.float32, copy=True)
        self._test_labels_np = self._dataset.test_labels.astype(np.float32, copy=True)
        self._train_inputs = jnp.asarray(self._train_inputs_np)
        self._train_labels = jnp.asarray(self._train_labels_np)
        self._test_inputs = jnp.asarray(self._test_inputs_np)
        self._test_labels = jnp.asarray(self._test_labels_np)

    def attach_dataset(self, dataset: BinaryClassificationDataset) -> None:
        if not isinstance(dataset, BinaryClassificationDataset):
            raise TypeError("attach_dataset expects a BinaryClassificationDataset instance.")
        if dataset.dataset_type != self._dataset.dataset_type:
            raise ValueError(
                f"attach_dataset expected dataset_type '{self._dataset.dataset_type}' "
                f"but received '{dataset.dataset_type}'."
            )
        if dataset.train_size != self._dataset.train_size or dataset.test_size != self._dataset.test_size:
            raise ValueError("attach_dataset received dataset with mismatched train/test sizes.")
        self._dataset = dataset
        self._refresh_dataset_arrays()

    def save_state(self) -> Dict[str, Any]:
        state = {
            "population": [genome.to_state() for genome in self._population],
            "best_fitness": self._best_fitness,
            "best_genome": self._best_genome.to_state() if self._best_genome else None,
            "best_params": np.asarray(self._best_params),
            "generation": self._generation,
            "spec_thresh": self._cfg["spec_thresh"],
            "innovation": self._innovation.to_state(),
            "rng_key": np.asarray(self._key),
        }
        state["backprop"] = {
            "dataset": self._dataset.state_dict(),
            "train_loss": float(self._last_training_loss) if not np.isnan(self._last_training_loss) else float("nan"),
            "train_accuracy": float(self._train_accuracy) if not np.isnan(self._train_accuracy) else float("nan"),
            "test_accuracy": float(self._test_accuracy) if not np.isnan(self._test_accuracy) else float("nan"),
        }
        return state

    def load_state(self, saved_state: Dict[str, Any]) -> None:
        if saved_state is None:
            return
        backprop_state = saved_state.get("backprop")
        if backprop_state is not None:
            dataset_state = backprop_state.get("dataset")
            if dataset_state is not None:
                self._dataset.load_state_dict(dataset_state)
                self._refresh_dataset_arrays()
            self._last_training_loss = float(backprop_state.get("train_loss", float("nan")))
            self._train_accuracy = float(backprop_state.get("train_accuracy", float("nan")))
            self._test_accuracy = float(backprop_state.get("test_accuracy", float("nan")))
        self._population = [
            Genome.from_state(state, self._n_input, self._n_output) for state in saved_state.get("population", [])
        ]
        for genome in self._population:
            genome.express()
        self._best_fitness = float(saved_state.get("best_fitness", float("-inf")))
        best_state = saved_state.get("best_genome")
        if best_state is not None:
            self._best_genome = Genome.from_state(best_state, self._n_input, self._n_output)
            self._best_genome.express()
            self._best_params = jnp.asarray(self._encode_genome(self._best_genome), dtype=jnp.float32)
        else:
            self._best_genome = None
            self._best_params = jnp.zeros((self.param_size,), dtype=jnp.float32)
        self._generation = int(saved_state.get("generation", 0))
        self._cfg["spec_thresh"] = float(saved_state.get("spec_thresh", self._cfg["spec_thresh"]))
        self._innovation = InnovationManager.from_state(saved_state.get("innovation", {}))
        key = saved_state.get("rng_key")
        if key is not None:
            self._key = jnp.asarray(key, dtype=jnp.uint32)
        else:
            self._key = random.PRNGKey(0)
        self._species = []
        if self._population:
            encoded = np.stack([self._encode_genome(genome) for genome in self._population], axis=0)
        else:
            encoded = np.zeros((self.pop_size, self.param_size), dtype=np.float32)
        self._encoded_genomes = jnp.asarray(encoded, dtype=jnp.float32)

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.asarray(self._best_params, dtype=jnp.float32)

    @best_params.setter
    def best_params(self, params: np.ndarray) -> None:
        flat = jnp.asarray(params, dtype=jnp.float32).reshape(-1)
        if flat.shape[0] != self.param_size:
            raise ValueError(f"Expected params of shape ({self.param_size},), received {flat.shape}.")
        self._best_params = flat
        genome = self._decode_params(np.asarray(flat))
        genome.express()
        self._best_genome = genome.clone()
        self._best_fitness = float("-inf")
        self._seed_population_around(genome)
        self._last_training_loss = float("nan")
        self._train_accuracy = float("nan")
        self._test_accuracy = float("nan")

    @property
    def training_loss(self) -> float:
        return float(self._last_training_loss)

    @property
    def train_accuracy(self) -> float:
        return float(self._train_accuracy)

    @property
    def test_accuracy(self) -> float:
        return float(self._test_accuracy)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _init_population(self) -> None:
        rng = self._make_rng()[0]
        base_nodes = {}
        base_nodes[0] = NodeGene(0, 4, self._ann_init_act)

        # input + output initial nodes (minimal architecture for NEAT)
        for idx in range(self._n_input):
            base_nodes[idx + 1] = NodeGene(idx + 1, 1, self._ann_init_act)

        for idx in range(self._n_output):
            node_id = 1 + self._n_input + idx
            base_nodes[node_id] = NodeGene(node_id, 2, self._ann_init_act)

        self._innovation.update_next_node(self._hidden_start)
        connection_templates: Dict[int, ConnectionGene] = {}
        inputs = list(range(0, self._n_input + 1))
        outputs = [1 + self._n_input + idx for idx in range(self._n_output)]

        for src in inputs:
            for dst in outputs:
                innov_id = self._innovation.ensure_connection(src, dst)
                connection_templates[innov_id] = ConnectionGene(innov_id, src, dst, 0.0, True)
        self._population = []

        for _ in range(self.pop_size):
            nodes = {nid: NodeGene(g.node_id, g.node_type, g.activation) for nid, g in base_nodes.items()}
            connections = {}
            rng_local = self._make_rng()[0]

            for innov_id, template in connection_templates.items():
                weight = float(rng_local.uniform(-self._cfg["ann_abs_w_cap"], self._cfg["ann_abs_w_cap"]))
                enabled = bool(rng_local.random() < self._cfg["prob_init_enable"])
                connections[innov_id] = ConnectionGene(
                    innovation=innov_id,
                    source=template.source,
                    target=template.target,
                    weight=weight,
                    enabled=enabled,
                )
            genome = Genome(nodes, connections, self._n_input, self._n_output)
            genome.birth = 0
            genome.express()
            self._population.append(genome)

        self._species = []
        self._generation = 0

    def _seed_population_around(self, champion: Genome) -> None:
        champion = champion.clone()
        champion.fitness = np.nan
        self._population = [champion]
        for _ in range(1, self.pop_size):
            clone = champion.clone()
            rng = self._make_rng()[0]
            clone.mutate(self._cfg, rng, self._innovation, self._generation, self._hidden_limit, self._activation_choices)
            clone.express()
            clone.fitness = np.nan
            self._population.append(clone)
        self._species = []

    def _prob_moo(self) -> None:
        if any(np.isnan(genome.fitness) for genome in self._population):
            raise RuntimeError("All individuals must have fitness before reproduction.")
        
        fitness = np.array([genome.fitness for genome in self._population])
        conn_counts = np.array([max(genome.n_conn, 1) for genome in self._population], dtype=float)
        # Softer structural pressure: penalize by inverse square-root of enabled connections.
        structure_cost = 1.0 / np.sqrt(conn_counts)
        obj = np.vstack([fitness, structure_cost]).T
        rng = self._make_rng()[0]

        if self._cfg["alg_prob_moo"] < rng.random():
            ranks = nsga_sort(obj)
        else:
            ranks = rank_array(-obj[:, 0])
            
        for genome, rank in zip(self._population, ranks):
            genome.rank = int(rank)

    def _speciate(self) -> None:
        # no species differentiation (crossover is among every genome)
        if self._cfg["alg_speciate"] != "neat":
            species = Species(self._population[0])
            species.members = list(self._population)
            species.n_offspring = self.pop_size
            for genome in self._population:
                genome.species = 0
            self._species = [species]
            return
        
        # ensure that species diversity remains close to target
        if len(self._species) > self._cfg["spec_target"]:
            self._cfg["spec_thresh"] += self._cfg["spec_compat_mod"]
        elif len(self._species) < self._cfg["spec_target"]:
            self._cfg["spec_thresh"] -= self._cfg["spec_compat_mod"]
        self._cfg["spec_thresh"] = max(self._cfg["spec_thresh"], self._cfg["spec_thresh_min"])

        # 
        self._species = self._assign_species()
        self._species = self._assign_offspring(self._species)


    def _assign_species(self) -> List[Species]:
        # create a single species
        if not self._species:
            new_species = Species(self._population[0])
            new_species.n_offspring = self.pop_size
            for genome in self._population:
                genome.species = 0
            new_species.members = list(self._population)
            return [new_species]
        
        # tabula rasa species (that is, species are updated each step)
        for species in self._species:
            species.members = []

        # go sequentially for each genome
        for genome in self._population:
            assigned = False
            # based on compatibility distance, we assign greedly to one species 
            for idx, species in enumerate(self._species):
                dist = self._compatibility_distance(species.seed, genome)
                if dist < self._cfg["spec_thresh"]:
                    genome.species = idx
                    species.members.append(genome)
                    assigned = True
                    break
            
            # create new species if genome is "very unique"
            if not assigned:
                genome.species = len(self._species)
                self._species.append(Species(genome))

        for species in self._species:
            if not species.members:
                species.members.append(species.seed.clone())

        return self._species


    def _assign_offspring(self, species_list: List[Species]) -> List[Species]:
        if len(species_list) == 1:
            species_list[0].n_offspring = self.pop_size
            species_list[0].members = list(self._population)
            return species_list
        fitness = np.array([genome.fitness for genome in self._population])
        ranks = tied_rank(fitness)
        if self._cfg["select_rank_weight"] == "exp":
            scores = 1.0 / ranks
        elif self._cfg["select_rank_weight"] == "lin":
            scores = 1.0 + np.abs(ranks - len(ranks))
        else:
            self._logger.warning("Unknown rank weighting '%s', defaulting to linear.", self._cfg["select_rank_weight"])
            scores = 1.0 + np.abs(ranks - len(ranks))
        spec_indices = np.array([genome.species for genome in self._population], dtype=int)
        species_fitness = np.zeros(len(species_list))
        species_top = np.zeros(len(species_list))
        for idx, species in enumerate(species_list):
            member_mask = spec_indices == idx
            if not np.any(member_mask):
                species_fitness[idx] = 0.0
                continue
            species_fitness[idx] = np.mean(scores[member_mask])
            species_top[idx] = np.max(fitness[member_mask])
            if species_top[idx] > species.best_fit:
                species.best_fit = species_top[idx]
                best_member = species.members[int(np.argmax([member.fitness for member in species.members]))]
                species.best_ind = best_member.clone()
                species.last_improved = 0
            else:
                species.last_improved += 1
            if species.last_improved > self._cfg["spec_drop_off_age"]:
                species_fitness[idx] = 0.0
        if species_fitness.sum() == 0:
            species_fitness = np.ones_like(species_fitness)
        offspring = best_int_split(species_fitness, self.pop_size)
        for species, n_offspring in zip(species_list, offspring):
            species.n_offspring = int(n_offspring)
        species_list = [species for species in species_list if species.n_offspring > 0]
        return species_list

    def _reproduce(self) -> List[Genome]:
        offspring: List[Genome] = []
        for species in self._species:
            offspring.extend(self._recombine_species(species))
        if len(offspring) != self.pop_size:
            if len(offspring) > self.pop_size:
                offspring = offspring[: self.pop_size]
            else:
                while len(offspring) < self.pop_size:
                    offspring.append(offspring[-1].clone())
        for genome in offspring:
            genome.fitness = np.nan
        return offspring

    def _recombine_species(self, species: Species) -> List[Genome]:
        # sort based on rank
        members = sorted(species.members, key=lambda g: g.rank)
        if not members:
            return []
        
        # reproduce only from top-k
        cull_count = int(np.floor(self._cfg["select_cull_ratio"] * len(members)))
        if cull_count > 0:
            members = members[:-cull_count]
        if not members:
            members = species.members

        # reproduce exactly top members
        elite_count = int(np.floor(len(members) * self._cfg["select_elite_ratio"]))
        elite_count = min(elite_count, species.n_offspring)
        offspring: List[Genome] = [members[i].clone() for i in range(elite_count)]
        for elite in offspring:
            elite.birth = self._generation

        # for all remaining genomes, do tournament
        remaining = species.n_offspring - elite_count
        tourn_size = max(1, self._cfg["select_tourn_size"])
        rng = self._make_rng()[0]

        for _ in range(remaining):
            # sample best rank among remaining two times
            idx_a = self._tournament(members, tourn_size, rng)
            idx_b = self._tournament(members, tourn_size, rng)
            # pick best parent
            parent_a_idx, parent_b_idx = sorted([idx_a, idx_b])
            parent_a = members[parent_a_idx]
            parent_b = members[parent_b_idx]
            child = parent_a.clone()
            
            # do crossover, otherwise clone just parent a 
            if rng.random() < self._cfg["prob_crossover"]:
                # get common connections based on innovation ID
                matching = set(parent_a.connections.keys()) & set(parent_b.connections.keys())
                for innov_id in matching:
                    # for each, flip coint, we overwrite the weight to b's weigth 
                    if rng.random() < 0.5:
                        child.connections[innov_id] = parent_b.connections[innov_id].clone()

            child.mutate(self._cfg, rng, self._innovation, self._generation, self._hidden_limit, self._activation_choices)
            child.express()
            child.birth = self._generation
            offspring.append(child)

        return offspring

    def _compatibility_distance(self, ref: Genome, ind: Genome) -> float:
        ref_ids = set(ref.connections.keys())
        ind_ids = set(ind.connections.keys())
        matching = ref_ids & ind_ids
        if matching:
            weight_diff = np.mean([abs(ref.connections[i].weight - ind.connections[i].weight) for i in matching])
        else:
            weight_diff = 0.0
        gene_diff = len(ref_ids - ind_ids) + len(ind_ids - ref_ids)
        longest = max(len(ref_ids), len(ind_ids)) - (self._n_input + self._n_output)
        longest = max(longest, 1)
        gene_term = gene_diff / (1 + longest)
        return gene_term * self._cfg["spec_gene_coef"] + weight_diff * self._cfg["spec_weight_coef"]


    def _encode_genome(self, genome: Genome) -> np.ndarray:
        vector = np.zeros(self.param_size, dtype=np.float32)
        active = vector[self._weights_slice].reshape(self._max_nodes, self._max_nodes)
        stored = vector[self._gene_weights_slice].reshape(self._max_nodes, self._max_nodes)
        state = vector[self._state_slice].reshape(self._max_nodes, self._max_nodes)
        node_types = vector[self._node_type_slice]
        node_acts = vector[self._node_act_slice]


        for node_id, node in genome.nodes.items():
            if node_id >= self._max_nodes:
                continue
            node_types[node_id] = float(node.node_type)
            node_acts[node_id] = float(node.activation)

        for conn in genome.connections.values():
            if conn.source >= self._max_nodes or conn.target >= self._max_nodes:
                continue
            stored[conn.source, conn.target] = float(conn.weight)
            if conn.enabled:
                active[conn.source, conn.target] = float(conn.weight)
                state[conn.source, conn.target] = 1.0
            else:
                active[conn.source, conn.target] = 0.0
                state[conn.source, conn.target] = -1.0

        return vector


    def _decode_params(self, params: np.ndarray) -> Genome:
        active = params[self._weights_slice].reshape(self._max_nodes, self._max_nodes)
        stored = params[self._gene_weights_slice].reshape(self._max_nodes, self._max_nodes)
        state = params[self._state_slice].reshape(self._max_nodes, self._max_nodes)
        node_types = params[self._node_type_slice]
        node_acts = params[self._node_act_slice]
        nodes: Dict[int, NodeGene] = {}
        for node_id in range(self._max_nodes):
            node_type = int(round(node_types[node_id]))
            if node_type <= 0:
                continue
            activation = int(round(node_acts[node_id]))
            nodes[node_id] = NodeGene(node_id, node_type, activation)
        connections: Dict[int, ConnectionGene] = {}
        for src in range(self._max_nodes):
            for dst in range(self._max_nodes):
                state_val = state[src, dst]
                if state_val == 0.0:
                    continue
                enabled = state_val > 0.0
                weight = float(stored[src, dst])
                innov_id = self._innovation.ensure_connection(src, dst)
                connections[innov_id] = ConnectionGene(innov_id, src, dst, weight, enabled)
        if nodes:
            self._innovation.update_next_node(max(nodes.keys()) + 1)
        genome = Genome(nodes, connections, self._n_input, self._n_output)
        genome.express()
        return genome

    def _make_rng(self, count: int = 1) -> List[np.random.Generator]:
        keys = self._split_keys(count)
        generators = []
        max_seed = (2**31) - 1
        for key in keys:
            seed = int(random.randint(key,
                                      shape=(),
                                      minval=0,
                                      maxval=max_seed,
                                      dtype=jnp.int32))
            generators.append(np.random.default_rng(seed))
        return generators

    def _split_keys(self, count: int) -> List[jnp.ndarray]:
        keys = random.split(self._key, count + 1)
        self._key = keys[0]
        return list(keys[1:])

    @staticmethod
    def _tournament(pool: List[Genome], size: int, rng: np.random.Generator) -> int:
        if len(pool) == 1:
            return 0
        idxs = rng.integers(0, len(pool), size=size)
        return int(np.min(idxs))


# ---------------------------------------------------------------------------
# NSGA-II helpers (same behaviour as prettyNEAT.nsga_sort).
# ---------------------------------------------------------------------------

def nsga_sort(values: np.ndarray) -> np.ndarray:
    fronts = get_fronts(values)
    for front in fronts:
        x1 = values[front, 0]
        x2 = values[front, 1]
        crowd = get_crowding_distance(x1) + get_crowding_distance(x2)
        order = np.argsort(-crowd)
        front[:] = [front[i] for i in order]
    flat = [idx for front in fronts for idx in front]
    rank = np.empty_like(flat)
    rank[flat] = np.arange(len(flat))
    return rank


def get_fronts(values: np.ndarray) -> List[List[int]]:
    values = np.asarray(values)
    domination_sets = [[] for _ in range(len(values))]
    dominated_count = np.zeros(len(values), dtype=int)
    fronts: List[List[int]] = [[]]
    for p in range(len(values)):
        domination_sets[p] = []
        dominated_count[p] = 0
        for q in range(len(values)):
            if _dominates(values[p], values[q]):
                domination_sets[p].append(q)
            elif _dominates(values[q], values[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in domination_sets[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return ((a[0] >= b[0] and a[1] > b[1]) or (a[0] > b[0] and a[1] >= b[1]))


def get_crowding_distance(fitness: np.ndarray) -> np.ndarray:
    order = np.argsort(fitness)
    sorted_vals = fitness[order]
    shifted = np.concatenate([[np.inf], sorted_vals, [np.inf]])
    prev_dist = np.abs(sorted_vals - shifted[:-2])
    next_dist = np.abs(sorted_vals - shifted[2:])
    crowd = prev_dist + next_dist
    if sorted_vals[-1] - sorted_vals[0] > 0:
        crowd *= 1.0 / abs(sorted_vals[-1] - sorted_vals[0])
    distances = np.empty_like(crowd)
    distances[order] = crowd
    return distances
