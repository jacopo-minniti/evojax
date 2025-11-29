#!/usr/bin/env python3
"""
Render a NEATBackprop genome to a PNG topology plot.

Example:
    python scripts/visualize_neat_backprop_topology.py \
        --model-path log/NEAT_backprop/binary_classification/default/best.npz
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml
from graphviz import Digraph

from evojax.algo.neat_backprop import NEATBackprop
from evojax.policy.neat import NEATPolicy


def _load_yaml(config_fname: str) -> dict:
    """Load YAML config file."""
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    with open(config_fname) as file:
        yaml_config = yaml.load(file, Loader=loader)
    return yaml_config


def _build_policy(config: Dict) -> Tuple[NEATPolicy, Dict]:
    es_cfg = dict(config.get("es_config", {}))
    n_input = int(es_cfg.get("n_input", 0))
    n_output = int(es_cfg.get("n_output", 0))
    if n_input <= 0 or n_output <= 0:
        raise ValueError("es_config.n_input and es_config.n_output must be set in the config.")
    max_hidden = int(es_cfg.get("max_hidden_nodes", config.get("max_hidden_nodes", 32)))
    propagation_steps = config.get("propagation_steps", es_cfg.get("propagation_steps"))
    policy = NEATPolicy(
        input_dim=n_input,
        output_dim=n_output,
        max_hidden_nodes=max_hidden,
        propagation_steps=propagation_steps,
    )
    es_cfg.setdefault("pop_size", config.get("pop_size", 128))
    es_cfg.setdefault("max_hidden_nodes", max_hidden)
    es_cfg.setdefault("activation_choices", [1, 5, 9])
    es_cfg.setdefault("n_input", n_input)
    es_cfg.setdefault("n_output", n_output)
    return policy, es_cfg


def _load_params(model_path: Path) -> np.ndarray:
    params = np.load(model_path)["params"]
    return np.asarray(params, dtype=np.float32).reshape(-1)


def _decode_genome(config: Dict, model_path: Path):
    policy, es_cfg = _build_policy(config)
    seed = int(config.get("seed", 0))
    solver = NEATBackprop(param_size=policy.num_params, seed=seed, **es_cfg)
    params_flat = _load_params(model_path)
    if params_flat.size != solver.param_size:
        raise ValueError(
            f"Model param length {params_flat.size} does not match solver expectation {solver.param_size}."
        )
    return solver._decode_params(params_flat)


def _build_graph(node_arr: np.ndarray, conn_arr: np.ndarray) -> Digraph:
    dot = Digraph("NEATBackprop_Topology", format="png")
    dot.attr(
        rankdir="LR",
        splines="true",
        concentrate="true",
        ranksep="3.4",
        nodesep="0.6",
        pad="0.5",
    )
    dot.attr("graph", ratio="1.6")
    dot.attr("node", shape="circle", style="filled", fontname="Helvetica", fontsize="10")

    color_map = {
        1: "lightblue",   # input
        2: "orange",      # output
        3: "lightgreen",  # hidden
        4: "lightgray",   # bias
    }
    activation_map = {
        1: "lin",
        2: "step",
        3: "sin",
        4: "gauss",
        5: "tanh",
        6: "sigm",
        7: "neg",
        8: "abs",
        9: "relu",
        10: "cos",
        11: "sq",
    }
    rank_names = ["bias", "input", "hidden", "output"]
    ranks = {name: [] for name in rank_names}
    type_to_rank = {1: "input", 2: "output", 3: "hidden", 4: "bias"}

    for node_id, node_type, act_id in node_arr.T:
        node_id, node_type, act_id = int(node_id), int(node_type), int(act_id)
        rank_name = type_to_rank.get(node_type)
        if rank_name:
            ranks[rank_name].append(node_id)
        label = "" if node_type == 2 else activation_map.get(act_id, "")
        dot.node(str(node_id), label=label, fillcolor=color_map.get(node_type, "white"))

    enabled_mask = conn_arr[4] >= 0.5
    weights = conn_arr[3, enabled_mask]
    if len(weights) > 0:
        norm_w = (np.abs(weights) - np.min(np.abs(weights))) / (np.ptp(np.abs(weights)) + 1e-9)
    else:
        norm_w = np.array([])
    w_iter = iter(norm_w)

    for _, src, dst, weight, enabled in conn_arr.T:
        if enabled < 0.5 or int(src) == int(dst):
            continue
        penwidth = 0.3 + 2.5 * next(w_iter, 0.5)
        dot.edge(str(int(src)), str(int(dst)), color="black", penwidth=str(penwidth))

    rank_groups = [
        ("input_group", ranks["bias"] + ranks["input"]),
        ("hidden", ranks["hidden"]),
        ("output", ranks["output"]),
    ]
    for _, nodes in rank_groups:
        if nodes:
            with dot.subgraph() as s:
                s.attr(rank="same")
                for n in nodes:
                    s.node(str(n))

    ordered_layers = [nodes for _, nodes in rank_groups if nodes]
    for src_nodes, dst_nodes in zip(ordered_layers, ordered_layers[1:]):
        dot.edge(str(src_nodes[0]), str(dst_nodes[0]), style="invis", weight="10")

    return dot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a NEATBackprop model topology to PNG.")
    parser.add_argument("--model-path", required=True, help="Path to a saved .npz (e.g. .../best.npz).")
    parser.add_argument(
        "--config-path",
        help="Path to the training config YAML (defaults to <model_dir>/config.yaml).",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for the PNG (defaults to <model_dir>/neat_backprop_topology.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cfg_path = Path(args.config_path).expanduser().resolve() if args.config_path else model_path.parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    config = _load_yaml(str(cfg_path))
    genome = _decode_genome(config, model_path)
    node_arr, conn_arr = genome.to_arrays()

    dot = _build_graph(node_arr, conn_arr)
    output_path = Path(args.output).expanduser().resolve() if args.output else model_path.parent / "neat_backprop_topology"
    dot.render(output_path, cleanup=True)
    print(f"Saved visualization to {output_path}.png")


if __name__ == "__main__":
    main()
