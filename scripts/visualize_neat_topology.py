#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import re
from pathlib import Path
from graphviz import Digraph
from evojax.algo.neat import NEAT
from evojax.policy.neat import NEATPolicy
from evojax.policy import MLPPolicy


def load_yaml(config_fname: str) -> dict:
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


def setup_slimevolley(config, max_steps: int = 3000):
    from evojax.task.slimevolley import SlimeVolley

    train_task = SlimeVolley(test=False, max_steps=max_steps)
    test_task = SlimeVolley(test=True, max_steps=max_steps)

    if config["es_name"] == "NEAT":
        max_hidden = config.get("max_hidden_nodes", 32)
        propagation_steps = config.get("propagation_steps")
        policy = NEATPolicy(
            input_dim=train_task.obs_shape[0],
            output_dim=train_task.act_shape[0],
            max_hidden_nodes=max_hidden,
            propagation_steps=propagation_steps,
        )
        es_cfg = config.setdefault("es_config", {})
        es_cfg.setdefault("pop_size", config.get("pop_size", 128))
        es_cfg.setdefault("n_input", policy.input_dim)
        es_cfg.setdefault("n_output", policy.output_dim)
        es_cfg.setdefault("max_hidden_nodes", max_hidden)
        es_cfg.setdefault("activation_choices", [1, 5, 9])
    else:
        policy = MLPPolicy(
            input_dim=train_task.obs_shape[0],
            hidden_dims=[config["hidden_size"]],
            output_dim=train_task.act_shape[0],
            output_act_fn="tanh",
        )
    return train_task, test_task, policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    config_path = Path(args.config_path or model_path.parent / "config.yaml").resolve()

    cfg = yaml.safe_load(config_path.read_text())
    _, _, policy = setup_slimevolley(cfg, max_steps=cfg.get("max_steps", 3000))
    solver = NEAT(param_size=policy.num_params, **cfg["es_config"], seed=cfg["seed"])

    params = np.load(model_path)["params"]
    solver.best_params = params.reshape(-1)
    genome = solver._best_genome

    node_arr, conn_arr = genome.to_arrays()

    # --- Graph setup ---
    dot = Digraph("NEAT_Topology", format="png")
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

    # --- Node color map ---
    # NEAT NodeGene types: 1=input, 2=output, 3=hidden, 4=bias.
    color_map = {
        1: "lightblue",   # input
        2: "orange",      # output
        3: "lightgreen",  # hidden
        4: "lightgray",   # bias
    }

    # --- Activation map ---
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

    # --- Node placement by type ---
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

    # --- Normalize weights ---
    enabled_mask = conn_arr[4] >= 0.5
    weights = conn_arr[3, enabled_mask]
    if len(weights) > 0:
        norm_w = (np.abs(weights) - np.min(np.abs(weights))) / (
            np.ptp(np.abs(weights)) + 1e-9
        )
    else:
        norm_w = np.array([])
    w_iter = iter(norm_w)

    # --- Edges ---
    for _, src, dst, weight, enabled in conn_arr.T:
        if enabled < 0.5:
            continue
        if int(src) == int(dst):
            # Ignore self-loops in the visualization.
            continue
        # softer thickness scaling
        penwidth = 0.3 + 2.5 * next(w_iter, 0.5)
        dot.edge(str(int(src)), str(int(dst)), color="black", penwidth=str(penwidth))

    # --- Rank organization ---
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

    # --- Enforce layer ordering left-to-right ---
    ordered_layers = [nodes for _, nodes in rank_groups if nodes]
    for src_nodes, dst_nodes in zip(ordered_layers, ordered_layers[1:]):
        dot.edge(str(src_nodes[0]), str(dst_nodes[0]), style="invis", weight="10")

    # --- Output path (same directory as model) ---
    out_path = model_path.parent / "neat_topology"
    dot.render(out_path, cleanup=True)
    print(f"Saved visualization to {out_path}.png")


if __name__ == "__main__":
    main()
