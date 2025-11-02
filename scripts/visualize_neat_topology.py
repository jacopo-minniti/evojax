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
    dot.attr(rankdir="LR", splines="true", concentrate="true", ranksep="2.0", nodesep="1.5")
    dot.attr("node", shape="circle", style="filled", fontname="Helvetica", fontsize="10")

    # --- Node color map ---
    color_map = {
        0: "lightblue",   # input
        1: "lightgray",   # bias
        2: "lightgreen",  # hidden
        3: "orange",      # output
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
    ranks = {"input": [], "bias": [], "hidden": [], "output": []}
    for node_id, node_type, act_id in node_arr.T:
        node_id, node_type, act_id = int(node_id), int(node_type), int(act_id)
        if node_type == 0:
            ranks["input"].append(node_id)
        elif node_type == 1:
            ranks["bias"].append(node_id)
        elif node_type == 2:
            ranks["hidden"].append(node_id)
        elif node_type == 3:
            ranks["output"].append(node_id)

        label = f"{activation_map.get(act_id, '')}"
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
        # softer thickness scaling
        penwidth = 0.3 + 2.5 * next(w_iter, 0.5)
        dot.edge(str(int(src)), str(int(dst)), color="black", penwidth=str(penwidth))

    # --- Rank organization ---
    for rank_name, nodes in ranks.items():
        if nodes:
            with dot.subgraph() as s:
                s.attr(rank="same")
                for n in nodes:
                    s.node(str(n))

    # --- Output path (same directory as model) ---
    out_path = model_path.parent / "neat_topology"
    dot.render(out_path, cleanup=True)
    print(f"Saved visualization to {out_path}.png")


if __name__ == "__main__":
    main()
