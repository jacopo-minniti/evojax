#!/usr/bin/env python3
import argparse, yaml, numpy as np
from pathlib import Path
from graphviz import Digraph

from evojax.algo.neat import NEAT

############################
# from scripts.benchmarks.problems import load_yaml, setup_problem
# from scripts.visualize_slimevolley import setup_slimevolley  # reuse helpers
import yaml
import re
from evojax.policy import MLPPolicy
from evojax.policy.neat import NEATPolicy
def load_yaml(config_fname: str) -> dict:
    """Load in YAML config file."""
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

def setup_problem(config, logger):
        return setup_slimevolley(config)
########################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--out", default="neat_topology")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    config_path = Path(args.config_path or model_path.parent / "config.yaml").resolve()

    cfg = yaml.safe_load(config_path.read_text())
    _, _, policy = setup_slimevolley(cfg, max_steps=cfg.get("max_steps", 3000))
    solver = NEAT(
        param_size=policy.num_params,
        **cfg["es_config"],
        seed=cfg["seed"],
    )

    params = np.load(model_path)["params"]
    solver.best_params = params.reshape(-1)
    genome = solver._best_genome  # contains nodes + connections

    node_arr, conn_arr = genome.to_arrays()

    dot = Digraph("neat")
    dot.attr(rankdir="TB")  # Top to bottom layout
    
    # Add nodes
    for node_id, node_type, act_id in node_arr.T:
        label = f"{int(node_id)}\nA{int(act_id)}"
        color = {1: "lightblue", 2: "lightgreen", 3: "orange", 4: "gray"}.get(int(node_type), "white")
        dot.node(str(int(node_id)), label=label, style="filled", fillcolor=color)

    # Calculate weight range for scaling edge thickness
    weights = conn_arr[3, conn_arr[4] >= 0.5]  # Only enabled connections
    if len(weights) > 0:
        max_weight = np.max(np.abs(weights))
        min_thickness = 0.5
        max_thickness = 5.0
    else:
        max_weight = 1.0
        min_thickness = max_thickness = 1.0

    # Add edges with thickness based on weight strength and activation function labels
    for _, src, dst, weight, enabled in conn_arr.T:
        if enabled < 0.5:
            continue
        
        # Calculate thickness based on absolute weight
        if max_weight > 0:
            thickness = min_thickness + (max_thickness - min_thickness) * (abs(weight) / max_weight)
        else:
            thickness = min_thickness
        
        # Color based on weight sign
        color = "red" if weight < 0 else "blue"
        
        dot.edge(str(int(src)), str(int(dst)), 
                penwidth=str(thickness), 
                color=color,
                tooltip=f"Weight: {weight:+.3f}")

    dot.render(args.out, format="png", cleanup=True)

if __name__ == "__main__":
    main()