#!/usr/bin/env python3
import argparse, yaml, numpy as np
from pathlib import Path
from graphviz import Digraph

from evojax.algo.neat import NEAT
from scripts.visualize_slimevolley import setup_slimevolley  # reuse helpers

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
    for node_id, node_type, act_id in node_arr.T:
        label = f"{int(node_id)}\nT{int(node_type)}\nA{int(act_id)}"
        color = {1: "lightblue", 2: "lightgreen", 3: "orange", 4: "gray"}.get(int(node_type), "white")
        dot.node(str(int(node_id)), label=label, style="filled", fillcolor=color)

    for _, src, dst, weight, enabled in conn_arr.T:
        if enabled < 0.5:
            continue
        dot.edge(str(int(src)), str(int(dst)), label=f"{weight:+.2f}")

    dot.render(args.out, format="png", cleanup=True)

if __name__ == "__main__":
    main()