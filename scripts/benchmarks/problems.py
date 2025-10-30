import ast
import re
import yaml
import json
import numpy as np
from evojax.policy import MLPPolicy
from evojax.policy.convnet import ConvNetPolicy
from evojax.policy.neat import NEATPolicy
from evojax.task import BinaryClassification
from evojax.task.binary_dataset import BinaryClassificationDataset


def setup_problem(config, logger):
    if config["problem_type"] == "cartpole_easy":
        return setup_cartpole(config, False)
    elif config["problem_type"] == "cartpole_hard":
        return setup_cartpole(config, True)
    elif config["problem_type"] == "brax":
        return setup_brax(config)
    elif config["problem_type"] == "mnist":
        return setup_mnist(config, logger)
    elif config["problem_type"] == "waterworld":
        return setup_waterworld(config)
    elif config["problem_type"] == "waterworld_ma":
        return setup_waterworld_ma(config)
    elif config["problem_type"] == "slimevolley":
        return setup_slimevolley(config)
    elif config["problem_type"] == "binary_classification":
        return setup_binary_classification(config)


def setup_cartpole(config, hard=False):
    from evojax.task.cartpole import CartPoleSwingUp

    train_task = CartPoleSwingUp(test=False, harder=hard)
    test_task = CartPoleSwingUp(test=True, harder=hard)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[config["hidden_size"]] * 2,
        output_dim=train_task.act_shape[0],
    )
    return train_task, test_task, policy


def setup_brax(config):
    from evojax.task.brax_task import BraxTask

    train_task = BraxTask(env_name=config["env_name"], test=False)
    test_task = BraxTask(env_name=config["env_name"], test=True)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        hidden_dims=[32, 32, 32, 32],
    )
    return train_task, test_task, policy


def setup_mnist(config, logger):
    from evojax.task.mnist import MNIST

    policy = ConvNetPolicy(logger=logger)
    train_task = MNIST(batch_size=config["batch_size"], test=False)
    test_task = MNIST(batch_size=config["batch_size"], test=True)
    return train_task, test_task, policy


def setup_waterworld(config, max_steps=500):
    from evojax.task.waterworld import WaterWorld

    train_task = WaterWorld(test=False, max_steps=max_steps)
    test_task = WaterWorld(test=True, max_steps=max_steps)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[
            config["hidden_size"],
        ],
        output_dim=train_task.act_shape[0],
        output_act_fn="softmax",
    )
    return train_task, test_task, policy


def setup_waterworld_ma(config, num_agents=16, max_steps=500):
    from evojax.task.ma_waterworld import MultiAgentWaterWorld

    train_task = MultiAgentWaterWorld(
        num_agents=num_agents, test=False, max_steps=max_steps
    )
    test_task = MultiAgentWaterWorld(
        num_agents=num_agents, test=True, max_steps=max_steps
    )
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[-1],
        hidden_dims=[
            config["hidden_size"],
        ],
        output_dim=train_task.act_shape[-1],
        output_act_fn="softmax",
    )
    return train_task, test_task, policy


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


def setup_binary_classification(config):
    dataset_cfg = config.setdefault("problem_config", {})
    dataset_cfg.setdefault("dataset_type", "circle")
    dataset_cfg.setdefault("train_size", 200)
    dataset_cfg.setdefault("test_size", 200)
    dataset_cfg.setdefault("dataset_noise", 0.5)
    dataset_cfg.setdefault("dataset_seed", config.get("seed", 0))
    dataset_cfg.setdefault("batch_size", 32)

    dataset = BinaryClassificationDataset(
        dataset_type=dataset_cfg["dataset_type"],
        train_size=int(dataset_cfg["train_size"]),
        test_size=int(dataset_cfg["test_size"]),
        noise=float(dataset_cfg["dataset_noise"]),
        seed=int(dataset_cfg["dataset_seed"]),
    )

    train_task = BinaryClassification(
        batch_size=int(dataset_cfg["batch_size"]),
        test=False,
        dataset_type=dataset_cfg["dataset_type"],
        train_size=int(dataset_cfg["train_size"]),
        test_size=int(dataset_cfg["test_size"]),
        noise=float(dataset_cfg["dataset_noise"]),
        dataset_seed=int(dataset_cfg["dataset_seed"]),
        dataset=dataset,
    )
    test_task = BinaryClassification(
        batch_size=int(dataset_cfg["test_size"]),
        test=True,
        dataset_type=dataset_cfg["dataset_type"],
        train_size=int(dataset_cfg["train_size"]),
        test_size=int(dataset_cfg["test_size"]),
        noise=float(dataset_cfg["dataset_noise"]),
        dataset_seed=int(dataset_cfg["dataset_seed"]),
        dataset=dataset,
    )

    es_cfg = config.setdefault("es_config", {})
    max_hidden = int(config.get("max_hidden_nodes", es_cfg.get("max_hidden_nodes", 16)))
    propagation_steps = config.get("propagation_steps")
    policy = NEATPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        max_hidden_nodes=max_hidden,
        propagation_steps=propagation_steps,
    )

    es_cfg.setdefault("pop_size", int(config.get("pop_size", 64)))
    es_cfg.setdefault("n_input", policy.input_dim)
    es_cfg.setdefault("n_output", policy.output_dim)
    es_cfg.setdefault("max_hidden_nodes", max_hidden)
    es_cfg.setdefault("activation_choices", [1, 5, 9])
    es_cfg.setdefault("batch_size", int(dataset_cfg["batch_size"]))
    es_cfg.setdefault("grad_steps", 4)
    es_cfg.setdefault("learning_rate", 1e-2)
    es_cfg.setdefault("dataset_type", dataset_cfg["dataset_type"])
    es_cfg.setdefault("train_size", int(dataset_cfg["train_size"]))
    es_cfg.setdefault("test_size", int(dataset_cfg["test_size"]))
    es_cfg.setdefault("dataset_noise", float(dataset_cfg["dataset_noise"]))
    es_cfg.setdefault("dataset_seed", int(dataset_cfg["dataset_seed"]))
    return train_task, test_task, policy


def convert(obj):
    """Conversion helper instead of JSON encoder for handling booleans."""
    if isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [convert(item) for item in obj]
    if isinstance(obj, dict):
        return {convert(key): convert(value) for key, value in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return convert(obj.tolist())
    if isinstance(obj, np.bool_):
        return int(obj)
    return obj


def save_yaml(obj: dict, filename: str) -> None:
    """Save object as yaml file."""
    data = json.dumps(convert(obj), indent=1)
    data_dump = ast.literal_eval(data)
    with open(filename, "w") as f:
        yaml.safe_dump(data_dump, f, default_flow_style=False)


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
