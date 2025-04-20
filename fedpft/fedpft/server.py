from typing import Callable, Dict, List, Tuple
from flwr.common import Metrics

def fedpft_get_on_fit_config_fn(
    n_mixtures: int, cov_type: str, seed: int, tol: float, max_iter: int
) -> Callable[[int], Dict[str, str]]:

    def fit_config(server_round: int) -> Dict[str, str]:
        config = {
            "n_mixtures": str(n_mixtures),
            "cov_type": cov_type,
            "seed": str(seed),
            "tol": str(tol),
            "max_iter": str(max_iter),
        }
        return config

    return fit_config

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:

    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}
