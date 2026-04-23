import os
import torch
import numpy as np
import pandas as pd

from datasets.datasets import IHDPDataset
from models.neural_nets.riesz_net import RieszNet
from models.neural_nets.functionals import AverageTreatmentEffect


def run_experiment(replication_id, seed):
    device = "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = IHDPDataset.load_replication(replication_id=replication_id)
    functional = AverageTreatmentEffect()
    model = RieszNet(
        moment_functional=functional,
        n_covariates=25,
        outcome_branch_type="t_learner",
        shared_hidden_layers=[100, 100, 100],
        not_shared_hidden_layers=[100, 100],
        activation=torch.nn.ELU,
        loss_weights={"riesz": 0.1, "outcome": 1, "tmle": 1},
    )
    model.to(device)
    model.fit(data=data)
    estimates = model.get_estimates(data)
    truth = data.truth
    return {
        "replication_id": replication_id,
        "seed": seed,
        "point_estimate": estimates["point_estimate"],
        "var_estimate": estimates["var_estimate"],
        "truth": truth,
    }


def _print_diagnostics(results):
    mse = sum((result["point_estimate"] - result["truth"]) ** 2 for result in results) / len(results)
    mae = sum(abs(result["point_estimate"] - result["truth"]) for result in results) / len(results)
    print("Iteration", len(results), "RMSE:", mse**0.5, "MAE", mae)


if __name__ == "__main__":
    results = []
    output_file = "results/results.csv"
    for replication_id in range(1000):
        result = run_experiment(replication_id=replication_id + 1, seed=replication_id + 1)
        file_exists = os.path.isfile(output_file)
        df = pd.DataFrame([result])
        df.to_csv(output_file, mode="a", index=False, header=not file_exists)
        results.append(result)
        _print_diagnostics(results)
