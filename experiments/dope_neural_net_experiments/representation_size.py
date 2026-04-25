import os
import torch
import numpy as np
import pandas as pd

from datasets.datasets import IHDPDataset
from models.neural_nets.dope_net import DOPENeuralNet
from models.neural_nets.functionals import AverageTreatmentEffect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def run_experiment(replication_id, seed, representation_size):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = IHDPDataset.load_replication(replication_id=replication_id)
    functional = AverageTreatmentEffect()
    model = DOPENeuralNet(
        moment_functional=functional,
        n_covariates=25,
        shared_hidden_layers=[100, 100, 100, representation_size],
        not_shared_hidden_layers=[100, 100],
        activation=torch.nn.ELU,
        outcome_branch_type="t_learner",
        riesz_branch_type="s_learner",
        activation_after_final_shared_layer=False,
    )
    model.to(device)

    model.fit_outcome_branch(data=data)
    model.freeze_shared_trunk()
    model.fit_riesz_branch(data=data)

    estimates = model.get_estimates(data)
    truth = data.truth
    return {
        "replication_id": replication_id,
        "seed": seed,
        "point_estimate": estimates["point_estimate"],
        "var_estimate": estimates["var_estimate"],
        "truth": truth,
        "representation_size": representation_size,
    }


if __name__ == "__main__":
    results = []
    output_file = "results/results.csv"
    for replication_id in range(1000):
        print(f"Replication ID: {replication_id + 1}")
        rows = []
        for representation_size in [1, 3, 10, 100]:
            rows.append(
                run_experiment(
                    replication_id=replication_id + 1, seed=replication_id + 1, representation_size=representation_size
                )
            )
        file_exists = os.path.isfile(output_file)
        df = pd.DataFrame(rows)
        df.to_csv(output_file, mode="a", index=False, header=not file_exists)
