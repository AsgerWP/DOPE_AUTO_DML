import os
import torch
import numpy as np
import pandas as pd

from datasets.datasets import IHDPDataset
from models.neural_nets.dope_net import DOPENeuralNet
from models.neural_nets.functionals import AverageTreatmentEffect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def run_experiment(replication_id, seed, lambda_lasso):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = IHDPDataset.load_replication(replication_id=replication_id)
    functional = AverageTreatmentEffect()
    model = DOPENeuralNet(
        moment_functional=functional,
        n_covariates=25,
        shared_hidden_layers=[100, 100, 100, 100],
        not_shared_hidden_layers=[100, 100],
        activation=torch.nn.ELU,
        outcome_branch_type="t_learner",
        riesz_branch_type="s_learner",
        activation_after_final_shared_layer=False,
    )
    model.to(device)

    model.fit_outcome_branch(data=data, lambda_lasso=lambda_lasso)
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
        "representation_size": lambda_lasso,
    }


if __name__ == "__main__":
    results = []
    output_file = "results/dope_neural_net_results/ihdp_lambda_lasso_experiment.csv"
    for replication_id in range(1000):
        print(f"Replication ID: {replication_id + 1}")
        rows = []
        for lambda_lasso in [0, 0.01, 0.1, 1, 10]:
            rows.append(
                run_experiment(
                    replication_id=replication_id + 1, seed=replication_id + 1, lambda_lasso=lambda_lasso
                )
            )
        file_exists = os.path.isfile(output_file)
        df = pd.DataFrame(rows)
        df.to_csv(output_file, mode="a", index=False, header=not file_exists)
