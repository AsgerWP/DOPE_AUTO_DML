import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class ATEDataset:
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        self.data = data
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column
        self.covariate_columns = covariate_columns

    def split_into_train_and_test_sets(self, train_size):
        train_data, test_data = train_test_split(
            self.data, train_size=train_size, stratify=self.data[:, self.treatment_column]
        )
        return (
            ATEDataset(
                data=train_data,
                treatment_column=self.treatment_column,
                outcome_column=self.outcome_column,
                covariate_columns=self.covariate_columns,
            ),
            ATEDataset(
                data=test_data,
                treatment_column=self.treatment_column,
                outcome_column=self.outcome_column,
                covariate_columns=self.covariate_columns,
            ),
        )

    def create_dataloader(self, batch_size):
        covariates = self.data[:, self.covariate_columns].astype(np.float32)
        treatments = self.data[:, self.treatment_column].astype(np.float32)
        outcomes = self.data[:, self.outcome_column].astype(np.float32)
        dataset = TensorDataset(
            torch.from_numpy(covariates),
            torch.from_numpy(treatments).reshape(-1, 1),
            torch.from_numpy(outcomes).reshape(-1, 1),
        )
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


class IHDPDataset(ATEDataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        super().__init__(data, treatment_column, outcome_column, covariate_columns)

    @classmethod
    def load_replication(cls, replication_id):
        path = "datasets/ihdp_replications/ihdp_" + str(replication_id) + ".csv"
        data = np.loadtxt(path)
        return cls(data=data, treatment_column=0, outcome_column=1, covariate_columns=[i + 5 for i in range(25)])
