import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class ATEDataset:
    def __init__(self, data, treatment_column, outcome_column, covariate_columns, truth=None, folds=None):
        self.data = data
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column
        self.covariate_columns = covariate_columns
        self.truth = truth
        self.folds = folds

    def split_into_train_and_validation_sets(self, train_size):
        train_data, test_data = train_test_split(
            self.data, train_size=train_size, stratify=self.data[:, self.treatment_column]
        )
        return (
            ATEDataset(
                data=train_data,
                treatment_column=self.treatment_column,
                outcome_column=self.outcome_column,
                covariate_columns=self.covariate_columns,
                truth=self.truth,
            ),
            ATEDataset(
                data=test_data,
                treatment_column=self.treatment_column,
                outcome_column=self.outcome_column,
                covariate_columns=self.covariate_columns,
                truth=self.truth,
            ),
        )

    def create_dataloader(self, batch_size):
        dataset = TensorDataset(self.covariates_tensor(), self.treatments_tensor(), self.outcomes_tensor())
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    def outcomes_tensor(self):
        outcomes = self.data[:, self.outcome_column].astype(np.float32)
        return torch.from_numpy(outcomes).reshape(-1, 1)

    def treatments_tensor(self):
        treatments = self.data[:, self.treatment_column].astype(np.float32)
        return torch.from_numpy(treatments).reshape(-1, 1)

    def covariates_tensor(self):
        covariates = self.data[:, self.covariate_columns].astype(np.float32)
        return torch.from_numpy(covariates)

    def create_folds(self, n_folds):
        number_of_samples = self.data.shape[0]
        indices = np.arange(number_of_samples, dtype=int)
        treated_indices = indices[self.data[:, self.treatment_column] == 1]
        control_indices = indices[self.data[:, self.treatment_column] == 0]
        np.random.shuffle(treated_indices)
        np.random.shuffle(control_indices)
        treated_fold_indices = np.array_split(treated_indices, n_folds)
        control_fold_indices = np.array_split(control_indices, n_folds)
        self.folds = [
            np.concat((treated, control)) for treated, control in zip(treated_fold_indices, control_fold_indices)
        ]

    def get_fit_and_test_folds(self, test_fold):
        fit_folds = [self.folds[i] for i in range(len(self.folds)) if i != test_fold]
        fit_fold_indices = np.concat(fit_folds)
        test_fold_indices = self.folds[test_fold]
        return (
            ATEDataset(
                data=self.data[fit_fold_indices, :],
                treatment_column=self.treatment_column,
                outcome_column=self.outcome_column,
                covariate_columns=self.covariate_columns,
                truth=self.truth,
            ),
            ATEDataset(
                data=self.data[test_fold_indices, :],
                treatment_column=self.treatment_column,
                outcome_column=self.outcome_column,
                covariate_columns=self.covariate_columns,
                truth=self.truth,
            ),
        )


class IHDPDataset(ATEDataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        super().__init__(data, treatment_column, outcome_column, covariate_columns)
        self.truth = np.mean(self.data[:, 4] - self.data[:, 3])

    @classmethod
    def load_replication(cls, replication_id):
        path = "datasets/ihdp_replications/ihdp_" + str(replication_id) + ".csv"
        data = np.loadtxt(path)
        return cls(data=data, treatment_column=0, outcome_column=1, covariate_columns=[i + 5 for i in range(25)])
