import numpy as np
from sklearn.model_selection import train_test_split


class ATEDataset:
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        self.data = data
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column
        self.covariate_columns = covariate_columns

    def split_into_train_and_test_sets(self, train_size=0.8):
        train_data, test_data = train_test_split(
            self.data,
            train_size=train_size,
            stratify=self.data[:, self.treatment_column],
        )

        return (
            ATEDataset(
                train_data,
                self.outcome_column,
                self.treatment_column,
                self.covariate_columns,
            ),
            ATEDataset(
                test_data,
                self.outcome_column,
                self.treatment_column,
                self.covariate_columns,
            ),
        )


class IHDPDataset(ATEDataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        super().__init__(data, treatment_column, outcome_column, covariate_columns)

    @classmethod
    def load_replication(cls, replication_id):
        path = "datasets/ihdp_replications/ihdp_" + str(replication_id) + ".csv"
        data = np.loadtxt(path)
        return cls(
            data=data,
            treatment_column=0,
            outcome_column=1,
            covariate_columns=[i + 5 for i in range(25)],
        )
