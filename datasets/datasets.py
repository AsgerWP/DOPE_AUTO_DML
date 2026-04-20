import numpy as np


class ATEDataset:
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        self.data = data
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column
        self.covariate_columns = covariate_columns


class IHDPDataset(ATEDataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        super().__init__(data, treatment_column, outcome_column, covariate_columns)

    @classmethod
    def load_replication(cls, replication_id):
        path = 'datasets/ihdp_replications/ihdp_' + str(replication_id) + '.csv'
        data = np.loadtxt(path)
        return cls(data=data, treatment_column=0, outcome_column=1, covariate_columns=[i + 5 for i in range(25)])
