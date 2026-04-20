class ATEDataset:
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        self.data = data
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column
        self.covariate_columns = covariate_columns


class IHDPDataset(ATEDataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        super().__init__(data, treatment_column, outcome_column, covariate_columns)