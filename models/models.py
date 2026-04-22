from torch import nn

from models.utils import SharedTrunk, THead, SHead


class DOPENeuralNet(nn.Module):
    def __init__(
        self,
        n_covariates,
        shared_hidden_layers,
        not_shared_hidden_layers,
        representation_size,
        activation,
        branch_type,
        dropout_prob=0,
    ):
        super().__init__()
        self.shared_trunk = SharedTrunk(
            n_covariates, shared_hidden_layers, representation_size, activation, dropout_prob, False
        )
        if branch_type == "T":
            self.outcome_head = THead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
            self.riesz_head = THead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
        elif branch_type == "S":
            self.outcome_head = SHead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
            self.riesz_head = SHead(representation_size, not_shared_hidden_layers, activation, dropout_prob)
        else:
            raise ValueError("Invalid branch type. Must be 'T' or 'S'.")

    def freeze_shared_trunk(self):
        for param in self.shared_trunk.parameters():
            param.requires_grad = False
