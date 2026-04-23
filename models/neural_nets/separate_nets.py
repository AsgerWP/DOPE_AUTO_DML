import copy

import torch
from torch import nn

from models.neural_nets.utils import MLP, TBranch, SBranch


class SeparateNeuralNets(nn.Module):
    def __init__(
        self,
        moment_functional,
        n_covariates,
        shared_hidden_layers,
        not_shared_hidden_layers,
        activation,
        outcome_branch_type,
        riesz_branch_type,
        activation_after_final_shared_layer,
        dropout_prob=0,
    ):
        super().__init__()
        self.moment_functional = moment_functional
        self.outcome_trunk = MLP(
            input_size=n_covariates,
            hidden_sizes=shared_hidden_layers[:-1],
            output_size=shared_hidden_layers[-1],
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=activation_after_final_shared_layer,
        )
        self.riesz_trunk = MLP(
            input_size=n_covariates,
            hidden_sizes=shared_hidden_layers[:-1],
            output_size=shared_hidden_layers[-1],
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=activation_after_final_shared_layer,
        )
        if outcome_branch_type == "T":
            self.outcome_branch = TBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        elif outcome_branch_type == "S":
            self.outcome_branch = SBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        else:
            raise ValueError("Invalid branch type. Must be 'T' or 'S'.")

        if riesz_branch_type == "T":
            self.riesz_branch = TBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        elif riesz_branch_type == "S":
            self.riesz_branch = SBranch(
                representation_size=shared_hidden_layers[-1],
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        else:
            raise ValueError("Invalid branch type. Must be 'T' or 'S'.")

    def outcome_forward(self, covariates, treatment):
        device = next(self.parameters()).device
        covariates = covariates.to(device)
        treatment = treatment.to(device)
        return self.outcome_branch(self.outcome_trunk(covariates), treatment)

    def riesz_forward(self, covariates, treatment):
        device = next(self.parameters()).device
        covariates = covariates.to(device)
        treatment = treatment.to(device)
        return self.riesz_branch(self.riesz_trunk(covariates), treatment)

    def get_outcome_mse_loss(self, batch):
        covariates, treatment, outcome = batch
        return nn.functional.mse_loss(self.outcome_forward(covariates, treatment), outcome)

    def get_riesz_loss(self, batch):
        covariates, treatment, _ = batch
        return (
            self.riesz_forward(covariates, treatment) ** 2
            - 2 * self.moment_functional(self.riesz_forward, covariates, treatment)
        ).mean()

    def get_estimates(self, data):
        covariates = data.covariates_tensor()
        treatment = data.treatments_tensor()
        outcome = data.outcomes_tensor()

        plugin_terms = self.moment_functional(self.outcome_forward, covariates, treatment)
        correction_terms = self.riesz_forward(covariates, treatment) * (
            outcome - self.outcome_forward(covariates, treatment)
        )
        dr_terms = plugin_terms + correction_terms

        return {"point_estimate": dr_terms.mean().item(), "var_estimate": dr_terms.var().item()}

    def fit_riesz_branch(self, batch_size, data, epochs, lr, patience, weight_decay):
        self._fit(data, self.get_riesz_loss, lr, weight_decay, batch_size, epochs, patience)

    def fit_outcome_branch(self, batch_size, data, epochs, lr, patience, weight_decay):
        self._fit(data, self.get_outcome_mse_loss, lr, weight_decay, batch_size, epochs, patience)

    def _fit(self, data, loss_fn, lr, weight_decay, batch_size, epochs, patience):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=1e-3,
            threshold_mode="rel",
            cooldown=2,
            min_lr=1e-6,
        )
        train_data, val_data = data.split_into_train_and_validation_sets(train_size=0.8)
        train_loader = train_data.create_dataloader(batch_size=batch_size)
        best = 1e6
        counter = 0
        best_state = copy.deepcopy(self.state_dict())
        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(batch)
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                val_loss = loss_fn(
                    (
                        val_data.covariates_tensor(),
                        val_data.treatments_tensor(),
                        val_data.outcomes_tensor(),
                    )
                )
                scheduler.step(val_loss)
                if val_loss.item() < best:
                    best = val_loss.item()
                    counter = 0
                    best_state = copy.deepcopy(self.state_dict())
                else:
                    counter += 1
                    if counter == patience:
                        self.load_state_dict(best_state)
                        break
