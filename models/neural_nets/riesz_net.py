import copy

import torch
from torch import nn

from models.neural_nets.utils import MLP, TBranch, SBranch


class RieszNet(nn.Module):
    def __init__(
        self,
        moment_functional,
        outcome_branch_type,
        n_covariates,
        shared_hidden_layers,
        not_shared_hidden_layers,
        activation,
        loss_weights,
        dropout_prob=0,
    ):
        super().__init__()
        self.moment_functional = moment_functional
        self.shared_trunk = MLP(
            input_size=n_covariates + 1,
            hidden_sizes=shared_hidden_layers[:-1],
            output_size=shared_hidden_layers[-1],
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=True,
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

        self.riesz_branch = nn.Linear(shared_hidden_layers[-1], 1)
        self.loss_weights = loss_weights
        self.epsilon = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def outcome_forward(self, covariates, treatment):
        device = next(self.parameters()).device
        covariates = covariates.to(device)
        treatment = treatment.to(device)
        net_input = torch.cat((covariates, treatment), dim=1)
        return self.outcome_branch(self.shared_trunk(net_input), treatment)

    def riesz_forward(self, covariates, treatment):
        device = next(self.parameters()).device
        covariates = covariates.to(device)
        treatment = treatment.to(device)
        net_input = torch.cat((covariates, treatment), dim=1)
        return self.riesz_branch(self.shared_trunk(net_input.to(device)))

    def corrected_outcome_forward(self, covariates, treatment):
        return self.outcome_forward(covariates, treatment) + self.epsilon * self.riesz_forward(covariates, treatment)

    def get_riesz_net_loss(self, batch):
        covariates, treatment, outcome = batch

        outcome_loss = nn.functional.mse_loss(self.outcome_forward(covariates, treatment), outcome)
        riesz_loss = (
            self.riesz_forward(covariates, treatment) ** 2
            - 2 * self.moment_functional(self.riesz_forward, covariates, treatment)
        ).mean()
        tmle_loss = nn.functional.mse_loss(self.corrected_outcome_forward(covariates, treatment), outcome)

        return (
            self.loss_weights["riesz"] * riesz_loss
            + self.loss_weights["outcome"] * outcome_loss
            + self.loss_weights["tmle"] * tmle_loss
        )

    def get_estimates(self, data):
        covariates = data.covariates_tensor()
        treatment = data.treatments_tensor()
        outcome = data.outcomes_tensor()

        plugin_terms = self.moment_functional(self.corrected_outcome_forward, covariates, treatment)
        correction_terms = self.riesz_forward(covariates, treatment) * (
            outcome - self.corrected_outcome_forward(covariates, treatment)
        )
        dr_terms = plugin_terms + correction_terms
        return {"point_estimate": dr_terms.mean().item(), "var_estimate": dr_terms.var().item()}

    def fit(self, data, lr, weight_decay, batch_size, epochs, patience):
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name == "epsilon":
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.Adam(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
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
                loss = self.get_riesz_net_loss(batch)
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                val_loss = self.get_riesz_net_loss(
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
