import copy

import torch
from torch import nn

from models.utils import MLP, TBranch, SBranch


class DOPENeuralNet(nn.Module):
    def __init__(
        self,
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
        self.shared_trunk = MLP(
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

    def freeze_shared_trunk(self):
        for param in self.shared_trunk.parameters():
            param.requires_grad = False

    def get_outcome_mse_loss(self, batch):
        covariates, treatment, outcome = batch
        representation = self.shared_trunk(covariates)
        outcome_prediction = self.outcome_branch(representation, treatment)
        return nn.functional.mse_loss(outcome_prediction, outcome)

    def get_riesz_loss(self, batch):
        covariates, treatment, _ = batch
        representation = self.shared_trunk(covariates)
        riesz_prediction = self.riesz_branch(representation, treatment)
        riesz_treatment = self.riesz_branch(representation, torch.ones_like(treatment))
        riesz_control = self.riesz_branch(representation, torch.zeros_like(treatment))
        return (riesz_prediction**2 - 2 * (riesz_treatment - riesz_control)).mean()

    def get_estimates(self, data):
        device = next(self.parameters()).device
        covariates = data.covariates_tensor().to(device)
        treatment = data.treatments_tensor().to(device)
        outcome = data.outcomes_tensor().to(device)
        representation = self.shared_trunk(covariates)
        outcome_prediction = self.outcome_branch(representation, treatment)
        riesz_prediction = self.riesz_branch(representation, treatment)
        treated_outcome_prediction = self.outcome_branch(representation, torch.ones_like(treatment))
        control_outcome_prediction = self.outcome_branch(representation, torch.zeros_like(treatment))
        plugin_terms = treated_outcome_prediction - control_outcome_prediction
        correction_terms = riesz_prediction * (outcome - outcome_prediction)
        dr_terms = plugin_terms + correction_terms
        return {"point_estimate": dr_terms.mean().item(), "var_estimate": dr_terms.var().item()}

    def fit_riesz_branch(self, batch_size, data, epochs, lambda_lasso: int, lr, patience, weight_decay):
        self._fit(data, self.get_riesz_loss, lr, weight_decay, batch_size, epochs, patience, lambda_lasso)

    def fit_outcome_branch(self, batch_size, data, epochs, lambda_lasso: int, lr, patience, weight_decay):
        self._fit(data, self.get_outcome_mse_loss, lr, weight_decay, batch_size, epochs, patience, lambda_lasso)

    def _fit(self, data, loss_fn, lr, weight_decay, batch_size, epochs, patience, lambda_lasso=0):
        device = next(self.parameters()).device
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
                batch = tuple(x.to(device) for x in batch)
                loss = loss_fn(batch)
                if lambda_lasso > 0:
                    final_layer_weights = self.shared_trunk.layers[-1].weight
                    lasso_loss = lambda_lasso * torch.norm(final_layer_weights, dim=1).sum()
                    loss += lasso_loss
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                val_loss = loss_fn(
                    (
                        val_data.covariates_tensor().to(device),
                        val_data.treatments_tensor().to(device),
                        val_data.outcomes_tensor().to(device),
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
