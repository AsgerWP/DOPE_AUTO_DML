import copy

import torch
from torch import nn

from models.utils import MLP, TBranch


class RieszNet(nn.Module):
    def __init__(
        self,
        n_covariates,
        shared_hidden_layers,
        not_shared_hidden_layers,
        activation,
        loss_weights,
        dropout_prob=0,
    ):
        super().__init__()
        self.shared_trunk = MLP(
            input_size=n_covariates + 1,
            hidden_sizes=shared_hidden_layers[:-1],
            output_size=shared_hidden_layers[-1],
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=True,
        )
        self.outcome_branch = TBranch(
            representation_size=shared_hidden_layers[-1],
            hidden_sizes=not_shared_hidden_layers,
            activation=activation,
            dropout_prob=dropout_prob,
        )
        self.riesz_branch = nn.Linear(shared_hidden_layers[-1], 1)
        self.loss_weights = loss_weights
        self.epsilon = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def get_riesz_net_loss(self, batch):
        covariates, treatment, outcome = batch

        representation = self.shared_trunk(torch.cat((covariates, treatment), dim=1))
        treated_representation = self.shared_trunk(torch.cat((covariates, torch.ones_like(treatment)), dim=1))
        control_representation = self.shared_trunk(torch.cat((covariates, torch.zeros_like(treatment)), dim=1))

        outcome_prediction = self.outcome_branch(representation, treatment)
        riesz_prediction = self.riesz_branch(representation)
        treated_riesz_prediction = self.riesz_branch(treated_representation)
        control_riesz_prediction = self.riesz_branch(control_representation)

        riesz_loss = (riesz_prediction**2 - 2 * (treated_riesz_prediction - control_riesz_prediction)).mean()
        outcome_loss = nn.functional.mse_loss(outcome_prediction, outcome)
        tmle_loss = nn.functional.mse_loss(outcome_prediction + self.epsilon * riesz_prediction, outcome)
        return (
            self.loss_weights["riesz"] * riesz_loss
            + self.loss_weights["outcome"] * outcome_loss
            + self.loss_weights["tmle"] * tmle_loss
        )

    def get_estimates(self, data):
        device = next(self.parameters()).device
        covariates = data.covariates_tensor().to(device)
        treatment = data.treatments_tensor().to(device)
        outcome = data.outcomes_tensor().to(device)

        representation = self.shared_trunk(torch.cat((covariates, treatment), dim=1))
        treated_representation = self.shared_trunk(torch.cat((covariates, torch.ones_like(treatment)), dim=1))
        control_representation = self.shared_trunk(torch.cat((covariates, torch.zeros_like(treatment)), dim=1))

        riesz_prediction = self.riesz_branch(representation)
        treated_riesz_prediction = self.riesz_branch(treated_representation)
        control_riesz_prediction = self.riesz_branch(control_representation)

        outcome_prediction = self.outcome_branch(representation, treatment) + self.epsilon * riesz_prediction
        treated_outcome_prediction = (
            self.outcome_branch(treated_representation, torch.ones_like(treatment))
            + self.epsilon * treated_riesz_prediction
        )
        control_outcome_prediction = (
            self.outcome_branch(control_representation, torch.zeros_like(treatment))
            + self.epsilon * control_riesz_prediction
        )

        plugin_terms = treated_outcome_prediction - control_outcome_prediction
        correction_terms = riesz_prediction * (outcome - outcome_prediction)
        dr_terms = plugin_terms + correction_terms
        return {"point_estimate": dr_terms.mean().item(), "var_estimate": dr_terms.var().item()}

    def fit(self, data, lr, weight_decay, batch_size, epochs, patience):
        device = next(self.parameters()).device
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
                batch = tuple(x.to(device) for x in batch)
                loss = self.get_riesz_net_loss(batch)
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                val_loss = self.get_riesz_net_loss(
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
