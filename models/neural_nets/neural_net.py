import copy
import torch
from torch import nn
from abc import ABC, abstractmethod

from models.neural_nets.functionals import MomentFunctional


class NeuralNetwork(nn.Module, ABC):
    def __init__(self, moment_functional: MomentFunctional):
        super().__init__()
        self.moment_functional = moment_functional

    @abstractmethod
    def outcome_forward(self, covariates, treatment):
        pass

    @abstractmethod
    def riesz_forward(self, covariates, treatment):
        pass

    def get_estimates(self, data):
        device = next(self.parameters()).device
        covariates = data.covariates_tensor().to(device)
        treatment = data.treatments_tensor().to(device)
        outcome = data.outcomes_tensor().to(device)

        plugin_terms = self.moment_functional(self.outcome_forward, covariates, treatment)
        correction_terms = self.riesz_forward(covariates, treatment) * (
            outcome - self.outcome_forward(covariates, treatment)
        )
        dr_terms = plugin_terms + correction_terms

        return {"point_estimate": dr_terms.mean().item(), "var_estimate": dr_terms.var().item()}

    def _cv(self, data, n_folds, loss_fn, trunk, lr, weight_decay, batch_size, epochs, patience, lambda_lasso):
        data.create_folds(n_folds=n_folds)
        cv_results = []
        for i in range(n_folds):
            fit_fold, test_fold = data.get_fit_and_test_folds(test_fold=i)
            self._fit(
                data=fit_fold,
                loss_fn=loss_fn,
                trunk=trunk,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                lambda_lasso=lambda_lasso,
            )
            self.eval()
            cv_results.append(
                loss_fn(
                    (
                        test_fold.covariates_tensor(),
                        test_fold.treatments_tensor(),
                        test_fold.outcomes_tensor(),
                    )
                ).item()
            )
            self._reset_parameters()
        return sum(cv_results) / len(cv_results)

    def _reset_parameters(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters") and m is not self:
                m.reset_parameters()

    def _fit(self, data, loss_fn, trunk, lr, weight_decay, batch_size, epochs, patience, lambda_lasso):
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
                loss = loss_fn(batch)
                if lambda_lasso > 0:
                    final_layer_weights = trunk.layers[-1].weight
                    lasso_loss = lambda_lasso * torch.norm(final_layer_weights, dim=1).sum()
                    loss += lasso_loss
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
