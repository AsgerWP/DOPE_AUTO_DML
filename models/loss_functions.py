import torch
import torch.nn as nn


def outcome_mse_loss(model, batch):
    covariates, treatment, outcome = batch
    representation = model.shared_trunk(covariates)
    outcome_prediction = model.outcome_head(representation, treatment)
    return nn.functional.mse_loss(outcome_prediction, outcome)


def riesz_loss(model, batch):
    covariates, treatment, _ = batch
    representation = model.shared_trunk(covariates)
    riesz_prediction = model.riesz_head(representation, treatment)
    riesz_treatment = model.riesz_head(representation, torch.ones_like(treatment))
    riesz_control = model.riesz_head(representation, torch.zeros_like(treatment))

    return (riesz_prediction**2 - 2 * (riesz_treatment - riesz_control)).mean()
