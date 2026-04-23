import torch
from abc import ABC, abstractmethod


class MomentFunctional(ABC):
    @abstractmethod
    def __call__(self, forward_fn, covariates, treatment):
        pass


class AverageTreatmentEffect(MomentFunctional):
    def __call__(self, forward_fn, covariates, treatment):
        return forward_fn(covariates, torch.ones_like(treatment)) - forward_fn(covariates, torch.zeros_like(treatment))
