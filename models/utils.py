import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation, dropout_prob, activation_after_final_layer):
        super().__init__()
        layers = []
        input_sizes = [input_size] + hidden_sizes
        output_sizes = hidden_sizes + [output_size]
        for i, (in_size, out_size) in enumerate(zip(input_sizes, output_sizes)):
            layers.append(nn.Linear(in_size, out_size))
            if i + 1 < len(input_sizes):
                layers.append(activation())
                if dropout_prob > 0:
                    layers.append(nn.Dropout(dropout_prob))
        if activation_after_final_layer:
            layers.append(activation())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class THead(nn.Module):
    def __init__(self, representation_size, hidden_sizes, activation, dropout_prob):
        super().__init__()
        self.t_layers = MLP(
            input_size=representation_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=False,
        )
        self.c_layers = MLP(
            input_size=representation_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=False,
        )

    def forward(self, representation, treatment):
        return self.t_layers(representation) * treatment + self.c_layers(representation) * (1 - treatment)


class SHead(nn.Module):
    def __init__(self, representation_size, hidden_sizes, activation, dropout_prob):
        super().__init__()
        self.layers = MLP(
            input_size=representation_size + 1,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=False,
        )

    def forward(self, representation, treatment):
        x = torch.cat([representation, treatment], dim=1)
        return self.layers(x)
