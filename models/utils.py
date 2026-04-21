from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation, activation_after_final_layer, dropout_prob=0):
        super().__init__()
        layers = []
        input_sizes = [input_size] + hidden_sizes
        output_sizes = hidden_sizes + [output_size]
        for i, (in_size, out_size) in enumerate(zip(input_sizes, output_sizes)):
            layers.append(nn.Linear(in_size, out_size))
            if i + 1 < len(input_sizes) or activation_after_final_layer:
                layers.append(activation())
                if dropout_prob > 0:
                    layers.append(nn.Dropout(dropout_prob))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
