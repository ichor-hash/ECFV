import torch
import torch.nn as nn

class FraudMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

import rtdl_revisiting_models as rtdl

def create_ft_transformer() -> rtdl.FTTransformer:
    model = rtdl.FTTransformer(
        n_cont_features=30,
        cat_cardinalities=[],
        d_block=128,
        d_out=1,
        n_blocks=3,
        attention_n_heads=8,
        ffn_d_hidden_multiplier=4.0,
        attention_dropout=0.2,
        ffn_dropout=0.1,
        residual_dropout=0.0
    )
    return model
