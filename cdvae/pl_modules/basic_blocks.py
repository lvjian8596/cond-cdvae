import torch.nn as nn


def build_mlp(
    in_dim, hidden_dim, fc_num_layers, out_dim, dropout=False, drop_ratio=0.5
):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        if dropout:
            mods += [nn.Dropout(drop_ratio)]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)
