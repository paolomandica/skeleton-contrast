import torch
import torch.nn as nn


def weights_init_gru(model):
    # initilize weight
    with torch.no_grad():
        for child in list(model.children()):
            print(child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('GRU weights initialization finished!')


def weights_init_gcn(layer):
    if isinstance(layer, nn.Conv2d):
        torch.nn.init.normal_(layer.weight, 1, 2.5)
        layer.bias.data.fill_(0.01)
