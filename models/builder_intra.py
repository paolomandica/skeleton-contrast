# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

from .GRU import BIGRU
from .AGCN import Model as AGCN
from .HCN import HCN
from .STGCN import Model as STGCN
from .utils.misc import *


class SimSiam(nn.Module):
    """
    Build a SimSiam model with: a query encoder and a key encoder.
    """

    def __init__(self, skeleton_representation, args_bi_gru, args_agcn, dim=128, pred_dim=512, mlp=False):
        """
        dim: feature dimension (default: 128)
        pred_dim: hidden dimension of the predictor (default: 128)
        args_bi_gru: model parameters BIGRU
        args_agcn: model parameters AGCN
        T: softmax temperature (default: 0.07)
        """
        super(SimSiam, self).__init__()

        # create encoders

        if skeleton_representation == 'seq-based':
            print("Creating BIGRU encoders")
            self.encoder_q = BIGRU(**args_bi_gru)
            self.encoder_k = BIGRU(**args_bi_gru)
            weights_init_gru(self.encoder_q)
            weights_init_gru(self.encoder_k)

        elif skeleton_representation == 'graph-based':
            self.encoder_q = AGCN(**args_agcn)
            self.encoder_k = AGCN(**args_agcn)
            # self.encoder_q = STGCN(in_channels=3, hidden_channels=64,
            #                        hidden_dim=256, num_class=args_agcn['num_class'],
            #                        graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            #                        edge_importance_weighting=True)
            # self.encoder_k = STGCN(in_channels=3, hidden_channels=64,
            #                        hidden_dim=256, num_class=args_agcn['num_class'],
            #                        graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            #                        edge_importance_weighting=True)

            # self.encoder_q.apply(weights_init_gcn)
            # self.encoder_k.apply(weights_init_gcn)

        # build 1-layer projection heads
        if mlp:
            dim_mlp_q = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp_q, dim_mlp_q), nn.ReLU(), self.encoder_q.fc)

            dim_mlp_k = self.encoder_k.fc.weight.shape[1]
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp_k, dim_mlp_k), nn.ReLU(), self.encoder_k.fc)

        # build a 3-layer projector (like in original SimSiam)
        if False:
            prev_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),  # second layer
                                            self.encoder.fc,
                                            nn.BatchNorm1d(dim, affine=False))  # output layer
            # hack: not use bias as it is followed by BN
            self.encoder.fc[6].bias.requires_grad = False

        # build 2-layer predictors
        self.predictor = nn.Sequential(nn.Linear(dim, dim//2, bias=False),
                                       nn.BatchNorm1d(dim//2),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(dim//2, dim))  # output layer

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * 0.999 + param_q.data * (1. - 0.999)

    def forward(self, im_q, im_k,):
        """
        Input:
            im_q: a batch of queries
            im_k: a batch of keys
        Output:
            logits, targets
        """

        # compute query features
        p = self.encoder_q(im_q)  # queries: NxC
        p = nn.functional.normalize(p, dim=1)
        p = self.predictor(p)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            z = self.encoder_k(im_k)	  # keys: NxC
            z = nn.functional.normalize(z, dim=1)

        return p, z


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, skeleton_representation, args_bi_gru, args_agcn, args_hcn, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        skeleton_representation:  input skeleton representation (graph-based, seq-based or image-based)
        args_bi_gru: model parameters BIGRU
        args_agcn: model parameters AGCN
        args_hcn: model parameters of HCN
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 16384)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        mlp = mlp
        print(" moco parameters", K, m, T, mlp)

        if skeleton_representation == 'seq-based':
            self.encoder_q = BIGRU(**args_bi_gru)
            self.encoder_k = BIGRU(**args_bi_gru)
            weights_init_gru(self.encoder_q)
            weights_init_gru(self.encoder_k)
        elif skeleton_representation == 'graph-based':
            self.encoder_q = AGCN(**args_agcn)
            self.encoder_k = AGCN(**args_agcn)
        elif skeleton_representation == 'image-based':
            self.encoder_q = HCN(**args_hcn)
            self.encoder_k = HCN(**args_hcn)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k,):
        """
        Input:
            im_q: a batch of queries
            im_k: a batch of keys
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)	  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
