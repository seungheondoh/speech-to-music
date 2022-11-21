import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative, size_average=True):
        cosine_positive = nn.CosineSimilarity(dim=-1)(anchor, positive)
        cosine_negative = nn.CosineSimilarity(dim=-1)(anchor, negative)
        losses = self.relu(self.margin - cosine_positive + cosine_negative)
        return losses.mean()

#######################################################################################################################
# Code is based on the Blackbox Combinatorial Solvers (https://github.com/martius-lab/blackbox-backprop) implementation
# from https://github.com/martius-lab/blackbox-backprop by Marin Vlastelica et al.
#######################################################################################################################

def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]


class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None


def batchwise_ranking_regularizer(features, targets, lambda_val):
    # Reduce ties and boost relative representation of infrequent labels
    unique_batch = torch.unique(targets, dim=1)
    unique_pred = []
    for pred, target, unique_item in zip(features, targets, unique_batch):
        indices = torch.stack([random.choice((target==i).nonzero()[0]) for i in unique_item])
        unique_pred.append(pred[indices])
    unique_pred = torch.stack(unique_pred)

    label_ranks = rank_normalised(unique_batch)
    feature_ranks = TrueRanker.apply(unique_pred, lambda_val)
    emo_loss = F.mse_loss(feature_ranks, label_ranks)
    return emo_loss

def batchwise_emotion_regularizer(features, targets):
    # Reduce ties and boost relative representation of infrequent labels
    unique_batch = torch.unique(targets, dim=1)
    unique_pred = []
    for pred, target, unique_item in zip(features, targets, unique_batch):
        indices = torch.stack([random.choice((target==i).nonzero()[0]) for i in unique_item])
        unique_pred.append(pred[indices])
    emotion_feature = torch.stack(unique_pred)
    emo_loss = F.mse_loss(emotion_feature, unique_batch)
    return emo_loss