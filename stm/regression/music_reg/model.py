import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from stm.modules.net_modules import MusicTaggingTransformer
from stm.constants import (DATASET, AUDIOSET_TAGS)

class MusicModel(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.feature_extractor = self._load_pretrained(pretrained_path = pretrained_path)
        self.feature_dim = 64
        self.loss_fn = nn.MSELoss()
        self.feature_extractor.train()
        self.n_class = 2
        self.mlp_head = nn.Linear(self.feature_dim, self.n_class)
        self.to_latent = nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x,y):
        feature = self.feature_extractor(x)
        feature = self.to_latent(feature[:, 0])
        feature = self.dropout(feature)
        logit = self.mlp_head(feature)
        loss = self.loss_fn(logit, y)
        return loss

    def infer(self, x):
        feature = self.feature_extractor(x)
        feature = self.to_latent(feature[:, 0])
        feature = self.dropout(feature)
        logit = self.mlp_head(feature)
        return logit

    def extractor(self, wav):
        feature = self.feature_extractor(wav)
        feature = self.to_latent(feature[:, 0])
        return feature

    def _load_pretrained(self, pretrained_path):
        pretrained = torch.load(pretrained_path)
        student_ckpt = {k[8:]: v for k, v in pretrained.items() if (k[:7] != 'teacher')}
        model = MusicTaggingTransformer(conv_ndim=128, attention_ndim=64, n_seq_cls=50)
        model.load_state_dict(student_ckpt)
        return nn.Sequential(*list(model.children())[:-3]) # to_latent