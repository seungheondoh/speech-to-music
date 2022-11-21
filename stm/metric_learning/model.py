import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, DistilBertModel
from stm.constants import (DATASET)
from stm.modules.loss_modules import TripletLoss, batchwise_ranking_regularizer, batchwise_emotion_regularizer


class EmbModel(nn.Module):
    def __init__(self, modality, framework, emo_lambda, is_ranksim=False):
        super().__init__()
        self.modality = modality
        self.framework = framework
        self.music_feature_dim = 64
        self.emo_lambda = emo_lambda
        self.interpolation_lambda = 2.0 # for rank sim
        self.is_ranksim = is_ranksim
        self.speech_audio_feature_dim = 768
        self.speech_text_feature_dim = 768
        self.speech_fusion_feature_dim = self.speech_audio_feature_dim + self.speech_text_feature_dim
        self.tag_feature_dim = 300
        self.projection_dim = 64
        if self.modality == "audio":
            self.speech_audio_mlp = self._build_mlp(num_layers=2, input_dim=self.speech_audio_feature_dim, mlp_dim=self.speech_audio_feature_dim, output_dim=self.projection_dim)
        elif self.modality == "text":
            self.speech_text_mlp = self._build_mlp(num_layers=2, input_dim=self.speech_text_feature_dim, mlp_dim=self.speech_text_feature_dim, output_dim=self.projection_dim)
        elif self.modality == "fusion":
            self.speech_fusion_mlp = self._build_mlp(num_layers=2, input_dim=self.speech_fusion_feature_dim, mlp_dim=self.speech_fusion_feature_dim, output_dim=self.projection_dim)
        self.music_mlp = self._build_mlp(num_layers=2, input_dim=self.music_feature_dim, mlp_dim=self.music_feature_dim, output_dim=self.projection_dim)
        if "sp" in self.framework:
            self.tag_mlp = self._build_mlp(num_layers=2, input_dim=self.tag_feature_dim, mlp_dim=self.tag_feature_dim, output_dim=self.projection_dim)
        self.to_latent = nn.Identity()
        self.triplet_loss = TripletLoss(margin=0.4)
        self.emosim_loss = nn.MSELoss()
        self.dropout = nn.Dropout(0.1)
    
    def speech_to_emb(self, speech_emb):
        if self.modality == "audio":
            speech_emb = self.dropout(speech_emb)
            speech_emb = self.speech_audio_mlp(speech_emb)
        elif self.modality == "text":
            speech_emb = self.dropout(speech_emb)
            speech_emb = self.speech_text_mlp(speech_emb)
        elif self.modality == "fusion":
            speech_emb = self.dropout(speech_emb)
            speech_emb = self.speech_fusion_mlp(speech_emb)
        else:
            speech_emb = None
        return speech_emb
        
    def music_to_emb(self, music):
        music = self.dropout(music)
        return self.music_mlp(music)

    def tag_to_emb(self, tag):
        tag = self.dropout(tag)
        return self.tag_mlp(tag)

    def forward(self, s_emb, s_tag_emb, s_va, s_binary, m_emb, m_tag_emb, m_va, m_binary):
        # for stop gradient
        emosim_loss = 0.
        s_va = s_va.detach()
        m_va = m_va.detach()
        s_emb = self.speech_to_emb(s_emb)
        m_emb = self.music_to_emb(m_emb)
        stm_anchor, stm_pos, stm_neg = self.triplet_sampling(s_emb, m_emb, s_binary)
        stm_loss = self.triplet_loss(stm_anchor, stm_pos, stm_neg) 
        if self.framework == "triplet_sp":
            s_tag_emb = self.tag_to_emb(s_tag_emb)
            m_tag_emb = self.tag_to_emb(m_tag_emb)
            tts_anchor, tts_pos, tts_neg = self.triplet_sampling(s_tag_emb, s_emb, s_binary)
            ttm_anchor, ttm_pos, ttm_neg = self.triplet_sampling(m_tag_emb, m_emb, m_binary)
            tts_loss = self.triplet_loss(tts_anchor, tts_pos, tts_neg)
            ttm_loss = self.triplet_loss(ttm_anchor, ttm_pos, ttm_neg)
            loss = (0.4 * stm_loss) + (0.3 * tts_loss) + (0.3 * ttm_loss)
        elif self.framework == "triplet_soft":
            s_emb = nn.functional.normalize(s_emb, dim=1)
            m_emb = nn.functional.normalize(m_emb, dim=1)
            latent_sim = s_emb @ m_emb.T
            label_sim = 1 - torch.cdist(s_va, m_va, p=2)
            if self.is_ranksim:
                emosim_loss = self.emo_lambda * (batchwise_ranking_regularizer(latent_sim, label_sim, self.interpolation_lambda))
            else:
                emosim_loss = self.emo_lambda * batchwise_emotion_regularizer(latent_sim, label_sim)
            triplet_loss = stm_loss
            loss = triplet_loss + emosim_loss
        elif self.framework == "triplet_sp_soft":
            s_tag_emb = self.tag_to_emb(s_tag_emb)
            m_tag_emb = self.tag_to_emb(m_tag_emb)
            tts_anchor, tts_pos, tts_neg = self.triplet_sampling(s_tag_emb, s_emb, s_binary)
            ttm_anchor, ttm_pos, ttm_neg = self.triplet_sampling(m_tag_emb, m_emb, m_binary)
            tts_loss = self.triplet_loss(tts_anchor, tts_pos, tts_neg)
            ttm_loss = self.triplet_loss(ttm_anchor, ttm_pos, ttm_neg)
            triplet_loss = (0.4 * stm_loss) + (0.3 * tts_loss) + (0.3 * ttm_loss)
            s_emb = nn.functional.normalize(s_emb, dim=1)
            m_emb = nn.functional.normalize(m_emb, dim=1)
            label_sim = 1 - torch.cdist(s_va, m_va, p=2)
            latent_sim = s_emb @ m_emb.T
            if self.is_ranksim:
                emosim_loss = self.emo_lambda * (batchwise_ranking_regularizer(latent_sim, label_sim, self.interpolation_lambda))
            else:
                emosim_loss = self.emo_lambda * batchwise_emotion_regularizer(latent_sim, label_sim)
            loss = triplet_loss + emosim_loss
        else:
            loss = stm_loss
        return loss, stm_loss, emosim_loss

    # https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)

    def triplet_sampling(self, anchor_emb, positive_emb, binary, is_weighted=True):
        num_batch = len(anchor_emb)
        if is_weighted:
            # get distance weights
            anchor_norm = anchor_emb / anchor_emb.norm(dim=1)[:, None]
            positive_norm = positive_emb / positive_emb.norm(dim=1)[:, None]
            dot_sim = torch.matmul(anchor_norm, positive_norm.T)
            weights = (dot_sim + 1) / 2

            # masking
            mask = 1 - torch.matmul(binary, binary.T)
            masked_weights = weights * mask

            # sampling
            index_array = torch.arange(num_batch)
            negative_ix = [random.choices(index_array, weights=masked_weights[i], k=1)[0].item() for i in range(num_batch)]
            negative_emb = positive_emb[negative_ix]
        else:
            num_batch = len(anchor_emb)
            # masking
            mask = 1 - torch.matmul(binary, binary.T)
            # sampling
            index_array = torch.arange(num_batch)
            negative_ix = [random.choices(index_array, weights=mask[i], k=1)[0].item() for i in range(num_batch)]
            negative_emb = positive_emb[negative_ix]
        return anchor_emb, positive_emb, negative_emb
