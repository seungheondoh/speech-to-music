import torch
import torch.nn as nn
from stm.constants import (HIKIA_TAGS, IEMOCAP_TAGS, RAVDESS_TAGS)
from transformers import Wav2Vec2Model, DistilBertForSequenceClassification

class AudioModel(nn.Module):
    def __init__(self, dataset_type, freeze_type):
        super().__init__()
        self.dataset_type = dataset_type
        self.wav2vec2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        self.feature_dim = 768
        self.projector = nn.Linear(self.feature_dim, self.feature_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        if dataset_type == "IEMOCAP":
            self.num_class = len(IEMOCAP_TAGS)
        elif dataset_type == "HIKIA":
            self.num_class = len(HIKIA_TAGS)
        elif dataset_type == "RAVDESS":
            self.num_class = len(RAVDESS_TAGS)
        self.classifier = nn.Linear(self.feature_dim, self.num_class)
        self.dropout = nn.Dropout(0.5)
        if freeze_type == "feature":
            self.wav2vec2.train()
            self.wav2vec2.feature_extractor._freeze_parameters()
        else:
            print("not freeze params")

    def forward(self, wav, mask, labels):
        feature = self.wav2vec2(
            wav,
            attention_mask=mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
            )
        last_hidden = feature['last_hidden_state']
        last_hidden = self.projector(last_hidden) # use last hidden layer
        projection_feature = last_hidden.mean(1,False) # average pooling
        dropout_feature = self.dropout(projection_feature)
        prediction = self.classifier(dropout_feature)
        loss = self.loss_fn(prediction, labels)
        return prediction, loss
    
    def infer(self, wav, mask=None):
        feature = self.wav2vec2(
            wav,
            attention_mask=mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
            )
        last_hidden = feature['last_hidden_state']
        last_hidden = self.projector(last_hidden) # use last hidden layer
        projection_feature = last_hidden.mean(1,False) # average pooling
        dropout_feature = self.dropout(projection_feature)
        prediction = self.classifier(dropout_feature)
        return prediction

    def pooling_extractor(self, wav, mask=None):
        feature = self.wav2vec2(
            wav,
            attention_mask=mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
            )
        last_hidden = feature['last_hidden_state']
        last_hidden = self.projector(last_hidden) # use last hidden layer
        projection_feature = last_hidden.mean(1,False) # average pooling
        return projection_feature
    
    def sequence_extractor(self, wav, mask=None):
        feature = self.wav2vec2(
            wav,
            attention_mask=mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
            )
        last_hidden = feature['last_hidden_state']
        last_hidden = self.projector(last_hidden) # use last hidden layer
        return last_hidden

class TextModel(nn.Module):
    def __init__(self, dataset_type, freeze_type=None):
        super().__init__()
        if dataset_type == "IEMOCAP":
            self.num_class = len(IEMOCAP_TAGS)
        elif dataset_type == "HIKIA":
            self.num_class = len(HIKIA_TAGS)
        elif dataset_type == "RAVDESS":
            self.num_class = len(RAVDESS_TAGS)
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', problem_type="multi_label_classification", num_labels=self.num_class)
        self.feature_dim = 768

    def forward(self, token, mask, labels):
        output = self.bert(token, mask, labels=labels)
        prediction = output['logits']
        loss = output['loss']
        return prediction, loss

    def infer(self, token, mask=None):
        output = self.bert(token, mask)
        prediction = output['logits']
        return prediction

    def pooling_extractor(self, token, mask):
        distilbert_output = self.bert.distilbert(input_ids=token, attention_mask=mask)
        hidden_state = distilbert_output[0]  # (b, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (b, dim)
        return pooled_output

    def sequence_extractor(self, token, mask):
        distilbert_output = self.bert.distilbert(input_ids=token, attention_mask=mask)
        hidden_state = distilbert_output[0]  # (b, seq_len, dim)
        return hidden_state
