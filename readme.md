## Textless Speech-to-Music Retrieval Using Emotion Similarity
This is a PyTorch implementation of [Textless Speech-to-Music Retrieval Using Emotion Similarity](#) for multi-modal music representation learning. Check our [demo](https://seungheondoh.github.io/speech-to-music-demo/)

> [**Textless Speech-to-Music Retrieval Using Emotion Similarity**](#)   
> SeungHeon Doh, Minz Won, Keunwoo Choi, Juhan Nam   
> submitted ICASSP 2023   


**TL;DR**

- For textless music retrieval scenario (smart speaker), we introduce a framework that recommends music based on the emotions of speech. 
- We explore different speech representations and report their impact on different speech types including acted speech ([IEMOCAP](https://sail.usc.edu/iemocap/)), lexically-matched speech ([RAVDESS](https://zenodo.org/record/1188976)), and wake-up word speech ([HIKIA](https://zenodo.org/record/7091465))
- We also propose an emotion similarity regularization term in cross-domain retrieval tasks.

<p align = "center">
<img src = "https://i.imgur.com/uw5kvdn.png">
</p>

### Main Results
The following results are precision@5. See our paper for more results on MRR, nDCG@5. **Pre-trained models** and **configs** can be found at [Zenodo-Pretrained](https://zenodo.org/record/7341484).

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Modality</th>
    <th>IEMOCAP</th>
    <th>RAVDESS</th>
    <th>HIKIA</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Triplet (Basline)</td>
    <td>Text</td>
    <td>0.68±0.03</td>
    <td>0.11±0.05</td>
    <td>0.17±0.08</td>
  </tr>
  <tr>
    <td>Triplet + Structure Preserving</td>
    <td>Text</td>
    <td>0.68±0.02</td>
    <td>0.16±0.06</td>
    <td>0.16±0.10</td>
  </tr>
  <tr>
    <td>Triplet + EmoSim Regularization</td>
    <td>Text</td>
    <td>0.69±0.02</td>
    <td>0.18±0.05</td>
    <td>0.1±0.03</td>
  </tr>
  <tr>
    <td>Triplet (Basline)</td>
    <td>Audio</td>
    <td>0.67±0.04</td>
    <td>0.65±0.03</td>
    <td>0.73±0.06</td>
  </tr>
  <tr>
    <td>Triplet + Structure Preserving</td>
    <td>Audio</td>
    <td>0.65±0.02</td>
    <td>0.65±0.05</td>
    <td>0.73±0.10</td>
  </tr>
  <tr>
    <td>Triplet + EmoSim Regularization</td>
    <td>Audio</td>
    <td>0.68±0.03	</td>
    <td>0.67±0.02	</td>
    <td>0.79±0.04</td>
  </tr>
  <tr>
    <td>Triplet (Basline)</td>
    <td>Fusion</td>
    <td>0.73±0.05</td>
    <td>0.65±0.02</td>
    <td>0.74±0.10</td>
  </tr>
  <tr>
    <td>Triplet + Structure Preserving</td>
    <td>Fusion</td>
    <td>0.74±0.02</td>
    <td>0.66±0.02</td>
    <td>0.75±0.07</td>
  </tr>
  <tr>
    <td>Triplet + EmoSim Regularization</td>
    <td>Fusion</td>
    <td>0.75±0.02</td>
    <td>0.63±0.05</td>
    <td>0.73±0.04</td>
  </tr>
</tbody>
</table>

### Emotion Similarity Regularization (EmoSim)
We propose emotion similarity regularization (EmoSim), modified version of RankSim (Gong et al), for cross-domain retrieval task. An overview of our approach is shown in Figure. In practice, our goal is to encourage alignment between the similarity of neighbors in emotion space S_y and the similarity of neighbors in feature space S_z. The EmoSim regularization term is formulated as follows:

```
# code reference: https://github.com/BorealisAI/ranksim-imbalanced-regression

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
```

### Requirements

1. Install python and PyTorch:
    - python==3.8
    - torch==1.12.1 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/).)
    
2. Other requirements:
    - pip install -e .
```
conda create -n YOUR_ENV_NAME python=3.8
conda activate YOUR_ENV_NAME
pip install -e .
```

### Check constants.py
```
DATASET="../../dataset" -> dataset path
INT_RANDOM_SEED = 42
MUSIC_SAMPLE_RATE = 22050
SPEECH_SAMPLE_RATE = 16000
STR_CH_FIRST = 'channels_first'
STR_CH_LAST = 'channels_last'
INPUT_LENGTH = MUSIC_SAMPLE_RATE * 10
CHUNK_SIZE = 16

# valuence arousal from https://saifmohammad.com/WebPages/nrc-vad.html
VA_MAP = {
    "scary":[0.062,0.952],
    "funny":[0.918,0.610],
    "tender":[0.630,0.520],
    "noise":[0.500,0.500],
    "neutral":[0.500,0.500], # "neutral":[0.469,0.184], # original
    "calm": [0.442,0.302],
    "happy":[1.000,0.735],
    "sad":[0.225,0.333],
    "angry":[0.122,0.830],
    "excitement":[0.896, 0.684],
    "exciting":[0.950, 0.792],
    "disgust":[0.052,0.775],
    'anger':[0.167,0.865],
    'surprised':[0.784,0.855],
    'fearful':[0.083,0.482],
    'fear':[0.073,0.840],
    'frustration': [0.060, 0.730]
}
```


### Download Pretrained Model
```
wget https://zenodo.org/record/7341484/files/stm.tar.gz
tar -zxvf stm.tar.gz 
cp -rl stm_src/* .
```

### 1. Emotion Recognition Pre-training

For the music encoder, we use the [Music Tagging Transformer-compact_student.ckpt](https://github.com/minzwon/semi-supervised-music-tagging-transformer/tree/master/data) which is the most recent state-of-the-art music tagging model. For the speech audio encoder, we use [Wav2vec~2.0](https://huggingface.co/docs/transformers/model_doc/wav2vec2) and take the average pooling on its phoneme-level feature to summarize them into utterance-level.

**Note:** 
- music: dataset/pretrained/music/compact_student.ckpt
- audio: wav2vec download from huggingface

```
cd stm/classification/{music_cls, speech_cls}

# fintune pretrained model
python train.py --dataset_type {Audioset, IEMOCAP, RAVDESS, HIKIA} --modality audio

# evaluation on testset
python test.py --dataset_type {Audioset, IEMOCAP, RAVDESS, HIKIA} --modality audio

# extract embedding
python infer.py --dataset_type {Audioset, IEMOCAP, RAVDESS, HIKIA} --modality audio
```

### 2. Speech to Music Retrieval (Metric Learning)

```
cd mtr/metric_learning

# train triplet network
python train.py --speech_type {IEMOCAP, RAVDESS, HIKIA} --modality audio --framework triplet_soft --emo_lambda 0.5

# eval shallow classifier
python test.py --speech_type {IEMOCAP, RAVDESS, HIKIA} --modality audio --framework triplet_soft --emo_lambda 0.5
```

### Visualization
First, we confirmed that all models successfully discriminate emotion semantics. However, in the case of the triplet model, scary music embeddings are located at between sad and happy speech embedding cluster.  This problem is alleviated in Triplet + EmoSim model. There are relatively few scary music samples closer to angry and frustration clusters. We believe that joint embedding space learned inter-modality neighborhood structure from the continuous emotion similarity.

<p align = "center">
  <img src = "https://i.imgur.com/HmvZnfP.png">
</p>


### Acknowledgement
We would like to thank the [Story2Music](https://github.com/minzwon/text2music-emotion-embedding) for its training code and [RankSim](https://github.com/BorealisAI/ranksim-imbalanced-regression) for rank similarity code.

### Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.
```
@inproceedings{toward2023doh,
  title={Textless Speech-to-Music Retrieval Using Emotion Similarity},
  author={SeungHeon Doh, Minz Won, Keunwoo Choi, Juhan Nam},
  booktitle = {},
  year={2023}
}
```
