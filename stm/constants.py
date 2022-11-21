DATASET="../../dataset"
CLSREG_DATASET = "../../../dataset"
INT_RANDOM_SEED = 42
MUSIC_SAMPLE_RATE = 22050
SPEECH_SAMPLE_RATE = 16000
STR_CH_FIRST = 'channels_first'
STR_CH_LAST = 'channels_last'
INPUT_LENGTH = MUSIC_SAMPLE_RATE * 10
CHUNK_SIZE = 16

AUDIOSET_TAGS = ['happy','funny','sad','tender','exciting','angry','scary','noise']
IEMOCAP_ORIGIN_TAGS = ["ang","hap","neu","sad","exc","fru"] # anger, happiness, sadness, neutral, excitement, and frustration.
IEMOCAP_TAGS = ["angry","happy","neutral","sad", "excitement", "frustration"]
HIKIA_TAGS = ["angry","sad","happy","neutral"]
IEMOCAP_TAGMAP = {'ang': 'angry','hap':'happy','neu':'neutral','sad':'sad','exc':'excitement','fru':'frustration'}
RAVDESS_TAGS = ["angry","calm","disgust","fearful","happy","neutral","sad","surprised"]
EMOFILM_TAGS = ['fear','disgust','happy','anger','sad']
EMOFILM_MAP = {"ans":"fear", "dis":"disgust", "gio":"happy", "rab":"anger", "tri":"sad"}

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

IEMOCAP_TO_AUDIOSET = {
    "angry" : ["angry","scary"],
    "happy" : ["happy","exciting","funny","tender"],
    "neutral" : ["noise"],
    "sad" : ["sad"]
}

RAVDESS_TO_AUDIOSET = {
    "angry" : ["angry","scary"],
    "happy" : ["happy","exciting","funny","tender"],
    "neutral" : ["noise"],
    "surprised" : ["exciting"],
    "fearful" : ["scary"],
    "disgust" : ["scary","angry"],
    "sad" : ["sad"]
    }

EMOFILM_TO_AUDIOSET = {
    "anger" : ["angry","scary"],
    "happy" : ["happy","exciting","funny","tender"],
    "fear" : ["scary"],
    "disgust" : ["scary","angry"],
    "sad" : ["sad"]
}
    
RAVDESS_CLASS_DICT = {
    "modality": {
        "01":"full-AV",
        "02":"video-only",
        "03":"audio-only"
    },
    "vocal_channel": {
        "01":"speech",
        "02":"song"
    },
    "emotion":{
        "01":"neutral", 
        "02":"calm", 
        "03":"happy", 
        "04":"sad", 
        "05":"angry", 
        "06":"fearful", 
        "07":"disgust", 
        "08":"surprised"
    },
    "intensity":{
        "01":"normal", 
        "02":"strong"
    },
    "statement":{
        "01":"Kids are talking by the door", 
        "02":"Dogs are sitting by the door",
    },
    "repetition":{
       "01":"1st repetition", 
        "02": "2nd repetition "
    }
}

EMOFILM_MAP = {
    "emotion":{
        "ans":"fear", 
        "dis":"disgust", 
        "gio":"happy", 
        "rab":"anger", 
        "tri":"sad"
    },
    "gender":{
        "f":"female",
        "m":"male"
    },
    "langauge":{
        "it":"italian",
        "es":"spanish",
        "en":"english"
    }
}
EMOTION_TO_COLOR = {
    'happy':"tab:olive",
    'funny':"darkkhaki",
    'exciting':"olive",
    'excitement':"olive",
    'tender':"khaki",
    'surprise':"yellow",
    'calm':"disque",
    'neutral':"#C9C9C9",
    'noise':"#C9C9C9",
    'sad':"tab:blue",
    'angry':"tab:red",
    'scary':"deeppink",
    "frustration": "maroon",
    'fear':"sienna",
    'disgust':"darkred",
    "fearful": "sienna"
    }