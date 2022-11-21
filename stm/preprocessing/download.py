import os
import csv
import numpy as np
import pandas as pd
import youtube_dl
import multiprocessing as mp
from speech_to_music.constants import (DATASET)


def _get_audioset_data(subset_path, label_dict, tag, split):
    results = []
    target_id = label_dict[tag]['mid']
    tag = tag.split()[0]
    with open(subset_path, 'r') as f:
        subset_data = csv.reader(f)
        for row_idx, row in enumerate(subset_data):
            # Skip commented lines
            if row[0][0] == '#':
                continue
            ytid, ts_start, ts_end, pos = row[0], float(row[1]), float(row[2]), row[3:]
            pos = " ".join(pos).replace('"', '').split()
            if target_id in pos:
                results.append({
                    "_id" : ytid,
                    "start" : ts_start,
                    "end" : ts_end,
                    "label" : tag.lower(),
                    "split" : split
                })
    return results

def audioset_filter():
    label_dict = pd.read_csv(os.path.join(DATASET, "raw/Audioset/class_labels_indices.csv")).set_index("display_name").to_dict('index')
    results = []
    target_tag = ["Happy music","Funny music",
                "Sad music","Tender music",
                "Exciting music","Angry music",
                "Scary music"]
    for tag in target_tag:
        train_result = _get_audioset_data(subset_path = os.path.join(DATASET, "raw/Audioset/unbalanced_train_segments.csv"), label_dict = label_dict, tag = tag, split= "TRAIN")
        eval_result = _get_audioset_data(subset_path = os.path.join(DATASET, "raw/Audioset/eval_segments.csv"), label_dict = label_dict, tag = tag, split= "EVAL")
        results.extend(train_result)
        results.extend(eval_result)
    df = pd.DataFrame(results)
    df_label = pd.get_dummies(df['label'])
    df = df.drop(columns=['label'])
    for i in df_label.columns:
        df[i] = df_label[i]
    # drop duplicate index
    df = df.set_index("_id")
    index= df.index
    is_duplicate = index.duplicated(keep="first")
    not_duplicate = ~is_duplicate
    df = df[not_duplicate]
    df.to_csv(os.path.join(DATASET, "raw/Audioset/audioset_mood.csv"))
    return df

def audio_crawl(url, audio_out_dir):
    ydl_opts = {
        'format': 'bestaudio/best',
        'writeinfojson': False,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }],
        'outtmpl': audio_out_dir
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download = True)
        
def list_of_item():
    df_audioset = audioset_filter()
    return [df_audioset.iloc[idx].name for idx in range(len(df_audioset))]

def extractor(_id):
    url = f"https://www.youtube.com/watch?v={_id}"
    audio_out_dir = os.path.join(DATASET, f"raw/Audioset/wav/{_id}.mp3")
    try:
        audio_crawl(url, audio_out_dir)
    except:
        error_path = os.path.join(DATASET, f"raw/Audioset/error/{_id}.npy")
        np.save(error_path, _id)

if __name__ == '__main__':
    ids = list_of_item()
    print(len(ids))
    pool = mp.Pool(20)
    pool.map(extractor, ids)
    print(len(os.listdir(os.path.join(DATASET, "raw/Audioset/wav/"))))
    print("Extraction finish")

    # check error sample again
    error_ids = [fname.replace(".npy","") for fname in os.listdir(os.path.join(DATASET, "raw/Audioset/error")) if ".npy" in fname]
    pool = mp.Pool(20)
    pool.map(extractor, error_ids)