import pandas as pd
from librosa import load
from os import listdir
import subprocess
from tqdm import tqdm
import numpy as np

clean_dir = listdir("./data/clean_train")
noisy_dir = listdir("./data/noisy_train")
distorted_dir = listdir("./data/distorted_train")
filtered_dir = listdir("./data/filtered_train")

sets = {"clean": clean_dir, "noisy": noisy_dir,
        "distorted": distorted_dir, "filtered": filtered_dir}

ENHANCE = False

# generate enhanced versions of a subsample of training data
if ENHANCE:
    for key in sets.keys():
        if not key == "clean":
            subprocess.call(
                ['python', 'SEGAN-pytorch/test_audio.py',
                 '--dir', f'data/{key}_train',
                 "--generator", f'models/{key}/generator-final.pkl',
                 '--outpath', f'data/{key}_enhanced'])

enhanced_sets = {
    "noisy": listdir("./data/noisy_enhanced"),
    "distorted": listdir("./data/distorted_enhanced"),
    "filtered": listdir("./data/filtered_enhanced")
}

# get MSE of the datasets compared to clean
mse = {}
for key in enhanced_sets.keys():
    mse[key] = []
    for i in range(50):
        if enhanced_sets[key][i].startswith("."):
            continue
        dirty, sr = load(
            f'./data/{key}_enhanced/{enhanced_sets[key][i]}', sr=16000)
        clean, sr = load(
            f'./data/clean_train/{enhanced_sets[key][i].split("_")[1]}', sr=16000)
        max_len = min(len(clean), len(dirty))
        mse[key].append(
            ((clean[:max_len-1] - dirty[max_len-1])**2).mean(axis=0))

for key in mse.keys():
    mmse = sum(mse[key]) / len(mse[key])
    print(key, mmse)
