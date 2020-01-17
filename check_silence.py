from librosa import load
from os import listdir
import numpy as np
from tqdm import trange

clean = listdir("./data/clean_train")
np.random.shuffle(clean)

counts = []

for i in trange(100):
    f, _ = load("./data/clean_train/" + clean[i], sr=16000)
    counts.append(sum(i < 0.001 for i in f)/len(f))

print("Average silence (< 0.001amp):", sum(counts)/len(counts))
