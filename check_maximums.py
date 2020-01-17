from librosa import load
from os import listdir
import numpy as np
from tqdm import trange

clean_dir = listdir("./data/clean_train")
noisy_dir = listdir("./data/noisy_train")
np.random.shuffle(clean_dir)
np.random.shuffle(noisy_dir)

c_maxes = []
n_maxes = []
c_sd = []
n_sd = []
c = []
n = []

for i in trange(100):
    clean, _ = load("./data/clean_train/" + clean_dir[i], sr=16000)
    noisy, _ = load("./data/noisy_train/" + noisy_dir[i], sr=16000)
    # c_maxes.append(sum(np.abs(clean)) / len(clean))
    # n_maxes.append(sum(np.abs(noisy)) / len(noisy))
    # c_sd.append(np.std(clean))
    # n_sd.append(np.std(noisy))
    c.append(clean)
    n.append(noisy)

print("Mean amplitude of clean:", np.mean(np.abs(clean)))
print("Mean amplitude of noisy:", np.mean(np.abs(noisy)))
print("Std. of clean:", np.std(clean))
print("Std. of noisy:", np.std(noisy))
