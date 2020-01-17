import librosa
import librosa.display
import soundfile
import os
from os import listdir
from matplotlib import pyplot as plt
import pylab
import numpy as np
import pandas as pd
from tqdm import tqdm
from pysndfx import AudioEffectsChain
import random
from scipy.signal import butter, lfilter
from shutil import rmtree


from config import *


os.chdir("/Users/Nick/Documents/coding/research")

# read in audio files from all sources
esc_50 = [
    f'./ESC-50-master/audio/{x}' for x in listdir("./ESC-50-master/audio")]
fsdkaggle_train = [
    f'./FSDKaggle_train/{x}' for x in listdir("./FSDKaggle_train")]
fsdkaggle_test = [f'./FSDKaggle_test/{x}' for x in listdir("./FSDKaggle_test")]

files = fsdkaggle_train + fsdkaggle_test + esc_50
np.random.shuffle(files)
files = files[:DATASET_SIZE]

# ---CONSTANTS---
SAVE_PICS = False
print(CONFIG_INFO)

# create directory for real_data and smartphone_data

rmtree(DATA_DIR, ignore_errors=True)
os.mkdir(DATA_DIR)
if(not os.path.exists(IMG_DIR)):
    os.mkdir(IMG_DIR)
for directory in DIRS:
    train_dir = f'{directory}_train'
    test_dir = f'{directory}_test'
    rmtree(train_dir, ignore_errors=True)
    rmtree(test_dir, ignore_errors=True)
    os.mkdir(train_dir)
    os.mkdir(test_dir)


def read_wav(f):
    raw, _ = librosa.core.load(
        f, sr=SR, res_type="kaiser_fast")
    return raw


def trim(raw):
    # cut random 5s section from long clips
    if len(raw) >= MAX_LENGTH:
        left_padding = random.randint(0, raw.shape[0]-MAX_LENGTH)
        real_data = raw[left_padding: left_padding + MAX_LENGTH]
    else:
        real_data = raw
    return real_data


# write audio file
def write(dirname, fpath, data):
    soundfile.write(
        f'{dirname}/{fpath.split("/")[-1]}', data, samplerate=SR)


# save numpy array
def save(dirname, fpath, array):
    np.save(f'{dirname}/{fpath.split("/")[-1]}', array)


# lift baseline gain of all sound files
def lift_baseline_gain(signal):
    fx = AudioEffectsChain().gain(2)
    return fx(signal)


# sd specified according to subjective expectations of real world noise
def add_noise(signal):
    return np.add(signal, random.random() * np.random.normal(0, 0.01, signal.shape[0]))


# clips off overhang
def add_clipping(signal):
    peak = np.max(signal)
    return np.clip((1+np.random.uniform(0.5, 1.0)*CLIPPING_FACTOR)*signal, -peak, peak)


# create a spectrogram the same shape as pix2pix input
def melspec(signal, hop_length):
    return librosa.feature.melspectrogram(y=signal, sr=SR, hop_length=hop_length, n_mels=N_MELS)


# adapted from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, order=1):
    nyq = 0.5 * SR
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# lowcut ~ U(0,400), highcut ~ U(4000,6000)
def butter_bandpass_filter(data, order=1):
    lowcut = random.random() * 400
    highcut = random.random() * 2000 + 4000
    b, a = butter_bandpass(lowcut, highcut, order=order)
    y = lfilter(b, a, data)
    return y


# 1. read files
# 2. add noise and distortion
# 3. save audio and melspectrograms
for i, f in enumerate(tqdm(files)):
    try:
        # CREATE AUDIO DATA SETS
        # (1) read file, (2) trim or pad length, (3) lift baseline gain
        clean_audio = lift_baseline_gain(trim(read_wav(f)))
        if clean_audio.shape[0] < MIN_LENGTH:
            continue

        # CREATE MODIFIED DATA SETS
        distorted_audio = add_clipping(clean_audio)
        noisy_audio = add_noise(clean_audio)
        filtered_audio = butter_bandpass_filter(clean_audio)

        # hop_length = clean_audio.shape[0] // (PIX2PIX_SHAPE[1]+1)
        # clean_spec = melspec(clean_audio, hop_length)
        # distorted_spec = melspec(distorted_audio, hop_length)
        # noisy_spec = melspec(noisy_audio, hop_length)
        # filtered_spec = melspec(filtered_audio, hop_length)

        datasets = [
            {"title": "Clean Audio", "data": clean_audio, "dir": CLEAN_AUDIO_DIR},
            {"title": "Noisy Audio", "data": noisy_audio, "dir": NOISY_AUDIO_DIR},
            {"title": "Distorted Audio", "data": distorted_audio,
                "dir": DISTORTED_AUDIO_DIR},
            {"title": "Filtered Audio", "data": filtered_audio,
                "dir": FILTERED_AUDIO_DIR}
        ]

        # CREATE TEST AND TRAIN DATASET
        if i % int(TEST_SIZE*len(files)) == 0:
            for i, d in enumerate(datasets):
                write(f'{d["dir"]}_test', f, d["data"])
        else:
            for i, d in enumerate(datasets):
                write(f'{d["dir"]}_train', f, d["data"])

        if SAVE_PICS:
            plt.figure(figsize=(12, 8))
            for i, d in enumerate(datasets):
                plt.subplot(2, 2, i+1)
                librosa.display.waveplot(d["data"], x_axis="time")
                plt.title(d["title"])

            plt.tight_layout()
            plt.savefig(f'{IMG_DIR}/{f.split("/")[-1]}.png')

    except Exception as e:
        print(f, e)
        pass
