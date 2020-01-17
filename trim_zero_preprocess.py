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

from config import *


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

# adapted from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html


def add_clipping(signal):
    peak = np.max(signal)
    return np.clip((1+np.random.uniform(0.5, 1.0)*CLIPPING_FACTOR)*signal, -peak, peak)


def butter_bandpass(lowcut, highcut, order=1):
    nyq = 0.5 * SR
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# lowcut ~ U(0,400), highcut ~ U(4000,6000)
def butter_bandpass_filter(data, order=1):
    lowcut = random.random() * 200
    highcut = random.random() * 1000 + 5000
    b, a = butter_bandpass(lowcut, highcut, order=order)
    y = lfilter(b, a, data)
    return y


esc_50 = [
    f'./ESC-50-master/audio/{x}' for x in listdir("./ESC-50-master/audio")]
fsdkaggle_train = [
    f'./FSDKaggle_train/{x}' for x in listdir("./FSDKaggle_train")]
fsdkaggle_test = [f'./FSDKaggle_test/{x}' for x in listdir("./FSDKaggle_test")]

files = fsdkaggle_train + fsdkaggle_test + esc_50
np.random.shuffle(files)
files = files[:5000]

# 1. read files
# 2. add noise and distortion
# 3. save audio and melspectrograms
if not os.path.exists("./data/filtered_trimmed"):
    os.mkdir("./data/filtered_trimmed")
if not os.path.exists("./data/distorted_trimmed"):
    os.mkdir("./data/distorted_trimmed")
if not os.path.exists("./data/clean_trimmed"):
    os.mkdir("./data/clean_trimmed")

for i, f in enumerate(tqdm(files)):
    try:
        # CREATE AUDIO DATA SETS
        # (1) read file, (2) trim or pad length, (3) lift baseline gain
        clean_audio = lift_baseline_gain(trim(read_wav(f)))
        clean_audio = [x for x in clean_audio if x > 0.1 or x < -0.1]
        clean_audio = np.array(clean_audio)
        if len(clean_audio) < MIN_LENGTH:
            continue

        filtered_audio = butter_bandpass_filter(clean_audio)
        distorted_audio = add_clipping(clean_audio)

        write('./data/clean_trimmed', f, clean_audio)
        write('./data/filtered_trimmed', f, filtered_audio)
        write('./data/distorted_trimmed', f, distorted_audio)

    except Exception as e:
        print(f, e)
        pass
