import seaborn as sns
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Plot audio")
parser.add_argument("--clean", type=str)
parser.add_argument("--dirty", type=str)
parser.add_argument("--enhanced", type=str)
opt = parser.parse_args()

clean, sr = librosa.load(opt.clean, sr=16000)
dirty, sr = librosa.load(opt.dirty, sr=16000)
enhanced, sr = librosa.load(opt.enhanced, sr=16000)

min_length = min([len(clean), len(dirty), len(enhanced)])

clean = clean[:min_length]
dirty = dirty[:min_length]
enhanced = enhanced[:min_length]


audio = {
    "clean": clean,
    "dirty-enhanced": dirty-enhanced,
    "dirty": dirty,
    "enhanced": enhanced
}

plt.figure(figsize=(6, 6))
i = 1
for key in audio.keys():
    plt.subplot(2, 2, i)
    librosa.display.waveplot(audio[key], alpha=0.8)
    plt.title(key)
    i += 1
plt.tight_layout()
plt.show()
