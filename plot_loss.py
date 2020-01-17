import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as plt

sns.set_context("paper")
sns.set(style="whitegrid")


parser = argparse.ArgumentParser("Process graphs for GAN")
parser.add_argument("--loss_file", required=True)
opt = parser.parse_args()
path = opt.loss_file

pdloss = pd.read_csv(path)
loss = pdloss.values

colnames = ["G-Adversarial", "G-L1", "G-Total Loss",
            "D-Clean", "D-Dirty", "D-Total Loss"]

x = loss[:, 0]
plt.figure(figsize=(12, 8))


def plot(i, pos):
    plt.subplot(3, 2, pos)
    plt.title(colnames[i], fontsize=20)
    sns.lineplot(x, loss[:, i + 1])


plot(0, 1)
plot(1, 3)
plot(2, 5)
plot(3, 2)
plot(4, 4)
plot(5, 6)
plt.tight_layout()
plt.show()
