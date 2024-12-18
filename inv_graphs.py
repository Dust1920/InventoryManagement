"""
    Graphics for the Inventory Control Model
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("S_100_((100, 0.1))_0.05_((20, 10)).xlsx", index_col=0)
print(data)

samples = data['SAMPLE'].unique()
figure, ax = plt.subplots(nrows=2, ncols=2, figsize = (12, 8))
for s in samples[:3]:
    data_s = data[data['SAMPLE'] == s]
    ax[0,0].plot(data_s['state'], lw = 0.7, label = f"l_{s}")
    ax[0,0].set_xlabel("Stages")
    ax[0,0].set_ylabel("Level")
    ax[0,1].plot(data_s['r'], lw = 0.7)
    ax[0,1].set_ylabel("Reward")
    ax[0,1].set_xlabel("Stages")
    ax[1,1].plot(data_s['r'].cumsum(), lw = 0.7)
    ax[1,1].set_xlabel("Stages")
    ax[1,1].set_ylabel("Acc. Reward")
figure.legend()
figure.savefig("Data.png")
plt.show()
