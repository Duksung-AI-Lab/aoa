import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('../aoa_dataset.csv', engine='python')
df.columns = ['aoa', 'rssi0', 'rssi1', 'pa0', 'pa1', 'angle']
print(df.head())

size = 100
data = []

### plot each angles ###
for k in range(10):
    for i in range(2):
        angle = (k*2+i)*10-90
        if angle > 90:
            break

        data.append(df[df['angle'] == angle])
        print(len(data))

        ax = plt.subplot(2, 1, i+1)
        ax.set_title(angle)

        x_len = np.arange(size)
        plt.ylim([-160, 190])
        ax.plot(x_len, data[k * 2 + i]['rssi0'][:size], c="blue", label='r0')
        ax.plot(x_len, data[k * 2 + i]['rssi1'][:size], c="red", label='r1')
        ax.plot(x_len, data[k * 2 + i]['aoa'][:size], c="black", label='aoa')
        ax.plot(x_len, data[k * 2 + i]['pa0'][:size], c="violet", label='p0')
        ax.plot(x_len, data[k * 2 + i]['pa1'][:size], c="dodgerblue", label='p1')

        plt.legend(loc=(1.0, 0.0), ncol=2, prop={'size': 7})
        ax.grid()

    plt.tight_layout()
    plt.show()

# ### plot every angles ###
# plt.figure(figsize=(12, 8))
# color = ["#F15F5F", "#F29661", "#E5D85C", "#86E57F", "#5CD1E5", "#6799FF", "#6B66FF", "#A566FF", "#F361DC", "#F361A6",
#          "#747474",
#          "#CC3D3D", "#CCA63D", "#9FC93C", "#47C83E", "#3DB7CC", "#4374D9", "#4641D9", "#8041D9", "#D941C5", "#D9418C"]
# for i in range(len(data)):
#     plt.ylim([-160, 190])
#     x_len = np.arange(size)
#     plt.plot(x_len, data[i]['pa0'][:size], c=color[i], label='pa0')
#     plt.plot(x_len, data[i]['pa1'][:size], c=color[i], label='pa1', linestyle="--")
# plt.show()