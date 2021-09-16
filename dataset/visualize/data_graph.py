import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('aoa/aoa_dataset.csv', engine='python')
df.columns = ['aoa', 'rssi0', 'rssi1', 'pa0', 'pa1', 'angle']
df = df.drop(['aoa', 'rssi0', 'rssi1'], axis=1)
print(df.head())
# print(max(df['pa0']), max(df['pa1']), min(df['pa0']), min(df['pa1']))
size = 20
data = []
for i in range(-90, 91, 10):
    a = df[df['angle'] == i]
    data.append(a)
    print(len(a))

for i in range(len(data)):
    plt.figure(figsize=(12, 8))
    plt.suptitle(i*10 - 90)
    for j in range(6):
        plt.subplot(3, 2, j+1)
        x_len = np.arange(len(data[i]['angle'][size*j:size*(j+1)]))
        plt.ylim([-160, 190])
        plt.plot(x_len, data[i]['pa0'][size*j:size*(j+1)], c="violet", label='pa0')
        plt.plot(x_len, data[i]['pa1'][size*j:size*(j+1)], c="dodgerblue", label='pa1')
    # plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# plt.figure(figsize=(12, 8))
# color = ["#F15F5F", "#F29661", "#E5D85C", "#86E57F", "#5CD1E5", "#6799FF", "#6B66FF", "#A566FF", "#F361DC", "#F361A6",
#          "#747474",
#          "#CC3D3D", "#CCA63D", "#9FC93C", "#47C83E", "#3DB7CC", "#4374D9", "#4641D9", "#8041D9", "#D941C5", "#D9418C"]
# for i in range(len(data)):
#     plt.ylim([-160, 190])
#     x_len = [j for j in range(size)]
#     plt.plot(x_len, data[i]['pa0'][:size], c=color[i], label='pa0')
#     plt.plot(x_len, data[i]['pa1'][:size], c=color[i], label='pa1', linestyle="--")
# plt.show()
