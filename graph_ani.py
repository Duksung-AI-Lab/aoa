import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import pandas as pd

angle = 90

df = pd.read_csv('aoa/aoa_dataset.csv', engine='python')    ### 경로 설정
df.columns = ['aoa', 'rssi0', 'rssi1', 'pa0', 'pa1', 'angle']
df = df.drop(['aoa', 'rssi0', 'rssi1'], axis=1)

data = []
for k in range(-90, 91, 10):
    a = df[df['angle'] == k]
    data.append(a)

pa0 = data[angle//10+9]['pa0']
pa1 = data[angle//10+9]['pa1']

fig1, ax = plt.subplots()
line0, = ax.plot([], [], lw=2, c="violet", label='pa0')
line1, = ax.plot([], [], lw=2, c="dodgerblue", label='pa1')


def init():
    ax.set_xlim((0, 20))
    ax.set_ylim((-160, 190))
    return line0, line1


def update(i):
    x = np.linspace(0, 19, 20)
    y0 = pa0[i:i+20]
    y1 = pa1[i:i+20]
    line0.set_data(x, y0)
    line1.set_data(x, y1)
    return line0, line1


anim = animation.FuncAnimation(fig1, update, init_func=init,
                               frames=len(pa0)-20, interval=20, blit=True)
rc('animation', html='html5')
plt.show()
