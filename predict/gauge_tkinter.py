import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
import collections
import time
import numpy as np
import pandas as pd

from serial import Serial
from tensorflow.keras.models import load_model

model1 = load_model('./model/210916_rnn_reg.h5')
# model2 = load_model('./model/LSTM_class2.h5')

# ser = Serial('COM9', 9600)  # 윈도우
ser = Serial('/dev/ttyACM0', 115200)  # 라즈베리파이

# scaler
df_train = pd.read_csv('./dataset/aoa_train.csv', engine='python')
scaler = RobustScaler()
df_train.columns = ['dummy', 'pa0', 'pa1', 'angle']
scaler.fit(df_train[['pa0', 'pa1']])

angle_scaler = MinMaxScaler()
angle_scaler.fit(df_train[['angle']])

p = collections.deque()
SIZE = 20
# label = [i for i in range(-90, 91, 10)]

WIN_SIZE = (500, 600)
ARC_SIZE = 100
angle = 0


def get_degree():
    global angle
    while True:
        if ser.readable():
            res = ser.readline()
            s = res.decode()[:len(res) - 1]

            a = s.split(',')
            try:
                # parsing
                pa0 = int(a[3].split()[-1])
                pa1 = int(a[4].split()[-1])
                # print(pa0, pa1)

            except IndexError:
                continue

            if len(p) < SIZE:
                p.append([pa0, pa1])
                continue
            else:
                p.popleft()
                p.append([pa0, pa1])
            # print(p)

            # RobustScale
            data = scaler.transform(p)

            # CNN data shape
            data = np.reshape(data, (-1, SIZE, 1, 2))
            # RNN data shape
            # data = np.reshape(data, (1, SIZE, 2))

            # print(data.shape)

            pred1 = model1.predict(data)
            angle = angle_scaler.inverse_transform(pred1)

            # aoa = int(a[0].split()[-1]) + 45
            # pred2 = np.argmax(model2.predict(data), axis=-1)
            # angle = label[pred2[0]]

            print(*angle)

            time.sleep(0.1)


class Gauge(ttk.Label):

    def __init__(self, parent, **kwargs):
        self.arc = None
        self.im = Image.new('RGBA', (1000, 1000))
        self.min_value = kwargs.get('minvalue') or 0
        self.max_value = kwargs.get('maxvalue') or 100
        self.size = kwargs.get('size') or 400
        self.font = kwargs.get('font') or 'helvetica 20 bold'
        self.background = kwargs.get('background')
        self.foreground = kwargs.get('foreground') or '#777'
        self.troughcolor = kwargs.get('troughcolor') or '#e0e0e0'
        self.indicatorcolor = kwargs.get('indicatorcolor') or '#01bdae'
        # self.arcvariable = tk.IntVar(value='text')
        # self.arcvariable.trace_add('write', self.update_arcvariable)
        self.textvariable = tk.StringVar()
        self.setup()

        super().__init__(parent, image=self.arc, compound='center', style='Gauge.TLabel',
                         textvariable=self.textvariable, **kwargs)

    def setup(self):
        """Setup routine"""
        style = ttk.Style()
        style.configure('Gauge.TLabel', font=self.font, foreground=self.foreground)
        if self.background:
            style.configure('Gauge.TLabel', background=self.background)
        draw = ImageDraw.Draw(self.im)
        draw.arc((0, 0, 990, 990), -180, 0, self.troughcolor, ARC_SIZE)
        self.arc = ImageTk.PhotoImage(self.im.resize((self.size, self.size), Image.LANCZOS))

    def update_arcvariable(self, *args):
        global angle
        while True:
            """Redraw the arc image based on variable settings"""
            # angle = int(float(self.arcvariable.get())) + 180
            self.im = Image.new('RGBA', (1000, 1000))
            draw = ImageDraw.Draw(self.im)
            draw.arc((0, 0, 990, 990), -180, 0, self.troughcolor, ARC_SIZE)
            draw.arc((0, 0, 990, 990), angle - 91, angle - 89, self.indicatorcolor, ARC_SIZE)

            font = ImageFont.truetype(font="TlwgTypo-Bold.ttf", size=50)
            draw.text((450, 500), str(angle), (0, 0, 0), font)

            self.arc = ImageTk.PhotoImage(self.im.resize((self.size, self.size), Image.LANCZOS))
            self.configure(image=self.arc)

            time.sleep(0.1)


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry(str(WIN_SIZE[0]) + 'x' + str(WIN_SIZE[1]))
    root.title("Angle")

    style = ttk.Style()
    gauge = Gauge(root, padding=50)
    gauge.pack()
    # ttk.Scale(root, from_=0, to=180, variable=gauge.arcvariable).pack(fill='x', padx=10, pady=10)
    # update the textvariable with the degrees information when the arcvariable changes
    # gauge.arcvariable.trace_add('write', lambda *args, g=gauge: g.textvariable.set(f'{g.arcvariable.get()} deg'))

    t1 = threading.Thread(target=get_degree)
    t1.start()
    t2 = threading.Thread(target=gauge.update_arcvariable)
    t2.start()
    root.mainloop()
