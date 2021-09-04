import threading
import tkinter as tk
from tkinter import ttk
import random
from PIL import Image, ImageTk, ImageDraw, ImageFont
from sklearn.preprocessing import MinMaxScaler
import collections
import time

from serial import Serial
from tensorflow.keras.models import load_model

# model = load_model('lstm_class.h5')
# ser = Serial('/dev/ttyACM0', 115200)
# p = collections.deque()
# SIZE = 100

WIN_SIZE = (500, 600)
ARC_SIZE = 100
angle = -180

def get_degree():
    global angle
    while True:
        '''
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
            # MinMaxScale (0~1)
            scaler = MinMaxScaler()
            data = scaler.fit_transform(p)
            data = np.reshape(data, (1, SIZE, 2))

            pred = np.argmax(model.predict(data), axis=-1)
            angle = list(map(lambda x: x * 10 - 90, pred))
            print(pred, angle, a[0].split()[-1])
            '''
        angle =random.randrange(-90,90,10)
        angle-=90
        time.sleep(0.2)

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
            draw.arc((0, 0, 990, 990), angle-1, angle+1, self.indicatorcolor, ARC_SIZE)

            font = ImageFont.truetype(font="AppleGothic.ttf", size=50)
            draw.text((450, 500), str(angle+90), (0,0,0), font)

            self.arc = ImageTk.PhotoImage(self.im.resize((self.size, self.size), Image.LANCZOS))
            self.configure(image=self.arc)

            time.sleep(0.2)


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
