import time
import threading
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk, ImageDraw

SIZE = (500, 600)
ARC_SIZE = 100

angle = -180


def get_degree():
    global angle
    while True:
        angle += 10
        time.sleep(1)


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
            draw.arc((0, 0, 990, 990), -180, angle, self.indicatorcolor, ARC_SIZE)
            self.arc = ImageTk.PhotoImage(self.im.resize((self.size, self.size), Image.LANCZOS))
            self.configure(image=self.arc)
            text.set(angle)
            label.configure()
            time.sleep(1)


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry(str(SIZE[0]) + 'x' + str(SIZE[1]))
    style = ttk.Style()
    gauge = Gauge(root, padding=50)
    gauge.pack()
    # ttk.Scale(root, from_=0, to=180, variable=gauge.arcvariable).pack(fill='x', padx=10, pady=10)
    # update the textvariable with the degrees information when the arcvariable changes
    # gauge.arcvariable.trace_add('write', lambda *args, g=gauge: g.textvariable.set(f'{g.arcvariable.get()} deg'))

    text = tk.StringVar()
    text.set(angle)
    label = tk.Label(root,
                     textvariable=text,
                     font=("gothic", "20"))
    label.pack()

    t1 = threading.Thread(target=get_degree)
    t1.start()
    t2 = threading.Thread(target=gauge.update_arcvariable)
    t2.start()
    root.mainloop()
