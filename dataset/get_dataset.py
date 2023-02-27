from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
import time
from serial import Serial
import pandas as pd
import time
import csv_edit
print("측정 각도 입력:", end='')
label=str(input())

st=time.time()
ser = Serial('/dev/ttyACM0', 115200) #라즈베리파이
f=open("/home/hyunji/aoa/dataset_"+label+".csv","w", newline='')
wr=csv_edit.writer(f)
if ser.isOpen()==False:
    ser.open()
i = 0
a=[]
while i<=20000:
    a=[]
    if ser.readable():
        res = ser.readline()
        s = res.decode()[:len(res) - 1]
        #[aoa, rssi0, rssi1, pa0, pa1]
    try:
        s=list(s.split())[1]
        a=list(map(int,s.split(",")))
        if i%1000==0 : print(i)

        a.append(label)
        wr.writerow(a)
    except IndexError:
        continue
    except (KeyboardInterrupt, ValueError):
        ser.flushInput()
        f.close()
    
    finally:
        i+=1


print("time:",time.time()-st)
f.close()
        
