import pandas as pd

df = pd.read_csv('aoa_dataset.csv', engine='python')
df.columns = ['aoa', 'rssi0', 'rssi1', 'pa0', 'pa1', 'angle']
df = df.drop(['aoa', 'rssi0', 'rssi1'], axis=1)
print(df.head())

SIZE = 2000
train, val, test = [], [], []

data = []
for i in range(-90, 91, 10):
    a = df[df['angle'] == i]
    a = a.reset_index()
    print(a.head())
    test.append(a.loc[:SIZE])
    val.append(a.loc[SIZE:SIZE*2])
    train.append(a.loc[SIZE*2:])

print(test)
pd.concat(test).to_csv("aoa_test.csv", index=False, mode='w', header=False)
pd.concat(val).to_csv("aoa_val.csv", index=False, mode='w', header=False)
pd.concat(train).to_csv("aoa_train.csv", index=False, mode='w', header=False)
