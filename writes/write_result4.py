from src.dragon2 import *

col_0 = ['龙头 (m/s)']
for i in range(1, 222):
    col_0.append(f'第{i}节龙身 (m/s)')
col_0.extend(['龙尾 (m/s)', '龙尾（后） (m/s)'])

data = {'': col_0}

dragon = Dragon()
for time in np.arange(-100, 101, 1):
    if time % 10 == 0:
        print(f"已求解到{time}s")

    dragon.set_time(time, need_vol=True)
    
    data[f"{time} s"] = dragon.vol

df = pd.DataFrame(data)
df.to_csv('my_result6.csv', index=False)