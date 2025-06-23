from src.dragon2 import *

col_0 = ['龙头x (m)', '龙头y (m)']
for i in range(1, 222):
    col_0.append(f'第{i}节龙身x (m)')
    col_0.append(f'第{i}节龙身y (m)')
col_0.extend(['龙尾x (m)', '龙尾y (m)', '龙尾（后）x (m)', '龙尾（后）y (m)'])

data = {'': col_0}

# 构造时间序列
time_lst = []
for time in range(0, 41):
    time_lst.append(time)
    time_lst.append(time + 0.01)

dragon = Dragon()
for time in time_lst:
    if time % 10 == 0:
        print(f"已求解到{time}s")

    dragon.set_time(time)
    xy_lst = []
    for x, y in dragon.pos:
        xy_lst.append(x)
        xy_lst.append(y)
    
    data[f"{time} s"] = xy_lst

df = pd.DataFrame(data)
df.to_csv('my_result5_2.csv', index=False)
