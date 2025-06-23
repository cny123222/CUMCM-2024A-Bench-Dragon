from src.dragon import *
import pandas as pd

col_time = 412.47383777 # 碰撞时刻(s)

dragon = Dragon()
dragon.set_time(col_time, need_vol=True)

col_0 = ['龙头']
for i in range(1, 222):
    col_0.append(f'第{i}节龙身')
col_0.extend(['龙尾', '龙尾（后）'])

data = {'': col_0}

data['横坐标x (m)'] = [ben_pos[0] for ben_pos in dragon.pos]
data['纵坐标y (m)'] = [ben_pos[1] for ben_pos in dragon.pos]
data['速度 (m/s)'] = dragon.vol

df = pd.DataFrame(data)
df.to_csv('my_result4.csv', index=False)