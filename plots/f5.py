"""
图5 -- t = 0s 至 t = 300s 部分把手的极角变化情况
"""
from src.dragon import *

plt.figure(figsize=(8, 5))

x = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []

dragon = Dragon()
for t in range(301):
    dragon.set_time(t)

    x.append(t)
    y1.append(dragon.ang[0])
    y2.append(dragon.ang[51])
    y3.append(dragon.ang[101])
    y4.append(dragon.ang[151])
    y5.append(dragon.ang[201])
    y6.append(dragon.ang[-1])


plt.plot(x, y1, color='#5EC25E', linestyle='-', label='龙头前把手极角')
plt.plot(x, y2, color='#00B580', linestyle='-', label='第51节龙身前把手极角')
plt.plot(x, y3, color='#00A59F', linestyle='-', label='第101节龙身前把手极角')
plt.plot(x, y4, color='#0092B4', linestyle='-', label='第151节龙身前把手极角')
plt.plot(x, y5, color='#007CBB', linestyle='-', label='第201节龙身前把手极角')
plt.plot(x, y6, color='#0064AF', linestyle='-', label='龙尾后把手极角')

# 设置图形属性
plt.xlabel('时间t(s)', fontproperties=font2, size=12)
plt.ylabel('极角θ', fontproperties=font2, size=12)

plt.xticks(fontproperties=font, size=12)
plt.yticks(fontproperties=font, size=12)

# 显示图形
plt.grid(True, alpha=0.2)  # 显示网格
plt.legend(prop=font2)

# plt.savefig("Figure_3.png", dpi=300)
plt.show()
