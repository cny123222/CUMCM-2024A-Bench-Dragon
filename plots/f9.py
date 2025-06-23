"""
图9 -- 最内圈板凳的外侧顶点距其他板凳中心线的最短距离与时间 t 的关系
"""
from src.dragon import *

dragon = Dragon()

x = []
y = []

for time in np.arange(380, 420, 0.1):

    print("time = ", time)

    x.append(time)
    y.append(dragon.min_dis(time))

plt.figure()
plt.plot(x, y, color='#317FE3')

plt.axhline(y=0.15, color='r', linestyle='--')

plt.xlabel('时间t(s)', fontproperties=font2, size=12)
plt.ylabel('A点和B点距其他板凳中心线的最短距离(m)', fontproperties=font2, size=12)

plt.xticks(fontproperties=font, size=12)
plt.yticks(fontproperties=font, size=12)

plt.grid(True, alpha=0.3)
# plt.savefig("Figure_12.png", dpi=300)
plt.show()
    