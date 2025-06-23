"""
图4 -- 不同时刻下板凳龙把手的速度情况
"""
from src.dragon import *

dragon = Dragon()

x = np.arange(0, 224, 1)

dragon.set_time(300, need_vol=True)
y1 = dragon.vol

dragon.set_time(200, need_vol=True)
y2 = dragon.vol

dragon.set_time(100, need_vol=True)
y3 = dragon.vol

dragon.set_time(0, need_vol=True)
y4 = dragon.vol

plt.figure(figsize=(9, 5))

plt.plot(x, y4, color='#3F7874', label='0s', linewidth=2)
plt.plot(x, y3, color='#D6A419', label='100s', linewidth=2)
plt.plot(x, y2, color='#317FE3', label='200s', linewidth=2)
plt.plot(x, y1, color='#8669C2', label='300s', linewidth=2)

plt.fill_between(x, y1, 0, where=(y1 >= np.zeros_like(y1)), color='#8669C2', interpolate=True, alpha=0.5)
plt.fill_between(x, y2, y1, color='#317FE3', interpolate=True, alpha=0.5)
plt.fill_between(x, y3, y2, color='#D6A419', interpolate=True, alpha=0.5)
plt.fill_between(x, y4, y3, color='#3F7874', interpolate=True, alpha=0.5)

plt.margins(x=0)

plt.xlabel('板凳把手编号', fontproperties=font2, size=12)
plt.ylabel('速度(m/s)', fontproperties=font2, size=12)

plt.xticks(fontproperties=font, size=12)
plt.yticks(fontproperties=font, size=12)

plt.legend(prop=font)


plt.ylim(0.996, 1)
# plt.savefig("Figure_13.png", dpi=300)
plt.show()