"""
图18 -- 板凳龙调头路径示意图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import *
from src.dragon import *

d = 1.7
r = 4.5

theta = 2 * pi * r / d
alpha = atan(theta)

R1 = 3 / sin(alpha)
R2 = 3 / (2 * sin(alpha))

O1x = r * cos(theta) - R1 * sin(theta + alpha)
O1y = r * sin(theta) + R1 * cos(theta + alpha)

O2x = -r * cos(theta) + R2 * sin(theta + alpha)
O2y = -r * sin(theta) - R2 * cos(theta + alpha)
 
# 创建角度数组
ang1 = np.linspace(theta, 8 * 2 * pi, 1000)

# 计算螺线的参数化方程
x1 = d / (2 * pi) * ang1 * np.cos(ang1)
y1 = d / (2 * pi) * ang1 * np.sin(ang1)

x2 = -x1
y2 = -y1

# 绘制螺线
plt.figure()

plt.plot(x1, y1, color='red', linewidth=2, label='盘入螺线')
plt.plot(x2, y2, color='blue', linewidth=2, label='盘出螺线')
plt.plot([], [], color='black', linewidth=2, label='调头路径')

circle = plt.Circle((0, 0), 4.5, color='yellow', fill=True, alpha=0.5)
plt.gca().add_artist(circle)

plt.scatter(O1x, O1y, color='black', marker='.')
plt.scatter(O2x, O2y, color='black', marker='.')

arc1 = patches.Arc((O1x, O1y), width=R1 * 2, height=R1 * 2, theta1=np.rad2deg(theta - alpha - pi / 2), theta2=np.rad2deg(theta + alpha - pi / 2), color='black', linewidth=2) 
plt.gca().add_patch(arc1)

arc2 = patches.Arc((O2x, O2y), width=R2 * 2, height=R2 * 2, theta1=np.rad2deg(theta - alpha + pi / 2), theta2=np.rad2deg(theta + alpha + pi / 2), color='black', linewidth=2) 
plt.gca().add_patch(arc2)

plt.axis('equal')
plt.xlabel('X', fontproperties=font, size=12)
plt.ylabel('Y', fontproperties=font, size=12)
plt.xticks(fontproperties=font, size=12)
plt.yticks(fontproperties=font, size=12)

plt.legend(prop=font2)
plt.grid(True, alpha=0.2)

# plt.savefig("Figure_7.png", dpi=300)
plt.show()