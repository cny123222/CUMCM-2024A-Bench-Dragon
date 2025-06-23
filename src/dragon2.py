"""
第4~5问板凳龙模型(含调头)
"""

"""
请注意
本文档中所有把手角度均为·二元组·
第一个数 表示 所在曲线(1/2/3/4)
第二个数 表示 极角
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import *
from matplotlib.font_manager import FontProperties
import os

font = FontProperties(fname="/System/Library/Fonts/Supplemental/Times New Roman.ttf")
font2 = FontProperties(fname="/System/Library/Fonts/Supplemental/Songti.ttc")

ben_wid = 0.3 # 板凳宽度(m)
ben_len_head = 3.41 # 龙头板长(m)
ben_len = 2.20 # 龙身及龙尾板长度(m)
hole_dis = 0.275 # 孔离板头距离(m)
hole_dia = 0.055 # 孔的直径(m)
dra_len = 223 # 板凳个数

d = 1.7 # 螺距(m)
r = 4.5 # 调头空间半径(m)

theta = 2 * pi * r / d # 图中theta角
alpha = atan(theta) # 图中alpha角

R1 = 3 / sin(alpha) # 大圆半径
R2 = 3 / (2 * sin(alpha)) # 小圆半径

# 大圆圆心坐标
O1x = r * cos(theta) - R1 * sin(theta + alpha)
O1y = r * sin(theta) + R1 * cos(theta + alpha)

# 小圆圆心坐标
O2x = -r * cos(theta) + R2 * sin(theta + alpha)
O2y = -r * sin(theta) - R2 * cos(theta + alpha)

# 盘入切点坐标
x1 = d / (2 * pi) * theta * cos(theta)
y1 = d / (2 * pi) * theta * sin(theta)

# 盘出切点坐标
x3 = -x1
y3 = -y1

# 两圆切点坐标
x2 = x1 / 3 + x3 * 2 / 3
y2 = y1 / 3 + y3 * 2 / 3


class Dragon():

    def __init__(self, v0=1): # v0为龙头前把手速度(m/s)

        self.d = d # 螺距(m)
        self.v0 = v0 # 龙头前把手速度(m/s)

        self.ang = None # 把手角度列表
        self.pos = None # 把手坐标列表
        self.vol = None # 把手速度列表

        self.time = None # 当前时间

    # 设定时间t(s)
    def set_time(self, t, need_vol=False):

        self.time = t # 更新当前时间

        # 求解龙头前把手位置
        theta_n = self.head_pos(t)
        self.ang = [theta_n]
        self.pos = [(self.arc_x(theta_n), self.arc_y(theta_n))]

        # 求解龙身及龙尾前把手、龙尾后把手位置
        for n in range(223):
            if n != 0:
                theta_n = self.next_pos(theta_n)
            else:
                theta_n = self.next_pos(theta_n, is_head=True)
            self.ang.append(theta_n)
            self.pos.append((self.arc_x(theta_n), self.arc_y(theta_n)))

        # 求解各把手速度
        if need_vol == False:
            return
        
        # 求解龙头前把手速度
        if self.ang[0][0] == 1:
            d_theta = -2 * pi * self.v0 / d / sqrt(1 + self.ang[0][1] ** 2)
        elif self.ang[0][0] == 2:
            d_theta = - self.v0 / R1
        elif self.ang[0][0] == 3:
            d_theta = self.v0 / R2
        elif self.ang[0][0] == 4:
            d_theta = 2 * pi * self.v0 / d / sqrt(1 + (self.ang[0][1] + pi) ** 2)
        self.vol = [self.v0]
        d_theta_1 = d_theta

        # 求解龙身及龙尾前把手、龙尾后把手速度
        for n in range(223):
            l = 2.86 if n == 0 else 1.65
            theta1 = self.ang[n][1]
            theta2 = self.ang[n + 1][1]
            (x1, y1) = self.pos[n]
            (x2, y2) = self.pos[n + 1]

            # 前一把手位于盘入螺线(当前把手位于盘入螺线)
            if self.ang[n][0] == 1:
                par1 = theta1 - theta2 * cos(theta2 - theta1) - theta1 * theta2 * sin(theta2 - theta1)
                par2 = theta2 - theta1 * cos(theta2 - theta1) + theta1 * theta2 * sin(theta2 - theta1)
                d_theta1 = -par1 / par2 * d_theta
                self.vol.append(self.d / (2 * pi) * abs(d_theta1) * sqrt(1 + theta2 ** 2))

            # 前一把手位于大圆圆弧
            elif self.ang[n][0] == 2:

                # 当前把手位于大圆圆弧
                if self.ang[n + 1][0] == 2:
                    d_theta1 = d_theta
                    self.vol.append(self.vol[n])

                # 当前把手位于盘入螺线
                elif self.ang[n + 1][0] == 1:
                    x1d = self.vol[n] * sin(theta1)
                    y1d = -self.vol[n] * cos(theta1)
                    par1 = (x1 - d / (2 * pi) * theta2 * cos(theta2)) * x1d + (y1 - d / (2 * pi) * theta2 * sin(theta2)) * y1d
                    par2 = (x1 - d / (2 * pi) * theta2 * cos(theta2)) * (cos(theta2) - theta2 * sin(theta2)) + (y1 - d / (2 * pi) * theta2 * sin(theta2)) * (sin(theta2) + theta2 * cos(theta2))
                    d_theta1 = 2 * pi / d * par1 / par2
                    self.vol.append(self.d / (2 * pi) * abs(d_theta1) * sqrt(1 + theta2 ** 2))
                
            # 前一把手位于小圆圆弧
            elif self.ang[n][0] == 3:

                # 当前把手位于小圆圆弧
                if self.ang[n + 1][0] == 3:
                    d_theta1 = d_theta
                    self.vol.append(self.vol[n])

                # 当前把手位于大圆圆弧
                elif self.ang[n + 1][0] == 2:
                    x1d = self.vol[n] * sin(theta1)
                    y1d = -self.vol[n] * cos(theta1)
                    par1 = (x1 - O1x - R1 * cos(theta2)) * x1d + (y1 - O1y - R1 * sin(theta2)) * y1d
                    par2 = (x1 - O1x - R1 * cos(theta2)) * sin(theta2) - (y1 - O1y - R1 * sin(theta2)) * cos(theta2)
                    d_theta1 = -par1 / (R1 * par2)
                    self.vol.append(abs(d_theta1) * R1)

            # 前一把手位于盘出螺线
            elif self.ang[n][0] == 4:

                # 当前把手位于盘出螺线
                if self.ang[n + 1][0] == 4:
                    theta1, theta2 = theta1 + pi, theta2 + pi
                    par1 = theta1 - theta2 * cos(theta2 - theta1) - theta1 * theta2 * sin(theta2 - theta1)
                    par2 = theta2 - theta1 * cos(theta2 - theta1) + theta1 * theta2 * sin(theta2 - theta1)
                    d_theta1 = -par1 / par2 * d_theta
                    self.vol.append(self.d / (2 * pi) * abs(d_theta1) * sqrt(1 + theta2 ** 2))
                    theta1, theta2 = theta1 - pi, theta2 - pi

                # 当前把手位于小圆圆弧
                elif self.ang[n + 1][0] == 3:
                    beta = theta1 + atan(theta1 + pi)
                    x1d = self.vol[n] * cos(beta)
                    y1d = self.vol[n] * sin(beta)
                    par1 = (x1 - O2x + R2 * cos(theta2)) * x1d + (y1 - O2y + R2 * sin(theta2)) * y1d
                    par2 = (x1 - O2x + R2 * cos(theta2)) * sin(theta2) - (y1 - O2y + R2 * sin(theta2)) * cos(theta2)
                    d_theta1 = par1 / (R2 * par2)
                    self.vol.append(abs(d_theta1) * R2)
                    
            d_theta = d_theta1

    # 打印板凳龙状态(无板凳)
    def print_status(self):
        
        # 计算盘入盘出螺线
        ang = np.linspace(theta, 10 * 2 * pi, 1000)
        xx1 = d / (2 * pi) * ang * np.cos(ang)
        yy1 = d / (2 * pi) * ang * np.sin(ang)
        xx2 = -xx1
        yy2 = -yy1

        plt.figure(figsize=(8, 8))

        # 绘制盘入盘出螺线
        plt.plot(xx1, yy1, color='red', linewidth=2, label='盘入螺线', alpha=0.5)
        plt.plot(xx2, yy2, color='blue', linewidth=2, label='盘出螺线', alpha=0.5)
        plt.plot([], [], color='black', linewidth=2, label='调头路径', alpha=0.5)

        # 绘制调头空间
        circle = plt.Circle((0, 0), 4.5, color='yellow', fill=True, alpha=0.4)
        plt.gca().add_artist(circle)

        # 绘制两圆圆心
        # plt.scatter(O1x, O1y, color='black', marker='.', alpha=0.5)
        # plt.scatter(O2x, O2y, color='black', marker='.', alpha=0.5)

        # 绘制调头路径
        arc1 = patches.Arc((O1x, O1y), width=R1 * 2, height=R1 * 2, theta1=np.rad2deg(theta - alpha - pi / 2), theta2=np.rad2deg(theta + alpha - pi / 2), color='black', linewidth=2, alpha=0.5) 
        plt.gca().add_patch(arc1)

        arc2 = patches.Arc((O2x, O2y), width=R2 * 2, height=R2 * 2, theta1=np.rad2deg(theta - alpha + pi / 2), theta2=np.rad2deg(theta + alpha + pi / 2), color='black', linewidth=2, alpha=0.5) 
        plt.gca().add_patch(arc2)

        x_lst = [ben_pos[0] for ben_pos in self.pos]
        y_lst = [ben_pos[1] for ben_pos in self.pos]

        plt.scatter(x_lst, y_lst, color='black', marker='.')

        plt.axis('equal')
        plt.xlabel('X', fontproperties=font, size=12)
        plt.ylabel('Y', fontproperties=font, size=12)
        plt.xticks(fontproperties=font, size=12)
        plt.yticks(fontproperties=font, size=12)

        plt.grid(True, alpha=0.3)
        plt.show()

    # 打印板凳龙图像(有板凳)
    def print_img(self, save_pth=None, magn=None, show=True):

        # save_pth 保存路径，默认不保存
        # magn 是否放大中心，默认不放大
        # show 是否展示，默认展示

        # 初始化图像
        plt.figure(figsize=(8, 8))

        # 计算盘入盘出螺线
        ang = np.linspace(theta, 10 * 2 * pi, 1000)
        xx1 = d / (2 * pi) * ang * np.cos(ang)
        yy1 = d / (2 * pi) * ang * np.sin(ang)
        xx2 = -xx1
        yy2 = -yy1

        # 绘制盘入盘出螺线
        plt.plot(xx1, yy1, color='red', linewidth=2, label='盘入螺线', alpha=0.5)
        plt.plot(xx2, yy2, color='blue', linewidth=2, label='盘出螺线', alpha=0.5)
        plt.plot([], [], color='black', linewidth=2, label='调头路径', alpha=0.5)

        # 绘制调头空间
        circle = plt.Circle((0, 0), 4.5, color='yellow', fill=True, alpha=0.4)
        plt.gca().add_artist(circle)

        # 绘制两圆圆心
        # plt.scatter(O1x, O1y, color='black', marker='.', alpha=0.5)
        # plt.scatter(O2x, O2y, color='black', marker='.', alpha=0.5)

        # 绘制调头路径
        arc1 = patches.Arc((O1x, O1y), width=R1 * 2, height=R1 * 2, theta1=np.rad2deg(theta - alpha - pi / 2), theta2=np.rad2deg(theta + alpha - pi / 2), color='black', linewidth=2, alpha=0.5) 
        plt.gca().add_patch(arc1)

        arc2 = patches.Arc((O2x, O2y), width=R2 * 2, height=R2 * 2, theta1=np.rad2deg(theta - alpha + pi / 2), theta2=np.rad2deg(theta + alpha + pi / 2), color='black', linewidth=2, alpha=0.5) 
        plt.gca().add_patch(arc2)

        # 绘制把手
        x_lst = [ben_pos[0] for ben_pos in self.pos]
        y_lst = [ben_pos[1] for ben_pos in self.pos]

        plt.scatter(x_lst, y_lst, color='black', marker='.')

        # 绘制板凳
        a = hole_dis
        b = ben_wid / 2

        for i in range(223):

            li = 2.86 if i == 0 else 1.65

            delta_xi = self.pos[i + 1][0] - self.pos[i][0]
            delta_yi = self.pos[i + 1][1] - self.pos[i][1]

            s = - delta_yi / li # sin(alpha)
            c = - delta_xi / li # cos(alpha)

            x1 = self.pos[i][0] + a * c - b * s
            y1 = self.pos[i][1] + a * s + b * c
            x2 = self.pos[i + 1][0] - a * c - b * s
            y2 = self.pos[i + 1][1] - a * s + b * c
            x3 = self.pos[i + 1][0] - a * c + b * s
            y3 = self.pos[i + 1][1] - a * s - b * c
            x4 = self.pos[i][0] + a * c + b * s
            y4 = self.pos[i][1] + a * s - b * c

            rect = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

            x, y = zip(*rect)
            plt.plot(x + (x[0],), y + (y[0],), 'b-')

        # 设置图形属性
        plt.xlabel('X', fontproperties=font, size=12)
        plt.ylabel('Y', fontproperties=font, size=12)

        plt.xticks(fontproperties=font, size=12)
        plt.yticks(fontproperties=font, size=12)

        # 放大区域
        if magn is not None:
            plt.xlim(-magn, magn)
            plt.ylim(-magn, magn)

        # 显示图形
        # plt.axis('equal')  # 设置坐标轴等比例
        plt.grid(True, alpha=0.3)
        if save_pth:
            plt.savefig(save_pth, dpi=300)
        if show:
            plt.show()
        plt.close()

    # 保存板凳龙图像
    def save_imgs(self, start_t, end_t, save_pth="photos2", magn=8):
        for time in np.arange(start_t, end_t + 1, 1):
            print("time = ", time)
            self.set_time(time)
            self.print_img(os.path.join(save_pth, f"{time}.png"), magn=magn, show=False)

    # (辅助函数）求解龙头前把手角度
    def head_pos(self, t):

        # 位于盘入螺线
        if t <= 0:
            par1 = theta * (1 + theta**2)**0.5
            par2 = log(theta + (1 + theta**2)**0.5)
            par3 = 4 * pi * self.v0 * t / self.d

            def fn(x):
                return par1 + par2 - par3 - x * (1 + x**2)**0.5 - np.log(x + (1 + x**2)**0.5)
        
            return 1, binary_search(fn, theta, theta * 2, incre=False)

        # 位于大圆圆弧
        elif t < 6 * alpha / (sin(alpha) * self.v0):
            theta_n = theta + alpha - pi / 2 - self.v0 * t / R1
            return 2, theta_n

        # 位于小圆圆弧
        elif t < 9 * alpha / (sin(alpha) * self.v0):
            theta_n = theta - alpha - pi / 2 + self.v0 * (t - 2 * alpha * R1 / self.v0) / R2
            return 3, theta_n

        # 位于盘出螺线
        else:
            par1 = theta * (1 + theta**2)**0.5
            par2 = log(theta + (1 + theta**2)**0.5)
            par3 = 4 * pi * self.v0 * (t - 2 * alpha * (R1 + R2) / self.v0) / self.d

            def fn(x):
                return x * (1 + x**2)**0.5 + np.log(x + (1 + x**2)**0.5) - par1 - par2 - par3
        
            return 4, binary_search(fn, theta, theta * 20, incre=True) - pi
        
    # （辅助函数）龙头前把手路径图
    def head_img(self, save_pth=None):

        # 计算盘入盘出螺线
        ang = np.linspace(theta, 6 * 2 * pi, 1000)
        x1 = d / (2 * pi) * ang * np.cos(ang)
        y1 = d / (2 * pi) * ang * np.sin(ang)
        x2 = -x1
        y2 = -y1

        plt.figure(figsize=(8, 8))

        # 绘制盘入盘出螺线
        plt.plot(x1, y1, color='red', linewidth=2, label='盘入螺线', alpha=0.5)
        plt.plot(x2, y2, color='blue', linewidth=2, label='盘出螺线', alpha=0.5)
        plt.plot([], [], color='black', linewidth=2, label='调头路径', alpha=0.5)

        # 绘制调头空间
        circle = plt.Circle((0, 0), 4.5, color='yellow', fill=True, alpha=0.4)
        plt.gca().add_artist(circle)

        # 绘制两圆圆心
        plt.scatter(O1x, O1y, color='black', marker='.', alpha=0.5)
        plt.scatter(O2x, O2y, color='black', marker='.', alpha=0.5)

        # 绘制调头路径
        arc1 = patches.Arc((O1x, O1y), width=R1 * 2, height=R1 * 2, theta1=np.rad2deg(theta - alpha - pi / 2), theta2=np.rad2deg(theta + alpha - pi / 2), color='black', linewidth=2, alpha=0.5) 
        plt.gca().add_patch(arc1)

        arc2 = patches.Arc((O2x, O2y), width=R2 * 2, height=R2 * 2, theta1=np.rad2deg(theta - alpha + pi / 2), theta2=np.rad2deg(theta + alpha + pi / 2), color='black', linewidth=2, alpha=0.5) 
        plt.gca().add_patch(arc2)

        x = []
        y = []

        for t in np.arange(-100, 101):
            ang = self.head_pos(t)
            x.append(self.arc_x(ang))
            y.append(self.arc_y(ang))

        plt.scatter(x, y, marker='*', color='black')

        plt.axis('equal')
        plt.xlabel('X', fontproperties=font, size=12)
        plt.ylabel('Y', fontproperties=font, size=12)
        plt.xticks(fontproperties=font, size=12)
        plt.yticks(fontproperties=font, size=12)

        plt.legend(prop=font2)
        plt.grid(True, alpha=0.2)
        if save_pth is not None:
            plt.savefig(save_pth, dpi=300)
        plt.show()

    # （辅助函数）求解龙身及龙尾把手角度
    def next_pos(self, theta_n, is_head=False): # theta_n为前一把手角度

        l = 2.86 if is_head else 1.65 # 两孔距离(m)

        # 前一把手位于盘入螺线（当前把手位于盘入螺线）
        if theta_n[0] == 1:
            theta_n = theta_n[1]
            par1 = (2 * pi * l / self.d)**2

            def fn(x):
                return theta_n**2 + x**2 - 2 * theta_n * x * cos(x - theta_n) - par1
            
            return 1, binary_search(fn, theta_n, theta_n + pi, incre=True)
        
        # 前一把手位于大圆圆弧
        elif theta_n[0] == 2:
            xn = self.arc_x(theta_n)
            yn = self.arc_y(theta_n)

            # 当前把手位于盘入螺线
            if dist((xn, yn), (x1, y1)) <= l:
                
                def fn(x):
                    par1 = d / (2 * pi) * x * np.cos(x)
                    par2 = d / (2 * pi) * x * np.sin(x)
                    return (xn - par1)**2 + (yn - par2)**2 - l**2
                
                return 1, binary_search(fn, theta_n[1], theta_n[1] + pi, incre=True)

            # 当前把手位于大圆圆弧
            else:
                delta_theta = 2 * asin(l / (2 * R1))
                return 2, theta_n[1] + delta_theta

        # 前一把手位于小圆圆弧
        elif theta_n[0] == 3:
            xn = self.arc_x(theta_n)
            yn = self.arc_y(theta_n)

            # 当前把手位于大圆圆弧
            if dist((xn, yn), (x2, y2)) <= l:
                
                def fn(x):
                    par1 = xn - O1x - R1 * np.cos(x)
                    par2 = yn - O1y - R1 * np.sin(x)
                    return par1**2 + par2**2 - l**2
                
                return 2, binary_search(fn, theta - alpha - pi / 2, theta + alpha - pi / 2, incre=True)
                
            # 当前把手位于小圆圆弧
            else:
                delta_theta = 2 * asin(l / (2 * R2))
                return 3, theta_n[1] - delta_theta

        # 前一把手位于盘出螺线
        elif theta_n[0] == 4:
            xn = self.arc_x(theta_n)
            yn = self.arc_y(theta_n)
            
            # 当前把手位于小圆圆弧
            if dist((xn, yn), (x3, y3)) <= l and theta_n[1] <= theta:
                
                def fn(x):
                    par1 = xn - O2x + R2 * np.cos(x)
                    par2 = yn - O2y + R2 * np.sin(x)
                    return par1**2 + par2**2 - l**2
                
                return 3, binary_search(fn, theta - alpha - pi / 2, theta + alpha - pi / 2, incre=False)
                
            # 当前把手位于盘出螺线
            else:
                theta_n = theta_n[1]
                par1 = (2 * pi * l / self.d)**2

                def fn(x):
                    return (theta_n + pi)**2 + (x + pi)**2 - 2 * (theta_n + pi) * (x + pi) * cos(x - theta_n) - par1
            
                return 4, binary_search(fn, theta_n - pi, theta_n, incre=False)

    # （辅助函数）角度转换为x坐标
    def arc_x(self, arc):
        
        # 位于盘入螺线
        if arc[0] == 1:
            return self.d / (2 * pi) * arc[1] * cos(arc[1])
        
        # 位于大圆圆弧
        elif arc[0] == 2:
            return O1x + R1 * cos(arc[1])

        # 位于小圆圆弧
        elif arc[0] == 3:
            return O2x - R2 * cos(arc[1])
        
        # 位于盘出螺线
        elif arc[0] == 4:
            return self.d  / (2 * pi) * (arc[1] + pi) * cos(arc[1])

    # （辅助函数）角度转换为y坐标
    def arc_y(self, arc):

        # 位于盘入螺线
        if arc[0] == 1:
            return self.d / (2 * pi) * arc[1] * sin(arc[1])
        
        # 位于大圆圆弧
        elif arc[0] == 2:
            return O1y + R1 * sin(arc[1])

        # 位于小圆圆弧
        elif arc[0] == 3:
            return O2y - R2 * sin(arc[1])
        
        # 位于盘出螺线
        elif arc[0] == 4:
            return self.d  / (2 * pi) * (arc[1] + pi) * sin(arc[1])


# 二分搜索函数
def binary_search(fn, lb, ub, incre, eps=1e-8):

    cur = (lb + ub) / 2

    while ub - lb > eps:

        cur_res = fn(cur)

        # 若函数递增
        if incre:
            if cur_res < 0:
                lb = cur
            elif cur_res > 0:
                ub = cur
        
        # 若函数递减
        else:
            if cur_res > 0:
                lb = cur
            elif cur_res < 0:
                ub = cur

        cur = (lb + ub) / 2

    return cur

# 求两点间距离
def dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


# 临时测试函数
def test():
    dragon = Dragon()
    # dragon.set_time(300)
    # dragon.print_img(magn=8)
    dragon.head_img("Figure_8_new.png")


if __name__ == '__main__':
    test()