"""
第1~3问板凳龙模型(无调头)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from matplotlib.font_manager import FontProperties

ben_wid = 0.3 # 板凳宽度(m)
ben_len_head = 3.41 # 龙头板长(m)
ben_len = 2.20 # 龙身及龙尾板长度(m)
hole_dis = 0.275 # 孔离板头距离(m)
hole_dia = 0.055 # 孔的直径(m)
dra_len = 223 # 板凳个数

font = FontProperties(fname="/System/Library/Fonts/Supplemental/Times New Roman.ttf")
font2 = FontProperties(fname="/System/Library/Fonts/Supplemental/Songti.ttc")


class Dragon():

    def __init__(self, d=0.55, v0=1, theta0=32 * pi): # d为螺距(m), v0为龙头前把手速度(m/s)

        self.d = d # 螺距(m)
        self.v0 = v0 # 龙头前把手速度(m/s)
        self.theta0 = theta0 # 0时刻龙头前把手角度

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

        if need_vol:
            self.vol = [self.v0]
            d_theta_n = -2 * pi * self.v0 / (self.d * (1 + theta_n**2)**0.5)

        # 求解龙身及龙尾前把手、龙尾后把手位置
        for n in range(223):
            if n != 0:
                theta_n = self.next_pos(theta_n)
            else:
                theta_n = self.next_pos(theta_n, is_head=True)
            self.ang.append(theta_n)
            self.pos.append((self.arc_x(theta_n), self.arc_y(theta_n)))

            if need_vol:
                theta_n0 = self.ang[n]
                theta_n1 = self.ang[n + 1]

                par1 = 2 * theta_n0 - 2 * theta_n1 * cos(theta_n1 - theta_n0) - 2 * theta_n0 * theta_n1 * sin(theta_n1 - theta_n0)
                par2 = 2 * theta_n1 - 2 * theta_n0 * cos(theta_n1 - theta_n0) + 2 * theta_n0 * theta_n1 * sin(theta_n1 - theta_n0)

                d_theta_n1 = -(par1 / par2) * d_theta_n

                v = self.d / (2 * pi) * abs(d_theta_n1) * (1 + theta_n1**2)**0.5
                self.vol.append(v)

                d_theta_n = d_theta_n1

    # 判断是否碰撞
    def judge_col(self):
        dis_min = 1 # 最短距离(m)

        # 找到距龙头一圈外的第一个板凳
        i_max = 0
        while self.ang[i_max] - self.ang[0] <= 2 * pi:
            i_max += 1

        for i in range(0, i_max + 1):

            l_i = 2.86 if i == 0 else 1.65

            # 找到距第i个板凳一圈外的第一个板凳j0
            j0 = i_max
            while self.ang[j0] - self.ang[i] <= 2 * pi:
                j0 += 1

            # 遍历j0及前后各两个板凳
            # 判断是否相碰
            for j in range(j0 - 2, j0 + 3):
                l_j = 1.65
                delta_xi = self.pos[i + 1][0] - self.pos[i][0]
                delta_yi = self.pos[i + 1][1] - self.pos[i][1]
                delta_xj = self.pos[j + 1][0] - self.pos[j][0]
                delta_yj = self.pos[j + 1][1] - self.pos[j][1]

                s = - delta_yi / l_i # sin(alpha)
                c = - delta_xi / l_i # cos(alpha)

                x1 = self.pos[i][0] + hole_dis * c - (ben_wid / 2) * s
                y1 = self.pos[i][1] + hole_dis * s + (ben_wid / 2) * c
                x2 = self.pos[i + 1][0] - hole_dis * c - (ben_wid / 2) * s
                y2 = self.pos[i + 1][1] - hole_dis * s + (ben_wid / 2) * c
                
                dis1 = abs(delta_yj * x1 - delta_xj * y1 + self.pos[j + 1][0] * self.pos[j][1] - self.pos[j][0] * self.pos[j + 1][1]) / l_j
                dis2 = abs(delta_yj * x2 - delta_xj * y2 + self.pos[j + 1][0] * self.pos[j][1] - self.pos[j][0] * self.pos[j + 1][1]) / l_j

                if dis1 <= ben_wid / 2 or dis2 <= ben_wid / 2:
                    return True, min(dis1, dis2)
                
                dis_min = min(dis_min, dis1, dis2)

        return False, dis_min
    
    # 判断龙头前把手到外圈板凳的最短距离
    def min_dis(self, t):

        self.set_time(t)
        dis_min = 1 # 最短距离(m)

        # 找到距龙头一圈外的第一个板凳
        j0 = 0
        while self.ang[j0] - self.ang[0] <= 2 * pi:
            j0 += 1

        l_i = 2.86
        for j in range(j0 - 2, j0 + 3):
            l_j = 1.65
            delta_xi = self.pos[1][0] - self.pos[0][0]
            delta_yi = self.pos[1][1] - self.pos[0][1]
            delta_xj = self.pos[j + 1][0] - self.pos[j][0]
            delta_yj = self.pos[j + 1][1] - self.pos[j][1]

            s = - delta_yi / l_i # sin(alpha)
            c = - delta_xi / l_i # cos(alpha)

            x1 = self.pos[0][0] + hole_dis * c - (ben_wid / 2) * s
            y1 = self.pos[0][1] + hole_dis * s + (ben_wid / 2) * c
            x2 = self.pos[1][0] - hole_dis * c - (ben_wid / 2) * s
            y2 = self.pos[1][1] - hole_dis * s + (ben_wid / 2) * c
                
            dis1 = abs(delta_yj * x1 - delta_xj * y1 + self.pos[j + 1][0] * self.pos[j][1] - self.pos[j][0] * self.pos[j + 1][1]) / l_j
            dis2 = abs(delta_yj * x2 - delta_xj * y2 + self.pos[j + 1][0] * self.pos[j][1] - self.pos[j][0] * self.pos[j + 1][1]) / l_j

            dis_min = min(dis_min, dis1, dis2)

        return dis_min
    
    # 判断龙头前把手到外圈板凳的最短距离
    def min_dis2(self, t):

        self.set_time(t)
        dis_min = 1 # 最短距离(m)

        # 找到距龙头一圈外的第一个板凳
        j0 = 0
        while self.ang[j0] - self.ang[0] <= 2 * pi:
            j0 += 1

        l_i = 2.86

        dist1 = []
        dist2 = []

        for j in range(j0 - 2, j0 + 3):
            l_j = 1.65
            delta_xi = self.pos[1][0] - self.pos[0][0]
            delta_yi = self.pos[1][1] - self.pos[0][1]
            delta_xj = self.pos[j + 1][0] - self.pos[j][0]
            delta_yj = self.pos[j + 1][1] - self.pos[j][1]

            s = - delta_yi / l_i # sin(alpha)
            c = - delta_xi / l_i # cos(alpha)

            x1 = self.pos[0][0] + hole_dis * c - (ben_wid / 2) * s
            y1 = self.pos[0][1] + hole_dis * s + (ben_wid / 2) * c
            x2 = self.pos[1][0] - hole_dis * c - (ben_wid / 2) * s
            y2 = self.pos[1][1] - hole_dis * s + (ben_wid / 2) * c
                
            dis1 = abs(delta_yj * x1 - delta_xj * y1 + self.pos[j + 1][0] * self.pos[j][1] - self.pos[j][0] * self.pos[j + 1][1]) / l_j
            dis2 = abs(delta_yj * x2 - delta_xj * y2 + self.pos[j + 1][0] * self.pos[j][1] - self.pos[j][0] * self.pos[j + 1][1]) / l_j

            dist1.append(dis1)
            dist2.append(dis2)

        return min(dist1), min(dist2)
    
    # 求解相碰时间及此时龙头前把手距中心的距离
    def time_col(self):

        time = 0

        # 设定初始状态
        self.set_time(time)

        while not self.judge_col()[0]:
            time += 1
            self.set_time(time)
        
        return time, self.arc_r(self.ang[0]) # 龙头前把手离中心距离

    # 求解龙头前把手到达掉头空间的时间
    def arr_time(self):
        return binary_search(self.if_arrive, 0, 421, incre=True)

    # 打印板凳龙状态(无板凳)
    def print_status(self):

        theta = np.linspace(0, 20 * 2 * pi, 1500)
        x = self.d / (2 * pi) * theta * np.cos(theta)
        y = self.d / (2 * pi) * theta * np.sin(theta)

        x_lst = [ben_pos[0] for ben_pos in self.pos]
        y_lst = [ben_pos[1] for ben_pos in self.pos]

        plt.figure()
        plt.plot(x, y, color='black', linewidth=2, alpha=0.5)
        plt.scatter(x_lst, y_lst, color='red', marker='.')

        plt.xlabel('X', fontproperties=font, size=12)
        plt.ylabel('Y', fontproperties=font, size=12)

        plt.xticks(fontproperties=font, size=12)
        plt.yticks(fontproperties=font, size=12)

        plt.grid(True, alpha=0.3)
        plt.show()

    # 打印板凳龙图像(有板凳)
    def print_img(self, save_pth=None, magn=None, show=True, circle=False):

        # save_pth 保存路径，默认不保存
        # magn 是否放大中心，默认不放大

        a = hole_dis
        b = ben_wid / 2

        # 初始化图像
        plt.figure(figsize=(8, 8))

        # 绘制把手
        theta = np.linspace(0, 16 * 2 * pi, 1500)
        x = self.d / (2 * pi) * theta * np.cos(theta)
        y = self.d / (2 * pi) * theta * np.sin(theta)

        x_lst = [ben_pos[0] for ben_pos in self.pos]
        y_lst = [ben_pos[1] for ben_pos in self.pos]

        plt.plot(x, y, color='black', linewidth=2, alpha=0.2)
        plt.scatter(x_lst, y_lst, color='red', marker='.')

        # 绘制板凳
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

        if circle:
            circle = plt.Circle((0, 0), 4.5, color='yellow', fill=True, alpha=0.5)
            plt.gca().add_artist(circle)

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
        plt.grid(True, alpha=0.3)     # 显示网格
        if save_pth:
            plt.savefig(save_pth, dpi=300)
        if show:
            plt.show()
        plt.close()

    # (辅助函数）求解龙头前把手角度
    def head_pos(self, t):
        par1 = self.theta0 * (1 + self.theta0**2)**0.5
        par2 = log(self.theta0 + (1 + self.theta0**2)**0.5)
        par3 = 4 * pi * self.v0 * t / self.d

        def fn(x):
            return par1 + par2 - par3 - x * (1 + x**2)**0.5 - log(x + (1 + x**2)**0.5)
        
        return binary_search(fn, 0, self.theta0, incre=False)
    
    # (辅助函数) 龙头前把手路径图
    def head_img(self, save_pth=None):

        theta = np.linspace(0, 16 * 2 * pi, 1500)
        x = self.d / (2 * pi) * theta * np.cos(theta)
        y = self.d / (2 * pi) * theta * np.sin(theta)

        x_lst = []
        y_lst = []

        for t in np.arange(0, 301):
            ang = self.head_pos(t)
            x_lst.append(self.arc_x(ang))
            y_lst.append(self.arc_y(ang))

        plt.figure(figsize=(5, 5))
        plt.plot(x, y, color='#956866', linewidth=2, alpha=0.5)
        plt.scatter(x_lst, y_lst, color='red', marker='.')

        plt.xlabel('X', fontproperties=font, size=12)
        plt.ylabel('Y', fontproperties=font, size=12)

        plt.xticks(fontproperties=font, size=12)
        plt.yticks(fontproperties=font, size=12)

        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        if save_pth is not None:
            plt.savefig(save_pth, dpi=300)
        plt.show()

    
    # (辅助函数）判断龙头前把手是否进入调头空间
    def if_arrive(self, t):
        ang = self.head_pos(t)
        if self.arc_r(ang) <= 4.5:
            return 1
        else:
            return -1
    
    # （辅助函数）求解龙身及龙尾把手角度
    def next_pos(self, theta_n, is_head=False): # theta_n为前一把手角度
        l = 2.86 if is_head else 1.65 # 两孔距离(m)
        para1 = (2 * pi * l / self.d)**2

        def fn(x):
            return theta_n**2 + x**2 - 2 * theta_n * x * cos(x - theta_n) - para1
        
        return binary_search(fn, theta_n, theta_n + pi, incre=True)

    # （辅助函数）角度转换为x坐标
    def arc_x(self, arc):
        return self.d / (2 * pi) * arc * cos(arc)

    # （辅助函数）角度转换为y坐标
    def arc_y(self, arc):
        return self.d / (2 * pi) * arc * sin(arc)
    
    # （辅助函数）角度转换为半径
    def arc_r(self, arc):
        return self.d / (2 * pi) * arc


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


# 临时测试函数
def test():
    dragon = Dragon()
    dragon.set_time(412)
    dragon.print_img(magn=4)


if __name__ == '__main__':
    test()