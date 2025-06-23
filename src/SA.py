import numpy as np
import matplotlib.pyplot as plt
import random


def SA(inputfun, lb, ub):

    # 模拟退火算法的初始参数
    initT = 1       # 初始温度
    minT = 1e-3     # 最低温度
    iterL = 50      # 每个温度下的迭代次数
    delta = 0.95    # 温度衰减系数
    k = 1           # 玻尔兹曼常数

    # 随机选取初始点
    initx = random.uniform(lb, ub)

    nowt = initT  # 将当前温度设为初始温度
    print("初始解:", initx)

    # 准备数据以绘制目标函数图像
    xx = np.linspace(lb, ub, 1000)
    # yy = inputfun(xx)
    yy = np.array([inputfun(x) for x in xx])

    # 创建绘图以可视化目标函数
    plt.figure()
    plt.plot(xx, yy)
    plt.plot(initx, inputfun(initx), 'o')

    # plt.show()

    # 模拟退火算法来寻找函数的最小值
    while nowt > minT:  # 当温度大于最低温度时继续
        print("当前温度:", nowt)
        for i in np.arange(1, iterL, 1):
            funVal = inputfun(initx)  # 计算当前点的目标函数值
            xnew = initx + (2 * np.random.rand() - 1)  # 在范围内生成一个新的随机点
            if xnew >= lb and xnew <= ub:  # 检查新点是否在允许范围内
                funnew = inputfun(xnew)  # 计算新点的目标函数值
                res = funnew - funVal  # 计算函数值之间的差异
                if res < 0:  # 如果新点的函数值更低（更优）
                    initx = xnew  # 移动到新点
                else:
                    p = np.exp(-(res) / (k * nowt))  # 根据玻尔兹曼分布计算接受概率
                    if np.random.rand() < p:  # 根据随机值决定是否以一定概率移动到新点
                        initx = xnew  # 移动到新点
        nowt = nowt * delta  # 使用衰减系数降低温度，为下一次迭代做准备

    print("最优解:", initx)
    print("最优值:", inputfun(initx))
    plt.plot(initx, inputfun(initx), '*r')  # 在图上标记最优解
    plt.show()  # 显示绘图结果

    return initx, inputfun(initx)
