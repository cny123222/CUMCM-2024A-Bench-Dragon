"""
求解龙头前把手能进入调头空间的最小螺距
"""
import random
from dragon import *
from sko.PSO import PSO
from sko.tools import set_run_mode

from SA import *

def min_pitch(domain, method="PSO", fig1=False, fig2=False, fig3=False):

    if fig3:
        plt.figure()
        x = []
        y = []

    for d in domain:
        dragon = Dragon(d=d)
        time = dragon.arr_time()
        print(f'当前螺距为{d}m')

        set_run_mode(dragon.min_dis, 'multiprocessing')
        if method == "PSO":  
            pso = PSO(func=dragon.min_dis, n_dim=1, pop=40, max_iter=50, lb=0, ub=time, w=0.8, c1=0.5, c2=0.5)
            pso.run()
            dist = pso.gbest_y
            print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
        elif method == "SA":
            best_x, dist = SA(dragon.min_dis, 0, 300)
            print('best_x is ', best_x, 'best_y is', dist)

        # 作图1
        if fig1:
            xx = np.linspace(0, 250, 1000)
            yy = np.array([dragon.min_dis(x) for x in xx])
            plt.figure()
            plt.plot(xx, yy)
            plt.plot(pso.gbest_x, pso.gbest_y, '*r')
            plt.xlabel('时间t(s)', fontproperties=font2, size=12)
            plt.ylabel('A点和B点距其他板凳中心线的最短距离', fontproperties=font2, size=12)
            plt.xticks(fontproperties=font, size=12)
            plt.yticks(fontproperties=font, size=12)
            plt.grid(True, alpha=0.3)
            # plt.savefig("Figure_4_new.png", dpi=300)
            plt.show()

        # 作图2
        if fig2:
            plt.figure(figsize=(7, 5))
            plt.plot(pso.gbest_y_hist)
            plt.xlabel('粒子群迭代次数', fontproperties=font2, size=12)
            plt.ylabel('全局最优解的函数值', fontproperties=font2, size=12)
            plt.xticks(fontproperties=font, size=12)
            plt.yticks(fontproperties=font, size=12)
            # plt.savefig("Figure_6.png", dpi=300)
            plt.show()

        # 作图3
        if fig3:
            x.append(d)
            y.append(dist)

        # if not fig3:
        #     if dist > 0.15:
        #         break

    if fig3:
        plt.plot(x, y, marker='*')

        plt.axhline(y=0.15, color='r', linestyle='--')

        plt.xlabel('螺距d(m)', fontproperties=font2, size=12)
        plt.ylabel('龙头前把手到达调头空间前dis(t)的最小值(m)', fontproperties=font2, size=10)

        plt.xticks(fontproperties=font, size=12)
        plt.yticks(fontproperties=font, size=12)

        plt.grid(True, alpha=0.3)
        # plt.savefig("Figure_14.png", dpi=300)
        plt.show()

def multi_fig(iter_time, save_pth=None):

    plt.figure(figsize=(7, 5))

    for _ in range(iter_time):

        print(f"已迭代{_}次")

        dragon = Dragon(d=0.450337)
        time = dragon.arr_time()

        pso = PSO(func=dragon.min_dis, n_dim=1, pop=40, max_iter=50, lb=0, ub=time, w=0.8, c1=0.5, c2=0.5)
        pso.run()

        plt.plot(pso.gbest_y_hist, color='#317FE3')

    plt.xlabel('粒子群迭代次数', fontproperties=font2, size=12)
    plt.ylabel('全局最优解的函数值', fontproperties=font2, size=12)
    plt.xticks(fontproperties=font, size=12)
    plt.yticks(fontproperties=font, size=12)

    plt.grid(True, alpha=0.3)
    if save_pth is not None:
        plt.savefig(save_pth, dpi=300)
    plt.show()


def min_pitch2(domain, method="PSO", fig1=False, fig2=False, fig3=False):


    for d in domain:
        dragon = Dragon(d=d)
        # time = dragon.arr_time()
        # print(f'当前螺距为{d}m')

        # set_run_mode(dragon.min_dis, 'multiprocessing')
        # if method == "PSO":  
        #     pso = PSO(func=dragon.min_dis, n_dim=1, pop=40, max_iter=50, lb=0, ub=time, w=0.8, c1=0.5, c2=0.5)
        #     pso.run()
        #     dist = pso.gbest_y
        #     print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
        # elif method == "SA":
        #     best_x, dist = SA(dragon.min_dis, 0, 300)
        #     print('best_x is ', best_x, 'best_y is', dist)

        if fig1:
            xx = np.linspace(0, 200, 1000)
            y1 = np.array([dragon.min_dis2(x)[0] for x in xx])
            y2 = np.array([dragon.min_dis2(x)[1] for x in xx])
            # yy = np.array([dragon.min_dis(x) for x in xx])
            plt.figure()
            # yy = [min(y1[i], y2[i]) for i in range(len(xx))]
            plt.plot(xx, y1, color='r')
            plt.plot(xx, y2, color='b')
            # plt.plot(pso.gbest_x, pso.gbest_y, '*r')
            plt.xlabel('时间t(s)', fontproperties=font2, size=12)
            plt.ylabel('A点和B点距其他板凳中心线的最短距离', fontproperties=font2, size=12)
            plt.xticks(fontproperties=font, size=12)
            plt.yticks(fontproperties=font, size=12)
            plt.grid(True, alpha=0.3)
            # plt.savefig("Figure_4_new.png", dpi=300)
            plt.show()
        

if __name__ == '__main__':
    # min_pitch(np.arange(0.30, 0.55, 0.01))
    # min_pitch([0.450337], method="SA")
    min_pitch2([0.450337], fig1=True)
