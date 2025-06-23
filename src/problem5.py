"""
求解全局把手最大速度
"""
from dragon2 import *

dragon = Dragon()


# 求某时刻的最大速度
def max_vt(time):
    dragon.set_time(time, need_vol=True)
    return max(dragon.vol)
    

# 求某时间段内的最大速度
def max_v(start, end, step, fig=False, save_pth=None):

    glo_max_v = 0 # 全局最大速度
    max_time = 0 # 最大速度对应时间

    for time in np.arange(start, end, step):
        dragon.set_time(time, need_vol=True)
        cur_max_v = max(dragon.vol)

        if cur_max_v > glo_max_v:
            glo_max_v = cur_max_v
            max_time = time

        print("time = ", time, " max_v = ", cur_max_v)

    print(glo_max_v, max_time)

    if fig:
        plt.figure(figsize=(20, 7))

        xx = np.arange(start, end, step)
        yy = [max_vt(x) for x in xx]
        # plt.plot(xx, yy, color='#00C5D7', linewidth=2)
        plt.plot(xx, yy, linewidth=2)

        plt.xlabel('时间t(s)', fontproperties=font2, size=12)
        plt.ylabel('把手最大速度(m/s)', fontproperties=font2, size=12)
        plt.xticks(fontproperties=font, size=12)
        plt.yticks(fontproperties=font, size=12)

        plt.grid(True, alpha=0.3)
        if save_pth:
            plt.savefig(save_pth, dpi=300)
        plt.show()


# 三分搜索算法(求极大值)
def ternary_search(fn, lb, ub, eps=1e-8):

    cnt = 0

    while ub - lb >= eps:

        cnt += 1

        margin = (ub - lb) / 3

        x1 = lb + margin
        x2 = ub - margin

        if fn(x1) <= fn(x2):
            lb = x1
        else:
            ub = x2

    print(cnt)

    return  (lb + ub) / 2


# 三分搜索法求最大速度
def main(save_pth=None, show=False):

    t = ternary_search(max_vt, 14, 15)
    f_t = max_vt(t)

    print(t, f_t)

    if show:
        plt.figure()

        xx = np.arange(14, 15, 0.01)
        yy = [max_vt(x) for x in xx]
        # plt.plot(xx, yy, color='#00A7E3', linewidth=2)
        plt.plot(xx, yy, linewidth=2)

        plt.plot(t, f_t, marker='*', color='red')

        plt.xlabel('时间t(s)', fontproperties=font2, size=12)
        plt.ylabel('把手最大速度(m/s)', fontproperties=font2, size=12)
        plt.xticks(fontproperties=font, size=12)
        plt.yticks(fontproperties=font, size=12)

        plt.grid(True, alpha=0.3)
        if save_pth is not None:
            plt.savefig(save_pth, dpi=300)
        plt.show()


if __name__ == '__main__':
    # max_v(13, 16, 0.05)
    # max_v(376, 384, 0.05)
    main()