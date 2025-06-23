"""
变步长求解碰撞时刻
"""
from dragon import *

dragon = Dragon()


def check_col(start, end, step):
    for time in np.arange(start, end, step):
        dragon.set_time(time)
        coll = dragon.judge_col()
        print("time = ", time, " if_collision = ", coll[0], " min_dis = ", coll[1])


# 变步长搜索
# check_col(300, 350, 1)
# check_col(350, 400, 0.5)
# check_col(400, 410, 0.1)
# check_col(408, 410, 0.01)
# check_col(412, 413, 0.05)
check_col(415, 415.5, 0.01)

# 二分搜索确定碰撞时刻
def distFn(time):
    dragon.set_time(time)
    return dragon.judge_col()[1] - 0.15

# print(binary_search(distFn, 412, 413, incre=False))

