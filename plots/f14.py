"""
图14 -- 最小螺距下龙头进入调头空间的状态
"""
from src.dragon import *

dragon = Dragon(d=0.450337)
dragon.set_time(dragon.arr_time())
dragon.print_img(magn=6, circle=True)