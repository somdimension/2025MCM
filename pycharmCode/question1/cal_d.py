# @Version : 1.0
# @Author  : 亥子曜
# @File    : cal_d.py
# @Time    : 2025/9/5 0:02
import math


def lambd_in_SquareN(lambd):
    t1 = math.pow(lambd,2)*5.58245
    t2 = 2.468516 * math.pow(lambd,2)
    b1 = math.pow(lambd,2) - math.pow(0.1625394,2)
    b2 = math.pow(lambd,2) - math.pow(11.35656,2)
    SquareN = 1 + t1/b1 + 10*t2/b2
    return SquareN


def cal_D(T,lambd,inangle):# T是周期  lambd就是波长？还是波长数？  inangle是入射角
    t = math.pow(lambd,2)
    b = T*2*math.sqrt(lambd_in_SquareN(lambd))*math.sqrt(1-math.pow(math.sin(inangle),2)/lambd_in_SquareN(lambd))