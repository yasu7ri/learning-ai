# coding: UTF-8

import matplotlib.pyplot as plt
import numpy as np
import math


# シグモイド関数
def sigmoid(a):
    # ネイピア数
    e = math.e
    # y = 1 / (1+ e~(-x))
    s = 1 / (1 + e ** -a)
    return s


def d_sigmoid(a):
    d = sigmoid(a) * (1 - sigmoid(a))
    return d


dx = 0.1
x = np.arange(-8, 8, dx)

# シグモイド関数
y_sig = sigmoid(x)
# シグモイド関数の傾き：微分
y_dsid = (sigmoid(x + dx) - sigmoid(x)) / dx
# シグモイド関数の微分
# dy_dig = sigmoid(x) * (1 - sigmoid(x))
dy_dig = d_sigmoid(x)

plt.plot(x, y_sig, label="sigmoid")
plt.plot(x, y_dsid, label="d_sigmoid")
plt.plot(x, dy_dig, label="d_sigmoid")
plt.legend()
plt.show()
