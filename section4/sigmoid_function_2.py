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


dx = 0.1
x = np.arange(-8, 8, dx)

# シグモイド関数
y_sig = sigmoid(x)
# シグモイド関数の傾き
y_dsid = (sigmoid(x + dx) - sigmoid(x)) / dx

plt.plot(x, y_sig, label = "sigmoid")
plt.plot(x, y_dsid, label = "d_sigmoid")
plt.legend()
plt.show()
