import matplotlib.pyplot as plt
import numpy as np
import math

# ネイピア数
# e = math.e
# dx = 0.1
# x = np.arange(-8, 8, dx)
# y = 1/ (1+e~(-x))
# y_sig = 1 / (1 + e ** -x)

def sigmoid(a):
    # ネイピア数
    e = math.e
    s = 1 / (1 + e ** -a)
    return s

dx = 0.1
x = np.arange(-8, 8, dx)
y_sig = sigmoid(x)

plt.plot(x, y_sig)
plt.show()
