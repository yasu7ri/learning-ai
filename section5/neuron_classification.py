# codeing: UTF-8
import math
import matplotlib.pyplot as plt


# シグモイド関数
def sigmoid(a):
    return 1.0 / (1.0 + math.exp(-a))


# ニューロン
class Neuron:
    # 入力値の合計を格納する変数
    input_sum = 0.0
    # 出力
    output = 0.0

    def setInput(self, inp):
        self.input_sum += inp

    def getOutput(self):
        # 入力値の合計をシグモンド関数で変換する
        self.output = sigmoid(self.input_sum)
        return self.output

    def reset(self):
        self.input_sum = 0
        self.output = 0


# ニューラルネットワーク
class NeuralNetwork:
    # 入力の重み
    w = [-0.5, 0.5]
    # ニューロンのインスタンス
    neuron = Neuron()

    # 実行
    def commit(self, input_data):
        self.neuron.reset()

        self.neuron.setInput(input_data[0] * self.w[0])
        self.neuron.setInput(input_data[1] * self.w[1])
        return self.neuron.getOutput()


# 基準点（データの範囲を0.0 - 1.0の範囲に納めるため）
refer_point_0 = 34.5
refer_point_1 = 137.5

# ファイルの読み込み
trial_data = []
trial_data_file = open("trial_data", "r")
for line in trial_data_file:
    line = line.rstrip().split(",")
    trial_data.append([float(line[0]) - refer_point_0, float(line[1]) - refer_point_1])
trial_data_file.close()

# ニュラルネットワークのインスタンス
neural_network = NeuralNetwork()

# 実行
position_tokyo = [[], []]
position_kanagawa = [[], []]
for data in trial_data:
    if neural_network.commit(data) < 0.5:
        # 経度
        position_tokyo[0].append(data[1] + refer_point_1)
        # 緯度
        position_tokyo[1].append(data[0] + refer_point_0)
    else:
        position_kanagawa[0].append(data[1] + refer_point_1)
        position_kanagawa[1].append(data[0] + refer_point_0)

# プロット(散布図を表示)
plt.scatter(position_tokyo[0], position_tokyo[1], c="red", label="Tokyo", marker="+")
plt.scatter(position_kanagawa[0], position_kanagawa[1], c="blue", label="Kanagawa", marker="+")
plt.legend()
plt.show()
