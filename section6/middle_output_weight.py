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
    # 入力層と中間層の重み
    w_im = [[0.496, 0.512], [-0.501, 0.998], [0.498, -0.502]]
    # 中間層と出力層の重み
    w_mo = [0.121, -0.4996, 0.200]

    # 各層の宣言
    input_layer = [0.0, 0.0, 1.0]
    middle_layer = [Neuron(), Neuron(), 1.0]
    output_layer = Neuron()

    # 実行
    def commit(self, input_data):
        # 各層のリセット
        self.input_layer[0] = input_data[0]
        self.input_layer[1] = input_data[1]

        self.middle_layer[0].reset()
        self.middle_layer[1].reset()

        self.output_layer.reset()

        # 入力層ー＞中間層
        self.middle_layer[0].setInput(self.input_layer[0] * self.w_im[0][0])
        self.middle_layer[0].setInput(self.input_layer[1] * self.w_im[1][0])
        self.middle_layer[0].setInput(self.input_layer[2] * self.w_im[2][0])

        self.middle_layer[1].setInput(self.input_layer[0] * self.w_im[0][1])
        self.middle_layer[1].setInput(self.input_layer[1] * self.w_im[1][1])
        self.middle_layer[1].setInput(self.input_layer[2] * self.w_im[2][1])

        # 中間層ー＞出力層
        self.output_layer.setInput(self.middle_layer[0].getOutput() * self.w_mo[0])
        self.output_layer.setInput(self.middle_layer[1].getOutput() * self.w_mo[1])
        self.output_layer.setInput(self.middle_layer[2] * self.w_mo[0])

        return self.output_layer.getOutput()

    def learn(self, input_data):
        print(input_data)

        # 出力値
        output_data = self.commit([input_data[0], input_data[1]])
        # 正解値
        corrrect_value = input_data[2]
        # 学習係数
        k = 0.3

        # 出力値-中間層
        p = (corrrect_value - output_data) * output_data * (1.0 - output_data)
        old_w_mo = list(self.w_mo)
        delta_w_mo = [
            p * self.middle_layer[0].output,
            p * self.middle_layer[1].output,
            p * self.middle_layer[2]
        ]
        self.w_mo[0] += delta_w_mo[0] * k
        self.w_mo[1] += delta_w_mo[1] * k
        self.w_mo[2] += delta_w_mo[2] * k

        # 中間層 - 入力層
        delta_w_im = [
            delta_w_mo[0] *  old_w_mo[0] * self.middle_layer[0].output * (1.0 - self.middle_layer[0].output),
            delta_w_mo[1] * old_w_mo[1] * self.middle_layer[1].output * (1.0 - self.middle_layer[1].output),
        ]
        self.w_im[0][0] += self.input_layer[0] * delta_w_im[0] * k
        self.w_im[0][0] += self.input_layer[0] * delta_w_im[0] * k
        self.w_im[0][0] += self.input_layer[0] * delta_w_im[0] * k
        self.w_im[0][0] += self.input_layer[0] * delta_w_im[0] * k
        self.w_im[0][0] += self.input_layer[0] * delta_w_im[0] * k
        self.w_im[0][0] += self.input_layer[0] * delta_w_im[0] * k


# 基準点（データの範囲を0.0 - 1.0の範囲に納めるため）
refer_point_0 = 34.5
refer_point_1 = 137.5
# ファイルの読み込み
training_data = []
training_data_file = open("training_data", "r")
for line in training_data_file:
    line = line.rstrip().split(",")
    training_data.append([float(line[0]) - refer_point_0, float(line[1]) - refer_point_1, int(line[2])])
training_data_file.close()
# ニュラルネットワークのインスタンス
neural_network = NeuralNetwork()
# 学習
neural_network.learn(training_data[0])
# 訓練用データの表示の準備
position_tokyo_learing = [[], []]
position_kanagawa_leaning = [[], []]
for data in training_data:
    if data[2] < 0.5:
        position_tokyo_learing[0].append(data[1] + refer_point_1)
        position_tokyo_learing[1].append(data[0] + refer_point_0)
    else:
        position_kanagawa_leaning[0].append(data[1] + refer_point_1)
        position_kanagawa_leaning[1].append(data[0] + refer_point_0)
# プロット
plt.scatter(position_tokyo_learing[0], position_tokyo_learing[1], c="red", label="tokyo", marker="+")
plt.scatter(position_kanagawa_leaning[0], position_kanagawa_leaning[1], c="blue", label="kanagawa", marker="+")
plt.legend()
plt.show()
