# codeing: UTF-8
import math


# シグモイド関数
def sigmoid(a):
    # math.exp(-a)はネイピア数eの-a乗を表す
    return 1.0 / (1.0 + math.exp(-a))


# ニューロン
class Neuron:
    input_sum = 0.0
    output = 0.0

    def setInput(self, inp):
        self.input_sum += inp

    def getOutput(self):
        self.output = sigmoid(self.input_sum)
        return self.output


# ニューラルネットワーク
class NeuralNetwork:
    # ニューロンのインスタンス
    neuron = Neuron()

    # 実行
    def commit(self, input_data):
        for data in input_data:
            self.neuron.setInput(data)
        return self.neuron.getOutput()


neural_network = NeuralNetwork()

trial_data = [-1.0, -2.0, -3.0]
print(neural_network.commit(trial_data))