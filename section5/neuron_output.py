# codeing: UTF-8

# ニューロン
class Neuron:
    # 入力値の合計を格納する変数
    input_sum = 0.0
    # 出力
    output = 0.0

    def setInput(self, inp):
        self.input_sum += inp
    # print(self.input_sum)

    def getOutput(self):
        self.output = self.input_sum
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


# ニュラルネットワークのインスタンス
neural_network = NeuralNetwork()

# 実行
trial_data = [1.0, 2.0, 3.0]
print(neural_network.commit(trial_data))
