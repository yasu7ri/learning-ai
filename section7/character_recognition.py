# codeing: UTF-8
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

digits = datasets.load_digits()

# print(digits.data)
# print(digits.data.shape)

n = len(digits.data)

# images = digits.images
# labels = digits.target
# 画像と正解値の表示
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
#     plt.axis("off")
#     plt.title("Training: " + str(labels[i]))
# plt.show()

clf = svm.SVC(gamma=0.001, C=100.0)
clf.fit(digits.data[:n * 6 / 10], digits.target[:n * 6 / 10])

# 最後の10個のデータをチェック
# print(digits.target[-10:])
# print(clf.predict(digits.data[-10:]))

# 正解
expected = digits.target[-n * 4 / 10:]
# 予測
predicted = clf.predict(digits.data[-n * 4 / 10:])
# 正解率
print(metrics.classification_report(expected, predicted))
# 誤認識のマトリックス
print(metrics.confusion_matrix(expected, predicted))

images = digits.images[-n * 4 / 10:]
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.axis("off")
    plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Guess: " + str(predicted[i]))
plt.show()
