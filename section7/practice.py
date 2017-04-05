# codeing: UTF-8
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()

# print(iris.data)
# print(iris.data.shape)

# n = len(iris.data)
# print(n)

# 線形サポートベクターマシン
clf = svm.LinearSVC()
# サポートベクターマシンによる訓練(データ、正解値)
print(iris.data)
print(iris.target)
clf.fit(iris.data, iris.target)

print(clf.predict([[5.1, 3.5, 1.4, 0.1]]))
