# coding: UTF-8
from sklearn import svm

stock_data = []
stock_data_file = open("stock-price", "r")
for line in stock_data_file:
    line = line.rstrip()
    stock_data.append(float(line))
stock_data_file.close()

# print(stock_data)
# print(len(stock_data))
count_s = len(stock_data)

modified_data = []
for i in range(1, count_s):
    modified_data.append(float(stock_data[i] - stock_data[i - 1]) / float(stock_data[i - 1]) * 20)
# print(modified_data)
count_m = len(modified_data)
# print(count_m)

successive_data = []
answers = []
for i in range(4, count_m):
    successive_data.append([modified_data[i - 4], modified_data[i - 3], modified_data[i - 2], modified_data[i - 1]])
    if modified_data[i] > 0:
        answers.append(1)
    else:
        answers.append(0)
# print(successive_data)
# print(answers)

n = len(successive_data)
# print(n)
m = len(answers)
# print(m)


clf = svm.LinearSVC()
clf.fit(successive_data[:n * 75 // 100], answers[:n * 75 // 100])

expected = answers[n * 25 // 100:]
predicted = clf.predict(successive_data[n * 25 // 100:])

print(expected[-10:])
print(list(predicted[-10:]))

correct = 0.0
wrong = 0.0
for i in range(n * 25 // 100):
    if expected[i] == predicted[i]:
        correct += 1
    else:
        wrong += 1
print("正解率: " + str(correct / (correct + wrong) * 100) + "%")
