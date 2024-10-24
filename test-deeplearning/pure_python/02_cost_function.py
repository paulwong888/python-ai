from init_path import init
init()
import commons.my_dataset as dataset
from matplotlib import pyplot as plt
import numpy as np

m = 100
xs, ys = dataset.get_simple_beans(m)

# print(xs)
# print(ys)

plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.scatter(xs, ys)



w = 0.5
y_pre = w * xs

plt.plot(xs, y_pre)
plt.show()

es = (ys - y_pre) ** 2
sum_e = np.sum(es)
sum_e = (1/m)*sum_e

print(sum_e)

ws = np.arange(0, 3, 0.1)

es = []
for w in ws:
    y_pre = w * xs
    e = (1/m)*np.sum((ys - y_pre) ** 2)
    es.append(e)

plt.title("Cost Function", fontsize=12)
plt.xlabel("w")
plt.ylabel("e")
plt.plot(ws, es)
plt.show()

w_min = np.sum(xs * ys) / np.sum(xs * xs)
print("e最小点：" + str(w_min))

y_pre = w_min * xs

plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.plot(xs, y_pre)
plt.scatter(xs, ys)
plt.show()