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

# for _ in range(100):
for i in range(100):
    x = xs[i]
    y = ys[i]
    k = 2 * (x ** 2) * w * (-2 * x * y)
    alpha = 0.1
    w = w - alpha * k
    plt.clf()
    plt.scatter(xs, ys)

    y_pre = w * xs

    plt.xlim(0, 1)
    plt.ylim(0, 1.2)
    plt.plot(xs, y_pre)
    plt.pause(0.01)
