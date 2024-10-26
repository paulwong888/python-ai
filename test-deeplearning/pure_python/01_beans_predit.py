from init_path import init
init()
import commons.my_dataset as dataset
from matplotlib import pyplot as plt

m = 100
xs, ys = dataset.get_simple_beans(m)

# print(xs)
# print(ys)

plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.scatter(xs, ys)



w = 0.5
for _ in range(100):
    for i in range(100):
        y_pre = w * xs[i]
        e = ys[i] - y_pre
        alpha = 0.05
        w = w + alpha * e * xs[i]
        print("w = " + str(w))

y_pre = w * xs
plt.plot(xs, y_pre)
plt.show()
plt.pause()