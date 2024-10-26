from init_path import init
init()
import commons.my_dataset as dataset
import numpy as np
from matplotlib import pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense

def get_data_set(m : int, n:int):
    w = 5
    b = 7
    x = np.random.rand(m, n)
    y = w * x + b
    return x, y

# 假设我们有5000条数据，每条数据只有一个特征
num_samples = 5000
num_features = 1

# 生成随机数据
Xs = np.random.rand(num_samples, num_features)
# 假设的标签，这里也是随机生成的
ys = np.random.rand(num_samples, num_features)

m = 5000
xs, ys = get_data_set(m) 



print(xs)
print(ys)

plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.scatter(xs, ys)
plt.show()
# plt.pause()

# 创建Sequential模型
model = Sequential()

# 添加一个Dense层，输入维度为1（因为我们只有一个特征），输出维度为1
# 激活函数使用linear，因为我们这里假设是一个回归问题
model.add(Dense(1, activation="linear", input_dim=1))

# 编译模型，使用mean_squared_error作为损失函数，adam优化器
model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])

# 训练模型
model.fit(xs, ys, epochs=100, verbose=1)

print(f"model.summary(): {model.summary()}")
print(f"model.get_weights(): {model.get_weights()}")


# 评估模型
loss = model.evaluate(xs, ys)
print(f"Loss: {loss}")