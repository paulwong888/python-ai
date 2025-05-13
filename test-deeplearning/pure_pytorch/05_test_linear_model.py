import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# torch.manual_seed(1)
# model1 = nn.Linear(in_features=1, out_features=1)
# print(model1.bias, model1.weight)

# x = torch.tensor([[2.0], [3.3]])
# print(model1(x))

############################

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        """
        这段代码是使用PyTorch框架写的，用于设置随机种子和定义一个线性模型。下面是代码的逐步解释：

        torch.manual_seed(1)：这行代码设置了PyTorch的随机种子为1。
        设置随机种子是为了确保代码的可重复性，使得每次运行代码时，随机数生成器产生的随机数序列都是相同的。
        这对于调试和复现实验结果非常有用。

        model = Linear(in_features=1, out_features=1)：
        这行代码定义了一个线性模型（Linear）。
        Linear是PyTorch中的一种神经网络层，用于实现线性变换，
        即y = xA^T + b，其中x是输入，A是权重矩阵，b是偏置项。
        在这个例子中，in_features=1表示输入特征的数量为1，out_features=1表示输出特征的数量为1。
        """
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)
    
    def forward(self, x):
        pred = self.linear(x)
        return pred
    
    def get_params(self):
        [w, b] = self.linear.parameters()
        w1 = w[0][0].item()
        b1 = b[0].item()
        print(w1, b1)
        return (w1, b1)
    
    def plot_fit(self, title):
        plt.title = title
        w1, b1 = self.get_params()
        x1 = np.array([-30, 30])
        print(f"np.array([-30, 30]) --> {x1}")
        y1 = w1*x1 + b1
        print(f"w1*x1 + b1 --> {y1}")
        plt.plot(x1, y1, "r")
        plt.scatter(X, y)
        plt.show()
    
# model = LR(1, 1)
# x = torch.tensor([2.0])
# print(model.forward(x))

############################

#产生一个二维数组，即n个一维数组，有两个规格，一维数组的个数和一维数组的里元素的个数
"""
这行代码使用PyTorch创建了一个形状为(100, 1)的张量x，
其中的元素是从标准正态分布（均值为0，标准差为1）中随机采样得到的。

这里是代码的详细解释：
torch.randn：这是PyTorch中的一个函数，用于生成一个形状为(*size)的张量，其中包含从标准正态分布中随机采样的值。
100, 1：这是torch.randn函数的参数，指定了生成张量的形状。

100表示生成的张量有100行，1表示每行有1列，因此整个张量是一个100行1列的列向量。
这个张量x可以用于多种用途，比如作为机器学习模型的输入数据。

由于元素是随机生成的，所以每次执行这行代码时，x中的值都会不同，
除非你设置了随机种子（如之前提到的torch.manual_seed(1)），
这会使得随机数生成器的序列固定，从而使得每次运行代码时生成的随机数相同。

y = X + 3 * torch.randn(100, 1)：
这行代码生成另一个形状为(100, 1)的张量y。y的值是X的值加上3倍的标准正态分布随机值。
这可以看作是在X的基础上添加了一些噪声。

print(X)：这行代码打印出X张量的内容。
plt.plot(X, y, "o")：这行代码使用Matplotlib库的plot函数来绘制X和y之间的散点图。
"o"参数指定了散点图的标记样式为圆圈。
plt.show()：这行代码显示刚才绘制的散点图。
"""
X = torch.randn(100, 1)*10 
y = X + 3*torch.randn(100, 1)
print(X)
plt.plot(X, y, "o")
plt.show()

model = LR(1, 1)
print(model)

model.plot_fit("Initial Model")

############################

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    print("epoch: ", i, "loss:", loss.item())

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.show()

model.plot_fit("Trainer Model")