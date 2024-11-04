import torch

v = torch.tensor([1, 2, 3, 4, 5, 6])
print(v[1:])

print(v.view(3, 2))
print(torch.zeros(3, 4))
print(torch.ones(3, 4))
print(torch.empty(3, 2))
print(torch.arange(0, 10, 2))
print(torch.linspace(1, 10, steps=5))
print(torch.randn(2, 3))
print(torch.arange(18).view(3, 2, 3))

print(torch.tensor(2.0, requires_grad=True))

"""
这段代码是使用PyTorch框架写的，用于计算梯度。下面是代码的逐步解释：
1. `x = torch.tensor(1.0, requires_grad=True)`：
    创建一个PyTorch张量`x`，其值为1.0，并设置`requires_grad=True`，表示需要计算关于`x`的梯度。
2. `z = torch.tensor(2.0, requires_grad=True)`：
    创建另一个PyTorch张量`z`，其值为2.0，并设置`requires_grad=True`，表示需要计算关于`z`的梯度。
3. `y = x**2 + z**3`：计算`y`的值，它是`x`的平方加上`z`的立方。
4. `y.backward()`：对`y`进行反向传播，计算关于`x`和`z`的梯度。由于`y`是`x`和`z`的函数，这个操作会计算`y`关于`x`和`z`的导数。
5. `print(x.grad)`：打印`x`的梯度。

在执行`y.backward()`之后，`x.grad`和`z.grad`将分别存储`y`关于`x`和`z`的梯度。对于这个特定的例子：
- `y`关于`x`的梯度是`2x`，因为`y = x**2 + z**3`，所以`dy/dx = 2x`。
- `y`关于`z`的梯度是`3z**2`，因为`y = x**2 + z**3`，所以`dy/dz = 3z**2`。

由于`x`的值是1.0，`z`的值是2.0，所以：
- `x.grad`将会是`2 * 1.0 = 2.0`。
- `z.grad`将会是`3 * 2.0**2 = 3 * 4 = 12.0`。

因此，执行这段代码后，`print(x.grad)`将会输出`2.0`。
"""
x = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(2.0, requires_grad=True)
y = x**2 + z**3
y.backward()
print(x.grad)