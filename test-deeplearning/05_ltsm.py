"""
4、LSTM（长短时记忆网络）

在处理序列数据时，传统的循环神经网络（RNN）面临着梯度消失和模型退化等问题，这限制了网络的深度和性能。为了解决这些问题，LSTM被提出。



图片

一、模型原理

LSTM 是一种特殊的循环神经网络（RNN）结构，主要用于处理序列数据。它通过引入门控机制来控制信息的流动和遗忘，从而有效地解决了传统 RNN 中存在的长期依赖问题。

LSTM 单元主要由三个门和一个记忆单元组成：

输入门（Input Gate）：决定当前输入信息有多少被存储到记忆单元中。
遗忘门（Forget Gate）：控制记忆单元中信息的遗忘程度。
输出门（Output Gate）：决定记忆单元中的信息有多少被输出到当前隐藏状态。
记忆单元（Cell State）：用于存储长期信息。


二、模型训练

数据准备：收集和整理序列数据，进行适当的预处理，如归一化等。
定义网络结构：确定 LSTM 的层数、隐藏单元数量等参数。
选择损失函数和优化算法：常见的有交叉熵损失和 Adam 优化器等。
前向传播：将序列数据依次输入 LSTM 网络，计算每个时间步的输出。
计算损失：根据输出结果与真实标签计算损失值。
反向传播：通过反向传播算法计算各参数的梯度。
更新参数：使用优化算法更新网络参数。
重复训练过程直到满足停止条件。


三、优点

能够处理长期依赖：有效地记忆和利用长序列中的信息。
门控机制灵活：可以根据不同的任务和数据自适应地控制信息的流动。
广泛应用：在自然语言处理、时间序列预测等领域表现出色。


四、缺点

计算复杂度较高：由于门控机制和复杂的结构，计算量相对较大。
训练时间较长：尤其是对于大规模数据集和深层网络。


五、使用场景

自然语言处理：如语言建模、机器翻译、文本分类等。
时间序列预测：股票价格预测、气象预测等。
音频处理：语音识别、音乐生成等。
"""

from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Dense

# 假设已经有了预处理后的文本数据和对应的标签
max_features = 20000
maxlen = 100

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32)

# 评估模型
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])