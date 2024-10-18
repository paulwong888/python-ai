"""
1、深度神经网络（DNN）
深度神经网络（DNN），又称为多层感知机，是最为普遍的深度学习算法。在发明之初，由于算力瓶颈的限制，它曾饱受质疑。
然而，直到近些年，随着算力的飞速提升以及数据的爆发式增长，深度神经网络才迎来了重大突破。
如今，它在图像识别、自然语言处理等众多领域都展现出了强大的实力。

一、模型原理
深度神经网络是一种具有多个隐藏层的人工神经网络。它通过多个神经元组成的层来对输入数据进行逐步抽象和特征提取。
每一层的神经元接收上一层的输出作为输入，并通过激活函数进行非线性变换后输出给下一层。
通过不断调整神经元之间的连接权重，使得网络能够学习到输入数据与输出目标之间的复杂映射关系。

二、模型训练
通常使用反向传播算法进行训练。
首先，给定一组输入数据和对应的目标输出，网络进行前向传播计算出实际输出。
然后，根据实际输出与目标输出之间的误差，通过反向传播算法计算出各层连接权重的梯度。
接着，使用优化算法（如随机梯度下降）根据梯度更新权重，以减小误差。
重复这个过程，直到误差达到一个可接受的水平或者达到预设的训练次数。

三、优点
强大的学习能力：能够学习到非常复杂的函数关系，对大规模数据具有很好的拟合能力。
自动特征提取：可以自动从原始数据中提取有用的特征，减少了人工特征工程的工作量。
通用性：适用于多种任务，如图像识别、语音识别、自然语言处理等。

四、缺点
计算量大：由于具有大量的参数和复杂的结构，训练和推理过程需要大量的计算资源和时间。
过拟合风险：容易对训练数据过度拟合，导致在新数据上的泛化能力不足。
黑箱性：难以解释网络的决策过程，对于一些对可解释性要求高的任务可能不适用。

五、使用场景
图像识别：可以识别图像中的物体、场景等。
语音识别：将语音信号转换为文本。
自然语言处理：如文本分类、机器翻译等任务。
推荐系统：根据用户的行为和偏好进行个性化推荐。

"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 构建模型
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)