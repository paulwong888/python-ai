"""
2、卷积神经网络（CNN）

一、模型原理
卷积神经网络主要由卷积层、池化层和全连接层组成。
卷积层通过卷积核与输入数据进行局部连接和权值共享，提取输入的局部特征。
卷积核在输入数据上滑动，对每个局部区域进行卷积操作，得到特征图。
池化层通常在卷积层之后，用于降低特征图的空间尺寸，减少参数数量和计算量，同时保留主要特征。
常见的池化方法有最大池化和平均池化。
全连接层在网络的最后，将提取到的特征进行整合，输出最终的分类或回归结果。

二、模型训练
准备数据集：收集并整理适合任务的数据集，进行数据预处理，如归一化、数据增强等。
定义网络结构：确定卷积神经网络的层数、卷积核大小、池化方式等参数。
选择损失函数和优化算法：常见的损失函数有交叉熵损失、均方误差等，优化算法如随机梯度下降、Adam 等。
前向传播：将输入数据送入网络，经过卷积、池化和全连接层的计算，得到输出结果。
计算损失：根据输出结果和真实标签计算损失值。
反向传播：根据损失值，通过反向传播算法计算各层参数的梯度。
更新参数：使用优化算法根据梯度更新网络参数。
重复以上步骤，直到达到预设的训练次数或损失收敛。

三、优点
局部连接和权值共享：减少了参数数量，提高了模型的泛化能力。
自动特征提取：能够自动学习到图像、文本等数据的特征，无需手动设计特征。
对平移、旋转和缩放具有一定的不变性：适合处理图像等具有空间结构的数据。
并行计算：卷积和池化操作可以并行计算，提高训练和推理速度。

四、缺点
对数据量要求较高：需要大量的标注数据才能训练出较好的模型。
计算量大：尤其是在训练深度较大的网络时，需要大量的计算资源和时间。
黑箱性：难以解释模型的决策过程。

五、使用场景
图像识别：如物体识别、人脸识别、图像分类等。
视频分析：动作识别、视频分类等。
自然语言处理：文本分类、情感分析等（结合一维卷积）。

"""


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# 构建模型
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)