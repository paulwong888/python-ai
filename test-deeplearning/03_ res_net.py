"""
3、残差网络（ResNet）
随着深度学习的迅猛发展，深度神经网络在诸多领域都取得了极为显著的成就。
然而，深度神经网络在训练过程中面临着诸如梯度消失以及模型退化等问题，这在很大程度上限制了网络的深度与性能。
为有效解决这些难题，残差网络（ResNet）应运而生。

一、模型原理
传统的深度神经网络在层数不断增加时会出现梯度消失或梯度爆炸问题，以及性能退化现象。
ResNet 通过引入残差块（residual block）来解决这些问题。
残差块由两部分组成：直接映射和残差映射。
直接映射就是输入直接连接到后面的层，残差映射则是通过一些卷积层等操作对输入进行处理得到的结果。
最终的输出是直接映射和残差映射的和。这种结构使得网络可以更容易地学习恒等映射，即使网络层数很深也能有效地训练。

二、模型训练
数据准备：同其他网络一样，准备合适的训练数据集并进行预处理。
定义网络结构：构建由多个残差块组成的 ResNet 网络，确定层数、卷积核大小等参数。
选择损失函数和优化算法：例如交叉熵损失和随机梯度下降优化器等。
前向传播：输入数据经过 ResNet 网络的各个层进行计算。
计算损失：根据输出结果与真实标签计算损失值。
反向传播：利用反向传播算法计算各层参数的梯度。
更新参数：使用优化算法更新网络参数。
重复训练过程直到满足停止条件。


三、优点
可以训练非常深的网络：解决了深度网络的性能退化问题，使得能够构建上百层甚至上千层的网络。
高效的训练：残差结构使得梯度更容易在网络中传播，加快了训练速度。
良好的泛化能力：在各种图像分类等任务中表现出优秀的性能。


四、缺点
计算复杂度较高：由于网络较深，计算量相对较大，需要较多的计算资源和时间。
内存占用较大：在训练和推理过程中可能需要较大的内存空间。


五、使用场景
图像分类：在大规模图像分类任务中取得了显著的成果。
目标检测：作为骨干网络用于目标检测算法中。
图像分割：可用于图像分割任务，提取有效的特征。

"""

from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from keras.models import Model

def residual_block(input_tensor, filters, stride=1):
    x = Conv2D(filters, (3, 3), strides=stride, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = input_tensor
    if stride!= 1 or input_tensor.shape[-1]!= filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    return Activation('relu')(Add()([x, shortcut]))

input_tensor = Input(shape=(28, 28, 1))
x = Conv2D(64, (7, 7), strides=2, padding='same')(input_tensor)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

x = residual_block(x, 64)
x = residual_block(x, 64)
x = residual_block(x, 128, stride=2)
x = residual_block(x, 128)
x = residual_block(x, 256, stride=2)
x = residual_block(x, 256)
x = residual_block(x, 512, stride=2)
x = residual_block(x, 512)

x = AveragePooling2D((7, 7))(x)
x = Flatten()(x)
output = Dense(10, activation='softmax')(x)

model = Model(input_tensor, output)

# 编译模型等后续步骤与其他网络类似