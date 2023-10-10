---
title: "pytorch基本使用和入门案例"
date: 2023-10-10
draft: false

tags: ["python"]
categories: ["deep"]
---

## TensorBoard

TensorBoard 是 TensorFlow 的可视化工具，用于帮助开发者直观地了解他们的深度学习模型。它提供了各种可视化界面，能够展示模型的结构、训练过程中的损失曲线、准确率等信息，还可以展示模型的参数分布、梯度信息等。TensorBoard 使得开发者可以更好地理解、调试和优化他们的深度学习模型。

>tensorboard --logdir="指定的文件名"

以下是 TensorBoard 的主要用途：

##### 1. **模型结构可视化：**

TensorBoard 可以绘制神经网络的结构图，包括各个层的连接关系，帮助你直观地了解模型的构建情况。

##### 2. **训练过程可视化：**

可以实时地监测模型在训练集和验证集上的性能，包括损失函数的变化、准确率等，帮助你判断模型的训练效果。

##### 3. **Embedding Projector：**

TensorBoard 提供了嵌入式向量（Embeddings）的可视化，你可以将高维嵌入空间的数据映射到三维空间中，帮助你更好地理解和分析数据。

##### 4. **计算图可视化：**

可以将 TensorFlow 的计算图导出并在 TensorBoard 中查看，这有助于理解 TensorFlow 模型的底层运行机制。

##### 5. **超参数优化：**

TensorBoard 可以帮助你比较不同超参数配置下模型的性能，从而指导你选择最优的超参数。

##### 6. **可视化工具：**

除了上述功能，TensorBoard 还提供了其他一些可视化工具，例如可视化训练数据、查看图像、文本、音频等。

总之，TensorBoard 是一个非常强大的工具，可以帮助深度学习开发者更好地了解他们的模型，提高模型的性能和效果。

## Transform

"Transform"（变换）是数据预处理中的一个重要步骤，特别在计算机视觉和自然语言处理等机器学习任务中。数据变换指的是将原始数据进行某种操作，使其更适合用于模型的训练。数据变换可以包括多种处理，取决于所处理的数据类型和任务需求。

##### 在计算机视觉中，数据变换可以包括：

1. **图像大小调整：** 将图像的大小调整为模型所需的输入大小。这是因为神经网络的输入层通常要求输入数据具有相同的尺寸。

2. **图像旋转和翻转：** 对图像进行随机或指定角度的旋转，或者在水平或垂直方向上进行翻转，增加数据的多样性。

3. **归一化：** 将图像的像素值缩放到一个固定的范围，比如[0, 1]或[-1, 1]，有助于提高模型的稳定性和训练效果。

4. **数据增强：** 对图像进行随机裁剪、旋转、缩放等操作，增加训练数据的多样性，提高模型的泛化能力。

##### 在自然语言处理中，数据变换可以包括：

1. **分词：** 将文本分割成词或子词的序列，方便模型处理。

2. **词嵌入：** 将词汇转换为密集向量，有助于提高模型对词汇语义的理解。

3. **填充和截断：** 将文本序列填充到相同的长度，或者截断超出指定长度的部分。

4. **字符级别的编码：** 将文本转换为字符级别的编码，适用于一些字符级别的任务。

5. **标签编码：** 将文本标签转换为模型可处理的编码，通常用于分类任务。

总的来说，数据变换的目的是提取数据中的有用特征，减少噪声，增加数据的多样性，以便更好地用于机器学习模型的训练。选择合适的数据变换方法可以大幅提高模型的性能和泛化能力。

## data操作

### datasets

```python
import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
text_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
```

### DataLoader

```python
import torchvision
from torch.utils.data import DataLoader

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform)

test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]
```

## TORCH.NN

`torch.nn` 是PyTorch中用于构建神经网络模型的模块。以下是一些常用的 `torch.nn` 中的函数和类的介绍：

### 1. **`nn.Module`**

`nn.Module` 是所有神经网络模块的基类。自定义的神经网络模型需要继承自`nn.Module`，并实现其`forward`方法，该方法定义了模型的前向传播逻辑。

### 2. **常用的层：**

- **线性层：** `nn.Linear(in_features, out_features)`，用于定义全连接层。
- **卷积层：** `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`，用于定义二维卷积层。
- **循环神经网络层：** `nn.RNN(input_size, hidden_size, num_layers)`，用于定义简单的循环神经网络层。

### 3. **激活函数：**

- **ReLU：** `nn.ReLU()`，修正线性单元激活函数。
- **Sigmoid：** `nn.Sigmoid()`，用于二分类问题的激活函数。
- **Softmax：** `nn.Softmax(dim)`，用于多分类问题的激活函数，dim参数指定计算softmax的维度。

### 4. **损失函数：**

- **交叉熵损失：** `nn.CrossEntropyLoss()`，用于多分类问题的损失函数。
- **均方误差：** `nn.MSELoss()`，用于回归问题的损失函数。

### 5. **优化器：**

- **随机梯度下降（SGD）：** `torch.optim.SGD(model.parameters(), lr=learning_rate)`，用于使用随机梯度下降算法进行优化。
- **Adam优化器：** `torch.optim.Adam(model.parameters(), lr=learning_rate)`，一种自适应学习率的优化算法。

### 6. **模型初始化：**

- **Xavier/Glorot初始化：** `torch.nn.init.xavier_uniform_(tensor)`，一种常用的权重初始化方法。

### 7. **模型容器：**

- **顺序容器：** `nn.Sequential(*args)`，按顺序包装多个模块，模块会按照它们被传入的顺序被执行。
- **模块列表：** `nn.ModuleList([module1, module2, ...])`，用于包装多个模块，可以像列表一样进行操作。
- **字典容器：** `nn.ModuleDict({'key1': module1, 'key2': module2, ...})`，用于包装多个模块，可以像字典一样进行操作。

## 两个简单的例子（看懂入门一半）

##### 1.假设我们有一个简单的线性回归任务，我们想要使用神经网络来拟合一个具有噪声的线性关系。我们有一组输入特征 \(x\) 和相应的目标值 \(y\)，我们希望训练一个神经网络来预测 \(y\)。：

首先，我们可以使用PyTorch创建一个带有一个线性层的简单神经网络，然后使用均方误差损失函数（Mean Squared Error Loss）来进行训练。这个线性层将输入特征映射到一个输出值，然后通过梯度下降等优化算法来学习适当的权重和偏置，以最小化预测值与真实值之间的均方误差。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 准备数据（假设有100个样本）
torch.manual_seed(0)
x = torch.rand(100, 1)  # 输入特征，假设是一个列向量
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)  # 目标值，符合 y = 2x + 1 的线性关系，加入了噪声

# 定义线性回归模型，包含一个线性层
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入特征维度为1，输出特征维度为1

    def forward(self, x):
        return self.linear(x)

# 初始化模型、损失函数和优化器
model = LinearRegressionModel()
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    predictions = model(x)
    loss = criterion(predictions, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 打印损失值
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型训练完成后，可以使用模型进行预测
input_value = torch.tensor([[0.8]])  # 输入一个特征值 0.8
predicted_output = model(input_value)
print(f'Predicted output: {predicted_output.item()}')
```

在这个例子中，我们定义了一个包含一个线性层的神经网络模型。该线性层将输入特征维度从1映射到1，即 \(y = wx + b\) 中的 \(w\) 和 \(b\)。我们使用随机生成的数据，其中 \(y = 2x + 1\) 是真实的线性关系，加入了一些噪声。模型使用均方误差损失函数来衡量预测值与真实值之间的差异，并通过梯度下降优化器进行训练。最后，我们使用训练好的模型对新的输入特征值（例如0.8）进行预测。

##### 2.训练CIFAR10模型入门代码：

```python
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("当前训练集的长度为：{}".format(train_data_size))
print("当前测试集的长度为：{}".format(test_data_size))

# DataLoader加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 搭建网络
class FirstModule(nn.Module):
    def __init__(self):
        super(FirstModule, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 示例化model
module = FirstModule()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 判断是否可以使用GPU
if torch.cuda.is_available():
    module = module.cuda()
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)

# 设置训练参数
total_train_step = 0
total_test_step = 0
epoch = 10

for i in range(epoch):
    # 训练代码
    print("----------------------第{}轮开始-------------------".format(i + 1))
    for data in train_dataloader:
        images, targets = data
        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()
        outputs = module(images)
        # 计算损失值
        loss = loss_fn(outputs, targets)

        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数{}，loss:{}".format(total_train_step, loss.item()))
    # 测试代码
    total_test_loss = 0
    total_accuracy = 0  # 测试整体正确的个数
    with torch.no_grad():
        for data in test_dataloader:
            images, targets = data
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()
            outputs = module(images)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    torch.save(module, "module_{}.pth".format(i + 1))
    print("模型已经保存！")

```

