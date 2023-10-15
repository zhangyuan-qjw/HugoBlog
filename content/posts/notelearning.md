---
title: "deep练手项目"
date: 2023-10-15
draft: false

tags: ["python"]
categories: ["deep"]
---

## 具体代码

>模型：多层感知器（MLP)
>
>数据集：MNIST
>
>损失函数：CrossEntropyLoss()
>
>优化器：torch.optim.Adam()

```python
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化的操作
])

train_dataset = datasets.MNIST('data/', transform=transform, train=True, download=True)
text_dataset = datasets.MNIST('data/', transform=transform, train=False, download=True)

batch_size = 64
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
text_loader = data.DataLoader(dataset=text_dataset, batch_size=batch_size)


class NoteModel(nn.Module):

    def __init__(self):
        super(NoteModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 一开始需要预估隐藏神经元的数量
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        out = x.view(-1, 28 * 28)  # 将输入的图像展开成一维向量
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)  # # 输出层，不应用激活函数（在交叉熵损失函数中包含了softmax操作）
        return out


model = NoteModel().to(device)

loss_fn = nn.CrossEntropyLoss()

opt = torch.optim.Adam(model.parameters(), lr=0.001)

total_train_step = 0
epoch = 20
for i in range(epoch):
    print('-------------------第{}轮训练开始-------------------'.format(i + 1))
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outs = model(inputs)
        loss = loss_fn(outs, labels)  # 获取一个小批次的平均损失值
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数{}，loss:{}".format(total_train_step, loss.item()))
    print('第{}轮训练结束！loss:{}'.format(i + 1, loss.item()))
    total_text_step = 0
    correct = 0
    print('-------------------第{}轮测试开始----------------'.format(i + 1))
    with torch.no_grad():
        for inputs, labels in text_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_text_step += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = (correct / total_text_step) * 100
        print(f'测试数据成功的概率为: {accuracy:.2f}%')
    torch.save(model, 'noteLearning_{}.pth'.format(i + 1))
    print('第{}训练轮模型已保存！'.format(i + 1))

```

# 用到的一些设计思路

- 判别项目所属的问题。选用合适的模型

- 对数据进行统一格式化，采用小批次训练的方法

- 归一化操作对数据进行更好的清洗和处理

- 选用模型并搭建，根据输入数据初始化输入层固定参数，之后不断尝试隐层调参，输出层按要求返回特征数据

- 对于分类问题一般采用概率值输出特征

- ```python
  loss_fn = nn.CrossEntropyLoss() # 该损失函数具体作用？该函数在python里面集合了sigmoid函数
  ```

## 问题

PyTorch 中 torch.optim.Adam() 和 torch.optim.SGD() 都是优化神经网络的优化器,主要有以下区别:

1. Adam属于自适应学习率优化算法,会根据训练迭代过程中参数的梯度变化,动态调整每个参数的学习率。而SGD采用固定的学习率。

2. Adam利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率,使之更稳定。SGD直接使用当前批次的梯度作为更新幅度。

3. Adam会对梯度进行偏置纠正(bias correction),避免学习率衰减过快。SGD没有此机制。

4. Adam的学习率不需要人工调节,SGD的学习率需要人工选择较优值。

5. Adam对非平稳目标函数收敛快,SGD对高频信号收敛快。

6. Adam需要调节两个超参数β1和β2,SGD只需要一个学习率参数。

7. Adam的计算开销要高于SGD。

总体来说,Adam适用于大多机器学习任务,收敛快,不易产生高方差。SGD对计算资源有限的环境更友好。但SGD也需要较多的超参数调节。在实际使用中,都需要根据具体任务选择优化器并调参。

## 无法理解的地方（待解决）

Adam优化器中偏置纠正(bias correction)的意思是:

Adam计算每个参数的自适应学习率时,会使用梯度的一阶矩估计(平均值)和二阶矩估计(未中心化的方差)。而这两个估计在迭代初期都会被初始化为0,这会导致学习率偏低。

为了解决这个问题,Adam对一阶矩估计和二阶矩估计进行偏置纠正(Bias correction):

1. 一阶矩估计: m_t / (1 - β1^t)  

2. 二阶矩估计: v_t / (1 - β2^t)

这里β1和β2是Adam的两个超参数。

通过放大一阶矩和二阶矩的估计,Adam抵消了它们初始化为0所产生的偏差,使得初始时的学习率不会被拉低。

所以偏置纠正让Adam可以更快地 warmup 并到达合理的学习率水平,从而加速模型的收敛速度。这是Adam相对于其他自适应学习率优化算法的一个重要改进。