# minst手写识别
## 0. 实验运行
- 安装依赖

`pip install -r requirement.txt`

- 运行 minst_train.py 得到模型 minst_model.pth

- 运行 minst_app.py 打开桌面 app 进行手写识别测试

## 1. 数据集

数据集文件内容

| 文件名                     | 内容                        |
| -------------------------- | --------------------------- |
| train-images-idx3-ubyte.gz | 55000张训练集，5000张验证集 |
| train-labels-idx1-ubyte.gz | 训练集标签                  |
| t10k-images-idx3-ubyte.gz  | 10000张测试集               |
| t10k-labels-idx1-ubyte.gz  | 测试集标签                  |

## 2. 数据集使用

使用pytorch 直接下载、在训练中加载


```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载minst数据集
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# 定义数据迭代器
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 训练
for data, label in train_loader:
    data, label = data.to(device), label.to(device)
    # training...
```


