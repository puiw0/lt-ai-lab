from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    batch_size = 64
    epoch = 5
    learning_rate = 0.001
    momentum = 0.9
    print_per_step = 100


class MinstNet(nn.Module):
    def __init__(self):
        super(MinstNet, self).__init__()
        self.conv1 = nn.Sequential(
            # 先 padding 填充 再 conv
            # outSize = (width - K + 2P) / S + 1
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            # nn.MaxPool2d(1, 1)
        )

        self.conv2 = nn.Sequential(
            # if padding = 0, inSize = 26 * 26 * 32
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )

        # 与上一个卷积核相乘
        self.fc1 = nn.Sequential(
            # if padding = 0, inSize = 24 * 24 * 32
            nn.Linear(28 * 28 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(256, 10)

    # 重载前向传播方法
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # (batch_size, out_channels, H, W)
        # (64, 64, 28, 28) -> (64, x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Trainer(object):
    def __init__(self):
        self.model = MinstNet().to(device)
        self.train_dataloader, self.test_dataloader = self.load_data()
        # 损失函数：交叉熵
        self.criterion = nn.CrossEntropyLoss()
        # 优化器：SGD or Adam
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.learning_rate)

    @staticmethod
    def load_data():
        print("Loading data...")
        # 下载数据集
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

        # 定义数据迭代器
        train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=Config.batch_size, shuffle=False)

        return train_loader, test_loader

    def train(self):
        steps = 0
        start_time = datetime.now()

        print("Starting training...")
        for epoch in range(Config.epoch):
            print("Epoch {}/{}".format(epoch + 1, Config.epoch))

            for data, label in self.train_dataloader:
                data, label = data.to(device), label.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                if steps % Config.print_per_step == 0:
                    _, predicted = torch.max(output.data, 1)
                    # 自动补全的correct 表达式
                    correct = predicted.eq(label).sum().item()
                    accuracy = correct / Config.batch_size
                    end_time = datetime.now()
                    duration = (end_time - start_time).seconds
                    time_elapsed = "{}m{}s".format(int(duration / 60), duration % 60)
                    msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{}".format(steps, loss, accuracy,
                                                                                            time_elapsed)
                    print(msg)

                steps += 1

        test_loss = 0
        test_correct = 0
        for data, label in self.test_dataloader:
            data, label = data.to(device), label.to(device)
            output = self.model(data)
            loss = self.criterion(output, label)
            test_loss += loss * Config.batch_size
            _, predicted = torch.max(output.data, 1)
            correct = predicted.eq(label).sum()
            test_correct += correct

        # 这里是dataset
        accuracy = test_correct / len(self.test_dataloader.dataset)
        loss = test_loss / len(self.test_dataloader.dataset)
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))

        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print("Time Usage: {:5.2f} min.".format(time_diff / 60.))


if __name__ == '__main__':
    print("is gpu cuda available? {}".format(torch.cuda.is_available()))
    trainer = Trainer()
    trainer.train()
    torch.save(trainer.model.state_dict(), 'model/minst_model.pth')
