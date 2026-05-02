import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# 1. 定义经典 LeNet-5 网络（适配 28x28 输入）
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1 卷积层: 输入通道 1, 输出通道 6, 卷积核 5x5, 填充 2 (使 28x28 变成 28x28)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # S2 池化层: 2x2 最大池化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # C3 卷积层: 输入通道 6, 输出通道 16, 卷积核 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # S4 池化层: 2x2 最大池化
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x))) # 输出: 6 x 14 x 14
        x = self.pool2(self.relu(self.conv2(x))) # 输出: 16 x 5 x 5
        x = x.view(x.size(0), -1)               # 展平
        x = self.relu(self.fc1(x))               # 输出: 120
        x = self.relu(self.fc2(x))               # 输出: 84
        x = self.fc3(x)                          # 输出: 10
        return x

# 2. 训练与评估函数
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    print(f"正在使用设备: {device} 进行 LeNet-5 训练...")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    train_duration = time.time() - start_time
    print(f"LeNet-5 训练完成！总耗时: {train_duration:.2f} 秒")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"LeNet-5 测试集准确率: {accuracy:.2f}%")
    return accuracy, train_duration

if __name__ == '__main__':
    train_and_evaluate()