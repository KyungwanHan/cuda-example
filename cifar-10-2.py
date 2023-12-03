import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# 컨볼루션 신경망 정의
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CIFAR-10 데이터셋 로드
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


# CPU에서 신경망 초기화
net_cpu = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_cpu.parameters(), lr=0.001, momentum=0.9)

# 학습 시작 시간 측정
start_time_cpu = time.time()

# 학습 과정
for epoch in range(2):  # 데모를 위해 2 에포크만 실행
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net_cpu(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 학습 종료 시간 측정
end_time_cpu = time.time()
print(f"CPU training time: {end_time_cpu - start_time_cpu:.3f} seconds")

# CUDA 사용 가능 여부 확인
print(f"CUDA is available(): {torch.cuda.is_available()}")

# GPU에서 신경망 초기화 (CUDA 사용 가능한 경우)
net_gpu = ConvNet()
if torch.cuda.is_available():
    net_gpu.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_gpu.parameters(), lr=0.001, momentum=0.9)

# 학습 시작 시간 측정
start_time_gpu = time.time()

# 학습 과정
for epoch in range(2):  # 데모를 위해 2 에포크만 실행
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data


# 학습 종료 시간 측정
end_time_gpu = time.time()
print(f"GPU training time: {end_time_gpu - start_time_gpu:.3f} seconds")