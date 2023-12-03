import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import platform

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

# CIFAR-10 데이터셋 로드 및 기타 설정 함수
def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    return trainloader

# 학습 함수
def train(net, trainloader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 학습 시작 시간 측정
    start_time = time.time()

    # 학습 과정
    for epoch in range(2):  # 데모를 위해 2 에포크만 실행
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # 데이터를 지정된 장치로 이동
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 학습 종료 시간 측정
    end_time = time.time()
    return end_time - start_time

def main():
    trainloader = load_data()

    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
        # MacOS에서 Apple Silicon의 Metal 가용성 확인
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Training on {device}")

    # 신경망 초기화 및 장치 할당
    net = ConvNet().to(device)

    # 학습
    training_time = train(net, trainloader, device)
    print(f"Training time on {device}: {training_time:.3f} seconds")

    # CPU에서 학습과 비교
    if device is not torch.device("cpu"):
        device = torch.device("cpu")
        net = ConvNet().to(device)

        # CPU에서 학습
        training_time = train(net, trainloader, device)  # 'device' 인자 추가
        print(f"Training time on CPU: {training_time:.3f} seconds")

if __name__ == '__main__':
    main()