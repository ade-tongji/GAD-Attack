import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

import torchvision.datasets as datasets
from torchattacks import PGD, FGSM
from models import MNISTClassifier


# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_train = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./dataset', train=False, transform=transform, download=True)

# 获取数据集总长度
total_length = len(mnist_train)

# 按比例分割数据集
train_length = int(total_length * 0.9)  # 90% 用于训练
val_length = total_length - train_length  # 剩余部分用于验证

# 动态分割数据集
train_dataset, val_dataset = random_split(mnist_train, [train_length, val_length], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 定义PGD攻击
def create_pgd_attack(model, eps=8/255, alpha=2/255, steps=7):
    return PGD(model, eps=eps, alpha=alpha, steps=steps)

pgd_attack = create_pgd_attack(model)

# 定义FGSM攻击
fgsm_attack = FGSM(model, eps=8/255)

# 对抗训练
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 生成对抗样本
        adv_images = pgd_attack(images, labels)

        # 前向传播
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()


    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

# 保存模型
torch.save(model.state_dict(), "MNIST_target_model.pth")
print("Model saved to MNIST_target_model.pth")
