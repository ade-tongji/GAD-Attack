
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models import ResNet18_SVHN


# 配置
batch_size = 256
epochs = 50  # 可根据需要调整
learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据增强与加载
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.SVHN('./dataset', split='train', transform=transform_train, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = datasets.SVHN('./dataset', split='test', transform=transform_test, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# 加载模型
model = ResNet18_SVHN(num_classes=10).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ========== PGD攻击相关 ==========
try:
    from torchattacks import PGD
except ImportError:
    PGD = None
    print("未检测到torchattacks库，无法进行PGD攻击评测。请先安装：pip install torchattacks")

def pgd_attack_success_rate(model, loader, eps=8/255, alpha=2/255, steps=7):
    if PGD is None:
        return None
    model.eval()
    attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
    total = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        adv_images = attack(images, labels)
        outputs = model(adv_images)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    succ_rate = 1 - (correct / total)
    return succ_rate


# ====== 第一阶段：正常训练 ======
print("\n===== 阶段1：SVHN正常训练 =====")
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
        if (i+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    print(f"Epoch [{epoch+1}/{epochs}] 训练集精度: {train_acc:.4f}  测试集精度: {test_acc:.4f}")

# 保存正常训练权重
torch.save(model.state_dict(), 'svhn_resnet18_weights_natural.pth')
print("正常训练完成，权重已保存为 svhn_resnet18_weights_natural.pth")

# ====== 第二阶段：PGD对抗训练 ======
print("\n===== 阶段2：PGD对抗训练 =====")
adv_epochs = 50  # 对抗训练轮数，可根据需要调整
for epoch in range(adv_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # 生成PGD对抗样本
        images.requires_grad = True
        adv_images = PGD(model, eps=8/255, alpha=2/255, steps=7)(images, labels)
        optimizer.zero_grad()
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 50 == 0:
            print(f"[ADV] Epoch [{epoch+1}/{adv_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    print(f"[ADV] Epoch [{epoch+1}/{adv_epochs}] 训练集精度: {train_acc:.4f}  测试集精度: {test_acc:.4f}")

    # 每10轮输出一次PGD攻击成功率
    if (epoch + 1) % 10 == 0 or (epoch + 1) == adv_epochs:
        succ_rate = pgd_attack_success_rate(model, test_loader)
        if succ_rate is not None:
            print(f"[PGD] ADV Epoch {epoch+1}: PGD攻击成功率（识别错误率）: {succ_rate:.4f} ({succ_rate*100:.2f}%)")
        else:
            print(f"[PGD] ADV Epoch {epoch+1}: 未检测到torchattacks库，跳过PGD攻击评测。")

# 保存对抗训练权重
torch.save(model.state_dict(), 'svhn_resnet18_weights_adv.pth')
print("对抗训练完成，权重已保存为 svhn_resnet18_weights_adv.pth")

# 最终自然精度
final_acc = evaluate(model, test_loader)
print(f"最终ResNet18_SVHN在SVHN测试集上的自然精度: {final_acc:.4f} ({final_acc*100:.2f}%)")

# 最终PGD攻击成功率
succ_rate = pgd_attack_success_rate(model, test_loader)
if succ_rate is not None:
    print(f"最终PGD攻击成功率（识别错误率）: {succ_rate:.4f} ({succ_rate*100:.2f}%)")
