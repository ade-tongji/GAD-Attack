import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torchattacks
import collections  # 用于创建有序字典，保持结果顺序

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 更新数据加载和归一化 ---
# 使用我们为SVHN计算出的均值和标准差
SVHN_MEAN = [0.4377, 0.4438, 0.4728]
SVHN_STD = [0.1201, 0.1231, 0.1052]

transform_svhn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD)
])

# 加载SVHN数据集
trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_svhn)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_svhn)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

print(f"SVHN trainset size: {len(trainset)}")
print(f"SVHN testset size: {len(testset)}")


# === 新增：训练教师模型函数 ===
def train_teacher(model, dataloader, testloader, epochs=10):
    """
    在SVHN上训练教师模型，使其成为一个合格的老师。
    """
    print("--- Training the Teacher Model on SVHN ---")
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in tqdm(dataloader, desc=f"Teacher Training Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 在测试集上评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}, Test Accuracy: {accuracy:.2f}%")
        model.train()

    model.eval()  # 训练结束后，设为评估模式
    print("--- Teacher Model Training Finished ---")
    return model


# 准备模型
# 1. 目标模型（老师），设为评估模式
# 首先创建一个模型，然后在SVHN上训练它，而不是直接用ImageNet预训练模型
teacher_model = torchvision.models.resnet18(pretrained=False)  # 不使用预训练权重
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)  # 适配SVHN (10类)

def train_static_distillation(student, teacher, dataloader, epochs=10, T=4.0):
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    # 使用KL散度损失函数
    distillation_loss = nn.KLDivLoss(reduction='batchmean')

    student.train()
    teacher.eval()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in tqdm(dataloader, desc=f"Static Distillation Epoch {epoch + 1}/{epochs}"):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # 获取老师的软标签 (不需要梯度)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            # 学生模型前向传播
            student_outputs = student(inputs)

            # 计算损失
            loss = distillation_loss(
                torch.nn.functional.log_softmax(student_outputs / T, dim=1),
                torch.nn.functional.softmax(teacher_outputs / T, dim=1)
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.4f}")


def train_gradient_alignment(student, teacher, dataloader, epochs=10, epsilon=0.1, alpha=1.0):
    """
    实现一阶蒸馏（梯度对齐）
    alpha: 梯度对齐损失的权重
    epsilon: 用于有限差分法的扰动大小
    """
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    student.train()
    teacher.eval()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in tqdm(dataloader, desc=f"Gradient Alignment Epoch {epoch + 1}/{epochs}"):
            inputs = inputs.to(device)
            inputs.requires_grad = True

            optimizer.zero_grad()

            # --- 核心：梯度对齐 ---
            # 1. 选择随机方向
            u = torch.randn_like(inputs, device=device)
            u = u / torch.norm(u.view(u.size(0), -1), p=2, dim=1).view(-1, 1, 1, 1)

            # 2. 估算老师的梯度投影 (2次查询)
            with torch.no_grad():
                teacher_plus = teacher(inputs + epsilon * u).sum(dim=1)
                teacher_minus = teacher(inputs - epsilon * u).sum(dim=1)
                delta_T = (teacher_plus - teacher_minus) / (2 * epsilon)

            # 3. 计算学生的梯度投影 (白盒计算)
            student_outputs_sum = student(inputs).sum()

            # 【关键修复】: 设置 create_graph=True 以便让损失可以反向传播到学生模型参数
            grad_S = torch.autograd.grad(student_outputs_sum, inputs, create_graph=True)[0]

            delta_S = torch.sum(grad_S.view(grad_S.size(0), -1) * u.view(u.size(0), -1), dim=1)

            # 4. 定义对齐损失
            loss_align = mse_loss(delta_S, delta_T)
            total_loss = alpha * loss_align

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        print(f"Epoch {epoch + 1}, Alignment Loss: {running_loss / len(dataloader):.4f}")


def train_adversarial_probing(student, teacher, dataloader, epochs, probing_attack_type='FGSM'):
    """
    实现动态蒸馏 (对抗探索)
    """
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    epsilon = 0.1  # 用于梯度探测的扰动大小

    print(f"--- Training with Adversarial Probing using {probing_attack_type} ---")

    for epoch in range(epochs):
        student.train()
        teacher.eval()
        running_loss = 0.0
        for inputs, _ in tqdm(dataloader, desc=f"Adv Probing ({probing_attack_type}) Epoch {epoch + 1}/{epochs}"):
            inputs = inputs.to(device)

            # --- 核心：对抗探索 ---
            # 1. 使用学生模型生成对抗样本作为探测点
            student.eval()
            probing_atk = get_attack_instance(student, probing_attack_type)
            with torch.no_grad():
                pseudo_labels = teacher(inputs).max(1)[1]
            adv_inputs = probing_atk(inputs, pseudo_labels).detach()
            student.train()

            # 2. 在对抗样本点 adv_inputs 附近进行梯度对齐
            adv_inputs.requires_grad = True
            optimizer.zero_grad()

            u = torch.randn_like(adv_inputs, device=device)
            u = u / torch.norm(u.view(u.size(0), -1), p=2, dim=1).view(-1, 1, 1, 1)

            with torch.no_grad():
                teacher_plus = teacher(adv_inputs + epsilon * u).sum(dim=1)
                teacher_minus = teacher(adv_inputs - epsilon * u).sum(dim=1)
                delta_T = (teacher_plus - teacher_minus) / (2 * epsilon)

            student_outputs_sum = student(adv_inputs).sum()
            grad_S = torch.autograd.grad(student_outputs_sum, adv_inputs, create_graph=True)[0]
            delta_S = torch.sum(grad_S.view(grad_S.size(0), -1) * u.view(u.size(0), -1), dim=1)

            loss_align = mse_loss(delta_S, delta_T)

            loss_align.backward()
            optimizer.step()
            running_loss += loss_align.item()

        print(f"Epoch {epoch + 1}, Adversarial Probing Loss: {running_loss / len(dataloader):.4f}")


def evaluate_attack(target_model, substitute_model, test_loader, attack_name):
    """
    评估黑盒攻击成功率
    """
    substitute_model.eval()
    target_model.eval()

    atk = get_attack_instance(substitute_model, attack_name)
    correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc=f"Attacking with {attack_name}"):
        images, labels = images.to(device), labels.to(device)
        adv_images = atk(images, labels)
        outputs = target_model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted != labels).sum().item()

    attack_success_rate = 100 * correct / total
    return attack_success_rate


def get_attack_instance(model, attack_name):
    """根据名称获取torchattacks的实例"""
    eps = 8 / 255
    alpha = 2 / 255
    steps = 10

    if attack_name == 'FGSM':
        return torchattacks.FGSM(model, eps=eps)
    elif attack_name == 'PGD':
        return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    elif attack_name == 'BIM':
        return torchattacks.BIM(model, eps=eps, alpha=alpha, steps=steps)
    elif attack_name == 'MI-FGSM':
        return torchattacks.MIFGSM(model, eps=eps, steps=steps, decay=1.0)
    elif attack_name == 'NI-FGSM':
        return torchattacks.NIFGSM(model, eps=eps, alpha=alpha, steps=steps, decay=1.0)
    elif attack_name == 'VNI-FGSM':
        return torchattacks.VNIFGSM(model, eps=eps, steps=steps, decay=1.0)
    elif attack_name == 'PI-FGSM++':
        print("Warning: PI-FGSM++ not in torchattacks, using MIFGSM with more steps as a placeholder.")
        return torchattacks.MIFGSM(model, eps=eps, steps=20, decay=1.0)
    else:
        raise ValueError(f"Attack {attack_name} not supported.")


def evaluate_white_box_attack(target_model, test_loader, attack_name):
    """评估白盒攻击"""
    target_model.eval()
    atk = get_attack_instance(target_model, attack_name)
    correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc=f"White-Box Attacking with {attack_name}"):
        images, labels = images.to(device), labels.to(device)
        adv_images = atk(images, labels)
        outputs = target_model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted != labels).sum().item()
        total += labels.size(0)

    attack_success_rate = 100 * correct / total
    return attack_success_rate


def run_experiment():
    # 为了让实验可以顺利运行，我们首先训练一个教师模型
    # 如果你已经有训练好的模型，可以注释掉这部分
    try:
        teacher_model.load_state_dict(torch.load("teacher_svhn.pth", map_location=device))
        print("Loaded pre-trained teacher model for SVHN.")
        teacher_model.to(device)
        teacher_model.eval()
    except FileNotFoundError:
        print("No pre-trained teacher model found. Training from scratch... (This may take a while)")
        # 训练一个合格的教师模型，10个epoch可能不足，但作为演示足够了
        train_teacher(teacher_model, trainloader, testloader, epochs=10)
        torch.save(teacher_model.state_dict(), "teacher_svhn.pth")

    teacher_model.eval()  # 确保教师模型处于评估模式

    ATTACKS_TO_TEST = ['FGSM', 'PGD', 'BIM', 'MI-FGSM', 'NI-FGSM', 'VNI-FGSM', 'PI-FGSM++']
    all_results = collections.OrderedDict()

    # 训练轮数（为节省演示时间，设为5，实际研究需要更多）
    EPOCHS = 10

    # --- 1. 白盒攻击 (上限) ---
    print("\n" + "=" * 50)
    print("Step 1: Running White-Box Attacks (Upper Bound)")
    print("=" * 50)
    white_box_results = collections.OrderedDict()
    for attack in ATTACKS_TO_TEST:
        success_rate = evaluate_white_box_attack(teacher_model, testloader, attack)
        white_box_results[attack] = success_rate
    all_results['白盒攻击'] = white_box_results

    # --- 2. 静态蒸馏 ---
    print("\n" + "=" * 50)
    print("Step 2: Training and Evaluating with Static Distillation")
    print("=" * 50)
    student_static = torchvision.models.resnet18(num_classes=10).to(device)
    train_static_distillation(student_static, teacher_model, trainloader, epochs=EPOCHS, T=4.0)
    static_results = collections.OrderedDict()
    for attack in ATTACKS_TO_TEST:
        acc = evaluate_attack(teacher_model, student_static, testloader, attack)
        static_results[attack] = acc
    all_results['静态蒸馏'] = static_results

    # --- 3. 动态蒸馏 (FGSM 对抗探索) ---
    print("\n" + "=" * 50)
    print("Step 3: Training with Dynamic Distillation (FGSM Probing)")
    print("=" * 50)
    student_adv_fgsm = torchvision.models.resnet18(num_classes=10).to(device)
    train_adversarial_probing(student_adv_fgsm, teacher_model, trainloader, epochs=EPOCHS, probing_attack_type='FGSM')
    adv_fgsm_results = collections.OrderedDict()
    for attack in ATTACKS_TO_TEST:
        acc = evaluate_attack(teacher_model, student_adv_fgsm, testloader, attack)
        adv_fgsm_results[attack] = acc
    all_results['动态蒸馏(FGSM)'] = adv_fgsm_results

    # --- 4. 动态蒸馏 (PGD 对抗探索) ---
    print("\n" + "=" * 50)
    print("Step 4: Training with Dynamic Distillation (PGD Probing)")
    print("=" * 50)
    student_adv_pgd = torchvision.models.resnet18(num_classes=10).to(device)
    train_adversarial_probing(student_adv_pgd, teacher_model, trainloader, epochs=EPOCHS, probing_attack_type='PGD')
    adv_pgd_results = collections.OrderedDict()
    for attack in ATTACKS_TO_TEST:
        acc = evaluate_attack(teacher_model, student_adv_pgd, testloader, attack)
        adv_pgd_results[attack] = acc
    all_results['动态蒸馏(PGD)'] = adv_pgd_results

    # --- 5. 一阶蒸馏 (本文算法) ---
    print("\n" + "=" * 50)
    print("Step 5: Training with First-Order Distillation (Gradient Alignment)")
    print("=" * 50)
    student_grad_align = torchvision.models.resnet18(num_classes=10).to(device)
    train_gradient_alignment(student_grad_align, teacher_model, trainloader, epochs=EPOCHS)
    grad_align_results = collections.OrderedDict()
    for attack in ATTACKS_TO_TEST:
        acc = evaluate_attack(teacher_model, student_grad_align, testloader, attack)
        grad_align_results[attack] = acc
    all_results['一阶蒸馏(本文)'] = grad_align_results

    # --- 6. 打印最终结果表 ---
    print("\n" + "=" * 80)
    print("           Final Experimental Results Summary (Attack Success Rate %)")
    print("=" * 80)
    header = f"{'攻击策略':<18}" + "".join([f"{atk_name:<10}" for atk_name in ATTACKS_TO_TEST])
    print(header)
    print("-" * len(header))
    for strategy_name, results in all_results.items():
        row_str = f"{strategy_name:<20}"
        for atk_name in ATTACKS_TO_TEST:
            row_str += f"{results.get(atk_name, 0.0):<10.2f}"
        print(row_str)
    print("=" * 80)


if __name__ == '__main__':
    run_experiment()