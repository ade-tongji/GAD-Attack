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
# 首先创建一个模型，然后在SVHN上训练它
teacher_model = torchvision.models.resnet18(pretrained=False)  # 不使用预训练权重
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)  # 适配SVHN (10类)

# 静态蒸馏
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


def train_gradient_alignment(student, teacher, dataloader, epochs=10, epsilon=0.01, alpha=0.3, temp=4.0):
    """
    实现一阶蒸馏（基于zl_1order的正确实现）
    使用delta对齐方法：对齐教师和学生在扰动前后的输出差异
    """
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    student_loss_fn = nn.CrossEntropyLoss()
    divergence_loss_fn = nn.KLDivLoss(reduction='batchmean')
    mse_loss = nn.MSELoss()

    student.train()
    teacher.eval()
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_student_loss = 0.0
        running_kd_loss = 0.0
        running_neighbor_loss = 0.0
        running_delta_loss = 0.0
        
        for inputs, labels in tqdm(dataloader, desc=f"First-Order Distillation Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # 生成邻居样本（添加小的随机扰动）
            noise = torch.randn_like(inputs) * epsilon
            neighbor_inputs = torch.clamp(inputs + noise, 0., 1.)

            # 获取教师模型在原始样本和邻居样本上的输出
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
                neighbor_teacher_outputs = teacher(neighbor_inputs)
                # 计算教师模型的输出差异（delta）
                delta_teacher = (torch.nn.functional.softmax(neighbor_teacher_outputs, dim=1) - 
                               torch.nn.functional.softmax(teacher_outputs, dim=1))

            # 获取学生模型在原始样本和邻居样本上的输出
            student_outputs = student(inputs)
            neighbor_student_outputs = student(neighbor_inputs)
            
            # 计算学生模型的输出差异（delta）
            delta_student = (torch.nn.functional.softmax(neighbor_student_outputs, dim=1) - 
                           torch.nn.functional.softmax(student_outputs, dim=1))

            # 1. 标准分类损失
            loss_student = student_loss_fn(student_outputs, labels)

            # 2. 知识蒸馏损失（原始样本）
            loss_kd = divergence_loss_fn(
                torch.nn.functional.log_softmax(student_outputs / temp, dim=1),
                torch.nn.functional.softmax(teacher_outputs / temp, dim=1)
            )

            # 3. 邻居知识蒸馏损失
            loss_neighbor_kd = divergence_loss_fn(
                torch.nn.functional.log_softmax(neighbor_student_outputs / temp, dim=1),
                torch.nn.functional.softmax(neighbor_teacher_outputs / temp, dim=1)
            )

            # 4. Delta对齐损失（一阶蒸馏的核心）
            loss_delta_align = mse_loss(delta_student, delta_teacher)

            # 总损失组合
            total_loss = (alpha * loss_student + 
                         (1 - alpha) * loss_kd + 
                         loss_neighbor_kd * 2.0 + 
                         loss_delta_align * 3.0)

            total_loss.backward()
            optimizer.step()
            
            # 统计损失
            running_loss += total_loss.item()
            running_student_loss += loss_student.item()
            running_kd_loss += loss_kd.item()
            running_neighbor_loss += loss_neighbor_kd.item()
            running_delta_loss += loss_delta_align.item()

        print(f"Epoch {epoch + 1}, Total Loss: {running_loss / len(dataloader):.4f}, "
              f"Student: {running_student_loss / len(dataloader):.4f}, "
              f"KD: {running_kd_loss / len(dataloader):.4f}, "
              f"Neighbor: {running_neighbor_loss / len(dataloader):.4f}, "
              f"Delta: {running_delta_loss / len(dataloader):.4f}")


def train_gradient_alignment_original(student, teacher, dataloader, epochs=10, epsilon=0.01, alpha=0.3, temp=4.0):
    """
    原始一阶蒸馏实现
    """
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    student_loss_fn = nn.CrossEntropyLoss()
    divergence_loss_fn = nn.KLDivLoss(reduction='batchmean')
    mse_loss = nn.MSELoss()

    student.train()
    teacher.eval()
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_student_loss = 0.0
        running_kd_loss = 0.0
        running_neighbor_loss = 0.0
        running_delta_loss = 0.0
        
        for inputs, labels in tqdm(dataloader, desc=f"First-Order Distillation Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # 生成邻居样本（添加小的随机扰动）
            noise = torch.randn_like(inputs) * epsilon
            neighbor_inputs = torch.clamp(inputs + noise, 0., 1.)

            # 获取教师模型在原始样本和邻居样本上的输出
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
                neighbor_teacher_outputs = teacher(neighbor_inputs)
                # 计算教师模型的输出差异（delta）
                delta_teacher = (torch.nn.functional.softmax(neighbor_teacher_outputs, dim=1) - 
                               torch.nn.functional.softmax(teacher_outputs, dim=1))

            # 获取学生模型在原始样本和邻居样本上的输出
            student_outputs = student(inputs)
            neighbor_student_outputs = student(neighbor_inputs)
            
            # 计算学生模型的输出差异（delta）
            delta_student = (torch.nn.functional.softmax(neighbor_student_outputs, dim=1) - 
                           torch.nn.functional.softmax(student_outputs, dim=1))

            # 1. 标准分类损失
            loss_student = student_loss_fn(student_outputs, labels)

            # 2. 知识蒸馏损失（原始样本）
            loss_kd = divergence_loss_fn(
                torch.nn.functional.log_softmax(student_outputs / temp, dim=1),
                torch.nn.functional.softmax(teacher_outputs / temp, dim=1)
            )

            # 3. 邻居知识蒸馏损失
            loss_neighbor_kd = divergence_loss_fn(
                torch.nn.functional.log_softmax(neighbor_student_outputs / temp, dim=1),
                torch.nn.functional.softmax(neighbor_teacher_outputs / temp, dim=1)
            )

            # 4. Delta对齐损失（一阶蒸馏的核心）
            loss_delta_align = mse_loss(delta_student, delta_teacher)

            # 总损失组合
            total_loss = (alpha * loss_student + 
                         (1 - alpha) * loss_kd + 
                         loss_neighbor_kd * 2.0 + 
                         loss_delta_align * 3.0)

            total_loss.backward()
            optimizer.step()
            
            # 统计损失
            running_loss += total_loss.item()
            running_student_loss += loss_student.item()
            running_kd_loss += loss_kd.item()
            running_neighbor_loss += loss_neighbor_kd.item()
            running_delta_loss += loss_delta_align.item()

        print(f"Epoch {epoch + 1}, Total Loss: {running_loss / len(dataloader):.4f}, "
              f"Student: {running_student_loss / len(dataloader):.4f}, "
              f"KD: {running_kd_loss / len(dataloader):.4f}, "
              f"Neighbor: {running_neighbor_loss / len(dataloader):.4f}, "
              f"Delta: {running_delta_loss / len(dataloader):.4f}")


def train_gaussian_probing(student, teacher, dataloader, epochs, sigma=0.01):
    """
    实现动态蒸馏 (高斯探索)
    使用高斯噪声而不是对抗样本来探索决策边界
    """
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    print(f"--- Training with Gaussian Probing (sigma={sigma}) ---")

    for epoch in range(epochs):
        student.train()
        teacher.eval()
        running_loss = 0.0
        running_align_loss = 0.0
        running_kd_loss = 0.0
        
        for inputs, _ in tqdm(dataloader, desc=f"Gaussian Probing Epoch {epoch + 1}/{epochs}"):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # --- 步骤1: 标准知识蒸馏（主要训练目标） ---
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            
            student_outputs = student(inputs)
            
            # 知识蒸馏损失
            T = 4.0
            loss_kd = kl_loss(
                torch.nn.functional.log_softmax(student_outputs / T, dim=1),
                torch.nn.functional.softmax(teacher_outputs / T, dim=1)
            )

            # --- 步骤2: 高斯探索（辅助训练目标） ---
            # 生成高斯噪声扰动
            noise = torch.randn_like(inputs) * sigma
            gaussian_inputs = torch.clamp(inputs + noise, 0., 1.)

            # 在高斯扰动点附近进行梯度对齐
            epsilon = 0.003  # 用于数值梯度计算的小扰动
            
            gaussian_inputs_clone = gaussian_inputs.clone().detach().requires_grad_(True)
            
            u = torch.randn_like(gaussian_inputs_clone, device=device)
            u = u / torch.norm(u.view(u.size(0), -1), p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)

            with torch.no_grad():
                teacher_plus = teacher(gaussian_inputs_clone + epsilon * u)
                teacher_minus = teacher(gaussian_inputs_clone - epsilon * u)
                delta_T = (teacher_plus.sum(dim=1) - teacher_minus.sum(dim=1)) / (2 * epsilon)

            student_outputs_gaussian = student(gaussian_inputs_clone)
            student_sum = student_outputs_gaussian.sum(dim=1).sum()
            grad_S = torch.autograd.grad(student_sum, gaussian_inputs_clone, create_graph=True)[0]
            delta_S = torch.sum(grad_S.view(grad_S.size(0), -1) * u.view(u.size(0), -1), dim=1)

            loss_align = mse_loss(delta_S, delta_T.detach())

            # --- 总损失：主要依靠知识蒸馏 ---
            total_loss = loss_kd + 0.1 * loss_align  # 高斯探索的权重稍高一些
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_align_loss += loss_align.item()
            running_kd_loss += loss_kd.item()

        print(f"Epoch {epoch + 1}, Total Loss: {running_loss / len(dataloader):.4f}, "
              f"KD Loss: {running_kd_loss / len(dataloader):.4f}, "
              f"Gaussian Alignment Loss: {running_align_loss / len(dataloader):.4f}")


def train_adversarial_probing(student, teacher, dataloader, epochs, probing_attack_type='FGSM'):
    """
    实现动态蒸馏 (对抗探索)
    """
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    epsilon = 0.003  # 用于梯度探测的扰动大小（减小到合理范围）

    print(f"--- Training with Adversarial Probing using {probing_attack_type} ---")

    for epoch in range(epochs):
        student.train()
        teacher.eval()
        running_loss = 0.0
        running_align_loss = 0.0
        running_kd_loss = 0.0
        
        for inputs, _ in tqdm(dataloader, desc=f"Adv Probing ({probing_attack_type}) Epoch {epoch + 1}/{epochs}"):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # --- 步骤1: 标准知识蒸馏（主要训练目标） ---
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            
            student_outputs = student(inputs)
            
            # 知识蒸馏损失
            T = 4.0
            loss_kd = kl_loss(
                torch.nn.functional.log_softmax(student_outputs / T, dim=1),
                torch.nn.functional.softmax(teacher_outputs / T, dim=1)
            )

            # --- 步骤2: 对抗探索（辅助训练目标） ---
            # 使用当前学生模型生成对抗样本
            student.eval()
            with torch.no_grad():
                # 创建攻击实例
                probing_atk = get_attack_instance(student, probing_attack_type)
                # 使用教师模型的预测作为目标标签
                pseudo_labels = teacher_outputs.max(1)[1]
                # 生成对抗样本
                try:
                    adv_inputs = probing_atk(inputs, pseudo_labels).detach()
                except:
                    # 如果攻击失败，使用原始输入
                    adv_inputs = inputs
            
            student.train()

            # 在对抗样本附近进行梯度对齐（权重较小）
            if not torch.equal(adv_inputs, inputs):  # 只有当对抗样本不同于原始输入时才进行
                adv_inputs_clone = adv_inputs.clone().detach().requires_grad_(True)
                
                u = torch.randn_like(adv_inputs_clone, device=device)
                u = u / torch.norm(u.view(u.size(0), -1), p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)

                with torch.no_grad():
                    teacher_plus = teacher(adv_inputs_clone + epsilon * u)
                    teacher_minus = teacher(adv_inputs_clone - epsilon * u)
                    delta_T = (teacher_plus.sum(dim=1) - teacher_minus.sum(dim=1)) / (2 * epsilon)

                student_outputs_adv = student(adv_inputs_clone)
                student_sum = student_outputs_adv.sum(dim=1).sum()
                grad_S = torch.autograd.grad(student_sum, adv_inputs_clone, create_graph=True)[0]
                delta_S = torch.sum(grad_S.view(grad_S.size(0), -1) * u.view(u.size(0), -1), dim=1)

                loss_align = mse_loss(delta_S, delta_T.detach())
            else:
                loss_align = torch.tensor(0.0, device=inputs.device)

            # --- 总损失：主要依靠知识蒸馏 ---
            total_loss = loss_kd + 0.05 * loss_align  # 进一步降低对抗探索的权重
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_align_loss += loss_align.item()
            running_kd_loss += loss_kd.item()

        print(f"Epoch {epoch + 1}, Total Loss: {running_loss / len(dataloader):.4f}, "
              f"KD Loss: {running_kd_loss / len(dataloader):.4f}, "
              f"Alignment Loss: {running_align_loss / len(dataloader):.4f}")


def evaluate_attack(target_model, substitute_model, test_loader, attack_name):
    """
    评估黑盒攻击成功率
    """
    substitute_model.eval()
    target_model.eval()

    atk = get_attack_instance(substitute_model, attack_name)
    
    # 统计指标
    total_samples = 0
    correctly_classified = 0  # 原本分类正确的样本数
    attack_successful = 0     # 攻击成功的样本数（原本正确，现在错误）

    for images, labels in tqdm(test_loader, desc=f"Attacking with {attack_name}"):
        images, labels = images.to(device), labels.to(device)
        
        # 先检查原始样本的分类准确性
        with torch.no_grad():
            clean_outputs = target_model(images)
            _, clean_predicted = torch.max(clean_outputs.data, 1)
            clean_correct_mask = (clean_predicted == labels)
            correctly_classified += clean_correct_mask.sum().item()
        
        # 生成对抗样本
        adv_images = atk(images, labels)
        
        # 测试对抗样本
        with torch.no_grad():
            adv_outputs = target_model(adv_images)
            _, adv_predicted = torch.max(adv_outputs.data, 1)
            adv_incorrect_mask = (adv_predicted != labels)
        
        # 只统计原本分类正确但现在分类错误的样本
        successful_attack_mask = clean_correct_mask & adv_incorrect_mask
        attack_successful += successful_attack_mask.sum().item()
        
        total_samples += labels.size(0)

    # 计算指标
    clean_accuracy = 100 * correctly_classified / total_samples
    attack_success_rate = 100 * attack_successful / correctly_classified if correctly_classified > 0 else 0
    
    print(f"  {attack_name}: Clean Acc: {clean_accuracy:.1f}%, "
          f"Attack Success: {attack_successful}/{correctly_classified} ({attack_success_rate:.1f}%)")
    
    return attack_success_rate


def get_attack_instance(model, attack_name):
    """根据名称获取torchattacks的实例"""
    eps = 8 / 255  # 最大扰动限制
    alpha = 2 / 255  # 单步扰动限制
    steps = 7  # 迭代次数

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
        return torchattacks.MIFGSM(model, eps=eps, steps=steps, decay=1.0)
    else:
        raise ValueError(f"Attack {attack_name} not supported.")


def evaluate_white_box_attack(target_model, test_loader, attack_name):
    """评估白盒攻击"""
    target_model.eval()
    atk = get_attack_instance(target_model, attack_name)
    
    # 统计指标
    total_samples = 0
    correctly_classified = 0  # 原本分类正确的样本数
    attack_successful = 0     # 攻击成功的样本数

    for images, labels in tqdm(test_loader, desc=f"White-Box Attacking with {attack_name}"):
        images, labels = images.to(device), labels.to(device)
        
        # 先检查原始样本的分类准确性
        with torch.no_grad():
            clean_outputs = target_model(images)
            _, clean_predicted = torch.max(clean_outputs.data, 1)
            clean_correct_mask = (clean_predicted == labels)
            correctly_classified += clean_correct_mask.sum().item()
        
        # 生成对抗样本
        adv_images = atk(images, labels)
        
        # 测试对抗样本
        with torch.no_grad():
            adv_outputs = target_model(adv_images)
            _, adv_predicted = torch.max(adv_outputs.data, 1)
            adv_incorrect_mask = (adv_predicted != labels)
        
        # 只统计原本分类正确但现在分类错误的样本
        successful_attack_mask = clean_correct_mask & adv_incorrect_mask
        attack_successful += successful_attack_mask.sum().item()
        
        total_samples += labels.size(0)

    # 计算指标
    clean_accuracy = 100 * correctly_classified / total_samples
    attack_success_rate = 100 * attack_successful / correctly_classified if correctly_classified > 0 else 0
    
    print(f"  {attack_name}: Clean Acc: {clean_accuracy:.1f}%, "
          f"Attack Success: {attack_successful}/{correctly_classified} ({attack_success_rate:.1f}%)")
    
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
        # 训练教师模型
        train_teacher(teacher_model, trainloader, testloader, epochs=50)
        torch.save(teacher_model.state_dict(), "teacher_svhn.pth")

    teacher_model.eval()  # 确保教师模型处于评估模式

    ATTACKS_TO_TEST = ['FGSM', 'PGD', 'BIM', 'MI-FGSM', 'NI-FGSM', 'VNI-FGSM', 'PI-FGSM++']
    all_results = collections.OrderedDict()

    # 训练轮数
    EPOCHS = 50

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

    # --- 3. 动态蒸馏 (高斯探索) ---
    print("\n" + "=" * 50)
    print("Step 3: Training with Dynamic Distillation (Gaussian Probing)")
    print("=" * 50)
    student_gaussian = torchvision.models.resnet18(num_classes=10).to(device)
    train_gaussian_probing(student_gaussian, teacher_model, trainloader, epochs=EPOCHS, sigma=0.01)
    gaussian_results = collections.OrderedDict()
    for attack in ATTACKS_TO_TEST:
        acc = evaluate_attack(teacher_model, student_gaussian, testloader, attack)
        gaussian_results[attack] = acc
    all_results['动态蒸馏(高斯)'] = gaussian_results

    # --- 4. 动态蒸馏 (FGSM 对抗探索) ---
    print("\n" + "=" * 50)
    print("Step 4: Training with Dynamic Distillation (FGSM Probing)")
    print("=" * 50)
    student_adv_fgsm = torchvision.models.resnet18(num_classes=10).to(device)
    train_adversarial_probing(student_adv_fgsm, teacher_model, trainloader, epochs=EPOCHS, probing_attack_type='FGSM')
    adv_fgsm_results = collections.OrderedDict()
    for attack in ATTACKS_TO_TEST:
        acc = evaluate_attack(teacher_model, student_adv_fgsm, testloader, attack)
        adv_fgsm_results[attack] = acc
    all_results['动态蒸馏(FGSM)'] = adv_fgsm_results

    # --- 5. 动态蒸馏 (PGD 对抗探索) ---
    print("\n" + "=" * 50)
    print("Step 5: Training with Dynamic Distillation (PGD Probing)")
    print("=" * 50)
    student_adv_pgd = torchvision.models.resnet18(num_classes=10).to(device)
    train_adversarial_probing(student_adv_pgd, teacher_model, trainloader, epochs=EPOCHS, probing_attack_type='PGD')
    adv_pgd_results = collections.OrderedDict()
    for attack in ATTACKS_TO_TEST:
        acc = evaluate_attack(teacher_model, student_adv_pgd, testloader, attack)
        adv_pgd_results[attack] = acc
    all_results['动态蒸馏(PGD)'] = adv_pgd_results

    # --- 6. 一阶蒸馏 (本文算法) ---
    print("\n" + "=" * 50)
    print("Step 6: Training with First-Order Distillation (Gradient Alignment)")
    print("=" * 50)
    student_grad_align = torchvision.models.resnet18(num_classes=10).to(device)
    train_gradient_alignment(student_grad_align, teacher_model, trainloader, epochs=EPOCHS)
    grad_align_results = collections.OrderedDict()
    for attack in ATTACKS_TO_TEST:
        acc = evaluate_attack(teacher_model, student_grad_align, testloader, attack)
        grad_align_results[attack] = acc
    all_results['一阶蒸馏(本文)'] = grad_align_results

    # --- 7. 打印最终结果表 ---
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

    # --- 8. 一阶蒸馏 原始实现 ---
    print("\n" + "=" * 50)
    print("Step 6: Training with Original First-Order Distillation")
    print("=" * 50)
    student_grad_align_original = torchvision.models.resnet18(num_classes=10).to(device)
    train_gradient_alignment_original(student_grad_align_original, teacher_model, trainloader, epochs=EPOCHS)
    grad_align_original_results = collections.OrderedDict()
    for attack in ATTACKS_TO_TEST:
        acc = evaluate_attack(teacher_model, student_grad_align_original, testloader, attack)
        grad_align_original_results[attack] = acc
    all_results['一阶蒸馏(原始)'] = grad_align_original_results

    # --- 7. 打印最终结果表 ---
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