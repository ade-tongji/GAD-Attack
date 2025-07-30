import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
from models import ResNet18_SVHN
import matplotlib.pyplot as plt
import os
from zl_1order import UnetCatGlobalGenerator

def main():
    # --- 配置 (与zl_1order.py保持一致) ---
    batch_size = 256
    epochs = 100
    lr = 1e-3
    alpha = 0.3
    temp = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for ODD training")

    # --- 数据加载 ---
    transform = transforms.ToTensor()
    svhn_train = torchvision.datasets.SVHN('./dataset', split='train', transform=transform, download=True)
    svhn_test = torchvision.datasets.SVHN('./dataset', split='test', transform=transform, download=True)
    train_loader = DataLoader(svhn_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- 模型初始化 ---
    # 学生模型 (待训练)
    student_model = ResNet18_SVHN(num_classes=10).to(device)
    student_model.train()

    # 教师模型 (固定)
    teacher_model = ResNet18_SVHN(num_classes=10).to(device)
    teacher_model.load_state_dict(torch.load("svhn_resnet18_weights_adv.pth", map_location=device))
    teacher_model.eval()

    # 扰动生成器
    netG = UnetCatGlobalGenerator(input_nc=3, output_nc=3).to(device)
    netG.train()

    # --- 优化器 ---
    optimizer_g = torch.optim.Adam(student_model.parameters(), lr=lr)
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr)
    
    # --- 损失函数 ---
    student_loss_fn = nn.CrossEntropyLoss()
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")

    # --- 损失记录 ---
    history = {
        'student_loss': [], 'loss_ce': [], 'loss_kl_clean': [], 'loss_kl_adv': [],
        'generator_loss': [], 'g_norm': [], 'g_diver': [],
        'val_fidelity': []
    }

    # --- 训练循环 ---
    for epoch in range(epochs):
        student_model.train()
        netG.train()
        
        # 初始化epoch损失记录器
        epoch_losses = {k: 0.0 for k in history.keys() if k not in ['val_fidelity']}
        
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            # === 步骤 1: 训练生成器 G ===
            # 目标: 最大化师生输出分歧 L'_G = - D_KL(f(x+δ) || g(x+δ))
            optimizer_G.zero_grad()
            
            # 冻结学生，只更新生成器
            for param in student_model.parameters(): param.requires_grad = False
            
            delta = netG(imgs)
            x_adv = torch.clamp(imgs + delta, 0., 1.)
            
            g_out_adv_for_G = student_model(x_adv)
            with torch.no_grad():
                f_out_adv_for_G = teacher_model(x_adv)
            
            # ODD核心：最大化输出分歧，即最小化负KL散度
            g_diver = -F.kl_div(F.log_softmax(f_out_adv_for_G, dim=1), F.softmax(g_out_adv_for_G, dim=1), reduction='batchmean')
            
            # 保留扰动大小的正则化项
            g_norm = torch.mean(torch.norm(delta.view(batch_size, -1), p=2, dim=1))
            
            # 生成器总损失 
            generator_loss = g_norm * 0.1 + g_diver * 1.0
            
            generator_loss.backward()
            optimizer_G.step()
            
            # 解冻学生
            for param in student_model.parameters(): param.requires_grad = True

            # === 步骤 2: 训练学生模型 g ===
            optimizer_g.zero_grad()
            
            # 使用更新后的G生成新的扰动，并阻断梯度流
            delta = netG(imgs).detach()
            x_adv = torch.clamp(imgs + delta, 0., 1.)

            # 计算各项损失
            with torch.no_grad():
                f_out_clean = teacher_model(imgs)
                f_out_adv = teacher_model(x_adv)

            g_out_clean = student_model(imgs)
            g_out_adv = student_model(x_adv)
            
            # 学生损失 L'_student (无梯度/差异项)
            loss_ce = student_loss_fn(g_out_clean, labels)
            loss_kl_clean = divergence_loss_fn(F.log_softmax(g_out_clean / temp, dim=1), F.softmax(f_out_clean / temp, dim=1))
            loss_kl_adv = divergence_loss_fn(F.log_softmax(g_out_adv / temp, dim=1), F.softmax(f_out_adv / temp, dim=1))
            
            student_loss = alpha * loss_ce + (1 - alpha) * loss_kl_clean + loss_kl_adv * 3
            
            student_loss.backward()
            optimizer_g.step()

            # 累加epoch损失
            epoch_losses['student_loss'] += student_loss.item()
            epoch_losses['loss_ce'] += loss_ce.item()
            epoch_losses['loss_kl_clean'] += loss_kl_clean.item()
            epoch_losses['loss_kl_adv'] += loss_kl_adv.item()
            epoch_losses['generator_loss'] += generator_loss.item()
            epoch_losses['g_norm'] += g_norm.item()
            epoch_losses['g_diver'] += g_diver.item()

        # --- 验证与记录 ---
        student_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                g_out, f_out = student_model(imgs), teacher_model(imgs)
                val_correct += (g_out.argmax(dim=1) == f_out.argmax(dim=1)).sum().item()
                val_total += imgs.size(0)
        
        num_batches = len(train_loader)
        for k in epoch_losses:
            history[k].append(epoch_losses[k] / num_batches)
        history['val_fidelity'].append(val_correct / val_total)
        
        print(f"[ODD] Epoch {epoch+1}/{epochs} | Student Loss: {history['student_loss'][-1]:.4f} | "
              f"Generator Loss: {history['generator_loss'][-1]:.4f} | Val Fidelity: {history['val_fidelity'][-1]:.4f}")

    # --- 保存模型 ---
    torch.save(student_model.state_dict(), 'SVHN_target_model_ODD.pth')
    print("ODD模型训练完成，已保存为 SVHN_target_model_ODD.pth")

    # --- 绘图分析 ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('ODD Training Analysis')
    # 学生损失分解
    axs[0, 0].plot(history['student_loss'], label='Total Student Loss')
    axs[0, 0].plot(history['loss_ce'], label='CrossEntropy (clean)', linestyle='--')
    axs[0, 0].plot(history['loss_kl_clean'], label='KL Div (clean)', linestyle='--')
    axs[0, 0].plot(history['loss_kl_adv'], label='KL Div (adv)', linestyle='--')
    axs[0, 0].set_title('Student Loss Components')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    # 生成器损失分解
    axs[0, 1].plot(history['generator_loss'], label='Total Generator Loss', color='red')
    axs[0, 1].plot(history['g_norm'], label='Perturbation Norm', color='orange', linestyle='--')
    axs[0, 1].plot(history['g_diver'], label='Neg. Output Divergence', color='magenta', linestyle='--')
    axs[0, 1].set_title('Generator Loss Components')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    # 验证集保真度
    axs[1, 1].plot(history['val_fidelity'], label='Validation Fidelity', color='purple')
    axs[1, 1].set_title('Validation Fidelity with Teacher')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # 移除一个子图以保持布局整洁
    fig.delaxes(axs[1,0])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('training_analysis_ODD.png')
    plt.show()

if __name__ == '__main__':
    main()