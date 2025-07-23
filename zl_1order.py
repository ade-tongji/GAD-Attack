import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
# 假设您的模型定义在 models.py 中
from models import ResNet18_SVHN
from torchvision.models import resnet18


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# --- [注意] 以下生成器G的代码保留，但其训练和使用将被注释掉或替换 ---
class UnetCatGlobalGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_downsampling=2, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', use_dropout=False, max_channel=512):
        assert (n_blocks >= 0)
        super(UnetCatGlobalGenerator, self).__init__()
        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks
        activation = nn.ReLU(True)

        self.first_conv = nn.Sequential(nn.ReflectionPad2d(2),
                                        nn.Conv2d(input_nc, ngf, kernel_size=5, padding=0, bias=False),
                                        norm_layer(ngf))
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            in_ch, out_ch = min(ngf * mult, max_channel), min(ngf * mult * 2, max_channel)
            setattr(self, 'down_sample_%d' % i,
                    nn.Sequential(activation, nn.ReflectionPad2d(1),
                                  nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=0, bias=False),
                                  norm_layer(out_ch)))

        # ### resnet blocks
        mult = 2 ** n_downsampling
        res_ch = min(ngf * mult, max_channel)
        for i in range(n_blocks):
            # if i == 0:
            setattr(self, 'res_block_%d' % i,
                    nn.Sequential(activation, ResnetBlock(res_ch, padding_type=padding_type,
                                                          norm_layer=norm_layer, use_dropout=use_dropout)))

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # in_ch = min(ngf * mult, max_channel) * 2
            in_ch = min(ngf * mult, max_channel)
            out_ch = min(int(ngf * mult / 2), max_channel)
            setattr(self, 'up_sample_%d' % i,
                    nn.Sequential(activation,
                                  nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1,
                                                     output_padding=0, bias=False),
                                  norm_layer(out_ch)))

        self.final_conv = nn.Sequential(activation,
                                        nn.ReflectionPad2d(2),
                                        nn.Conv2d(ngf, output_nc, kernel_size=5, padding=0, bias=False),
                                        nn.Tanh())

    def forward(self, input):
        x_list = []
        x_list.append(self.first_conv(input))
        for i in range(self.n_downsampling):
            x_list.append(getattr(self, 'down_sample_%d' % i)(x_list[-1]))
        x = x_list[-1]
        for i in range(self.n_blocks):
            x = getattr(self, 'res_block_%d' % i)(x)
        for i in range(self.n_downsampling):
            x = getattr(self, 'up_sample_%d' % i)(x)

        x = self.final_conv(x)
        perb = x * 0.2
        return perb


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.05)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # --- [修改] 在这里设置探索模式 ---
    # 可选模式: 'GAUSSIAN', 'FGSM', 'PGD'
    EXPLORATION_MODE = 'GAUSSIAN'
    print(f"当前探索模式: {EXPLORATION_MODE}")

    use_cuda = True
    image_nc = 3
    batch_size = 128  # 减小了batch_size以适应常见GPU显存

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # 加载SVHN数据集
    svhn_train = torchvision.datasets.SVHN('./dataset', split='train', transform=transforms.ToTensor(), download=True)
    svhn_test = torchvision.datasets.SVHN('./dataset', split='test', transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(svhn_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    # 目标模型和教师模型都用ResNet18_SVHN
    target_model = ResNet18_SVHN(num_classes=10).to(device)
    target_model.train()

    teacher_model = ResNet18_SVHN(num_classes=10).to(device)
    # 确保您有这个权重文件，或者使用一个预训练好的模型
    try:
        teacher_model.load_state_dict(torch.load("svhn_resnet18_weights_adv.pth", map_location=device))
    except FileNotFoundError:
        print("警告: 教师模型权重 'svhn_resnet18_weights_adv.pth' 未找到。使用随机初始化的教师模型。")
    teacher_model.eval()

    alpha = 0.3
    temp = 7

    student_loss_fn = nn.CrossEntropyLoss()
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")

    optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)

    # --- [修改] 移除了生成器G的实例化和其优化器，因为不再需要它们 ---
    # netG = UnetCatGlobalGenerator(input_nc=3, output_nc=3).to(device)
    # netG.train()
    # g_optimizer = torch.optim.Adam(netG.parameters(), lr=1e-3)

    # --- [修改] 导入 torchattacks 并实例化所需的攻击 ---
    from torchattacks import PGD, FGSM

    # 攻击参数
    epsilon = 8 / 255
    attack_steps = 7
    step_size = 2 / 255

    fgsm_attack = FGSM(target_model, eps=epsilon)
    pgd_attack = PGD(target_model, eps=epsilon, alpha=step_size, steps=attack_steps, random_start=True)
    # 高斯噪声的标准差
    gaussian_sigma = 8 / 255

    epochs = 100
    n_iter = 0
    for epoch in range(epochs):
        loss_epoch = 0
        loss_epoch1 = 0
        loss_epoch2 = 0
        loss_epoch3 = 0
        num_patch = 0
        for i, data in enumerate(train_dataloader, 0):
            n_iter += 1
            num_patch += 1
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)

            # --- [修改] 移除了生成器G的训练部分 ---
            # 原来的代码会在这里训练生成器G，我们不再需要这部分
            # netG.train()
            # target_model.eval()
            # ... (g_loss.backward(), g_optimizer.step())

            # --- 学生模型训练部分 ---
            target_model.train()

            # --- [修改] 核心部分：根据选择的模式生成邻居样本 ---
            # 将学生模型设置为评估模式以生成对抗样本，防止BN层等更新
            target_model.eval()
            if EXPLORATION_MODE == 'GAUSSIAN':
                # 动态蒸馏 (高斯探索)
                noise = torch.randn_like(train_imgs) * gaussian_sigma
                neighbor_train_imgs = train_imgs + noise
            elif EXPLORATION_MODE == 'FGSM':
                # 动态蒸馏 (FGSM 对抗探索)
                neighbor_train_imgs = fgsm_attack(train_imgs, train_labels)
            elif EXPLORATION_MODE == 'PGD':
                # 动态蒸馏 (PGD 对抗探索)
                neighbor_train_imgs = pgd_attack(train_imgs, train_labels)
            else:
                raise ValueError(f"未知的探索模式: {EXPLORATION_MODE}")

            # 生成邻居后，将学生模型切换回训练模式
            target_model.train()

            # 确保邻居样本在有效范围内，并从计算图中分离
            neighbor_train_imgs = torch.clamp(neighbor_train_imgs, 0., 1.).detach()

            # --- Loss 计算部分（与原代码保持一致） ---
            with torch.no_grad():
                teacher_preds = teacher_model(train_imgs)
                neighbor_teacher_preds = teacher_model(neighbor_train_imgs)
                delta_teacher = (F.softmax(neighbor_teacher_preds, dim=1) - F.softmax(teacher_preds, dim=1)).detach()

            student_preds = target_model(train_imgs)
            neighbor_student_preds = target_model(neighbor_train_imgs)

            student_loss = student_loss_fn(student_preds, train_labels)

            delta_student = F.softmax(neighbor_student_preds, dim=1) - F.softmax(student_preds, dim=1)

            ditillation_loss = divergence_loss_fn(
                F.log_softmax(student_preds / temp, dim=1),
                F.softmax(teacher_preds / temp, dim=1)
            )

            neighbor_ditillation_loss = divergence_loss_fn(
                F.log_softmax(neighbor_student_preds / temp, dim=1),
                F.softmax(neighbor_teacher_preds / temp, dim=1)
            )
            # 这个loss项鼓励学生模型的输出变化与教师模型的输出变化保持一致
            neighbor_mse_loss = F.mse_loss(delta_teacher, delta_student)

            # 最终的损失函数
            # loss_model = alpha * student_loss + (1 - alpha) * ditillation_loss
            # 加上邻居损失项
            loss_model = alpha * student_loss + (
                        1 - alpha) * ditillation_loss + neighbor_mse_loss * 3 + neighbor_ditillation_loss * 3

            loss_epoch += loss_model.item()
            loss_epoch1 += student_loss.item()
            loss_epoch2 += ditillation_loss.item()
            loss_epoch3 += (neighbor_mse_loss.item() * 3 + neighbor_ditillation_loss.item() * 3)  # 合并邻居损失项的记录

            optimizer.zero_grad()
            loss_model.backward()
            optimizer.step()

        print('loss in epoch %d: [student] %.4f [distillation] %.4f [neighbor_loss] %.4f' % (epoch,
                                                                                             loss_epoch1 / num_patch,
                                                                                             loss_epoch2 / num_patch,
                                                                                             loss_epoch3 / num_patch))

    # save model
    # --- [修改] 根据模式保存不同的模型文件名 ---
    targeted_model_file_name = f'SVHN_target_modelC_1order_{EXPLORATION_MODE}.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    print(f"模型已保存至: {targeted_model_file_name}")
    target_model.eval()

    # SVHN test dataset
    test_dataloader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=1)
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img), 1)
        num_correct += torch.sum(pred_lab == test_label, 0)

    print('accuracy in testing set: %f\n' % (num_correct.item() / len(svhn_test)))