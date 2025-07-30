import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
from models import  MNIST_target_net, MNISTClassifier, MNISTClassifierA, MNISTClassifierC, MNISTClassifierD, ResNet18_SVHN


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


class UnetCatGlobalGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_downsampling=2, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', use_dropout=False, max_channel=512):
        assert(n_blocks >= 0)
        super(UnetCatGlobalGenerator, self).__init__()
        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks
        activation = nn.ReLU(True)

        self.first_conv = nn.Sequential(nn.ReflectionPad2d(2),
                                        nn.Conv2d(input_nc, ngf, kernel_size=5, padding=0, bias=False),
                                        norm_layer(ngf))
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            in_ch, out_ch = min(ngf * mult, max_channel), min(ngf * mult * 2, max_channel)
            setattr(self, 'down_sample_%d' % i,
                    nn.Sequential(activation, nn.ReflectionPad2d(1),
                                  nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=0, bias=False),
                                  norm_layer(out_ch)))

        # ### resnet blocks
        mult = 2**n_downsampling
        res_ch = min(ngf * mult, max_channel)
        for i in range(n_blocks):
            # if i == 0:
            setattr(self, 'res_block_%d' % i,
                    nn.Sequential(activation, ResnetBlock(res_ch, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout)))

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            in_ch = min(ngf * mult, max_channel)
            out_ch = min(int(ngf * mult / 2), max_channel)
            setattr(self, 'up_sample_%d' % i,
                    nn.Sequential(
                        activation,
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False),
                        norm_layer(out_ch)
                    ))

        self.final_conv = nn.Sequential(activation,
                                        nn.ReflectionPad2d(2), nn.Conv2d(ngf, output_nc, kernel_size=5, padding=0, bias=False),
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
        perb = x * 8 / 255
        return perb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.05)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    def save_images_grid(imgs, filename, nrow=4):
        # imgs: [N, C, H, W], 0~1
        from torchvision.utils import make_grid
        grid = make_grid(imgs, nrow=nrow, padding=2, normalize=True)
        npimg = grid.cpu().numpy()
        plt.figure(figsize=(nrow, nrow))
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1,2,0)))
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    import matplotlib.pyplot as plt
    use_cuda = True
    image_nc = 3
    batch_size = 256

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # 加载SVHN数据集
    import torchvision
    # from models import ResNet18_SVHN
    svhn_train = torchvision.datasets.SVHN('./dataset', split='train', transform=transforms.ToTensor(), download=True)
    svhn_test = torchvision.datasets.SVHN('./dataset', split='test', transform=transforms.ToTensor(), download=True)
    # 这里直接用全部训练集
    train_dataloader = DataLoader(svhn_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)


    # 目标模型和教师模型都用ResNet18_SVHN
    target_model = ResNet18_SVHN(num_classes=10).to(device)
    target_model.train()

    teacher_model = ResNet18_SVHN(num_classes=10).to(device)
    teacher_model.load_state_dict(torch.load("svhn_resnet18_weights_adv.pth", map_location=device))
    teacher_model.eval()

    alpha = 0.3
    temp = 7

    student_loss_fn = nn.CrossEntropyLoss()
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")

    optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)

    # 生成器输入输出通道都为3
    netG = UnetCatGlobalGenerator(input_nc=3, output_nc=3).to(device)
    netG.train()
    g_optimizer = torch.optim.Adam(netG.parameters(), lr=1e-3)

    from torchattacks import PGD, FGSM, MIFGSM

    kkk = 2
    # attack_config = {
    #     'eps' : .3 / 3 * kkk, 
    #     'attack_steps': 10,
    #     'attack_lr': 0.05 / 3 * kkk, 
    #     'random_init': True, 
    # }
    attack_config = {
        # 'eps' : .3 / 3 * kkk, 
        'eps' : 8/255, 
        'attack_steps': 7,
        'attack_lr': 2/255, 
        'random_init': False, 
    }
    pgd_attack = PGD(target_model, eps=8/255, alpha=2/255, steps=7, random_start=True)


    epochs = 100
    success_rate = 0
    n_iter = 0
    # 新增：分别记录五个loss的曲线
    model_loss_curve = []
    gloss_curve = []
    student_loss_curve = []
    ditillation_loss_curve = []
    neighbor_ditillation_loss_curve = []
    g_norm_curve = []
    g_diver_curve = []
    for epoch in range(epochs):
        val_loss_epoch = 0
        val_num = 0
        loss_epoch = 0
        loss_epoch1 = 0
        loss_epoch2 = 0
        loss_epoch3 = 0
        g_loss_epoch1 = 0
        g_loss_epoch2 = 0
        num_patch = 0
        gloss_epoch = 0  # 每个epoch累加生成器loss
        for i, data in enumerate(train_dataloader, 0):
            n_iter += 1
            num_patch += 1
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)

            # neighbor_train_imgs = fgsm_attack(train_imgs, train_labels).detach()
            
            netG.train()
            target_model.eval()
            grad = netG(train_imgs)
            # neighbor_train_imgs = train_imgs + grad
            neighbor_train_imgs = torch.clamp(train_imgs + grad, 0., 1.)

            with torch.no_grad():
                teacher_x_preds = teacher_model(train_imgs).detach()
                teacher_preds = teacher_model(neighbor_train_imgs).detach()
                
                delta_teacher = (F.softmax(teacher_preds, dim=1) - F.softmax(teacher_x_preds, dim=1)).detach()

            student_preds = target_model(neighbor_train_imgs)
            student_x_preds = target_model(train_imgs)

            delta_student = F.softmax(student_preds, dim=1) - F.softmax(student_x_preds, dim=1)
            

            # print(grad.view(batch_size, -1).norm(2, dim=1).size())
            g_norm = torch.clamp_min(grad.view(batch_size, -1).norm(2, dim=1), 1).mean()
            # g_diver = -divergence_loss_fn(
            #     F.log_softmax(student_preds, dim=1),
            #     F.softmax(teacher_preds, dim=1)
            # )
            # g_diver = -F.mse_loss(F.softmax(student_preds, dim=1) - F.softmax(teacher_preds, dim=1), torch.zeros_like(teacher_preds).to(device))
            g_diver = -F.mse_loss(delta_teacher - delta_student, torch.zeros_like(delta_student).to(device))

            g_loss = g_norm * 0.1 + g_diver * 100
            # g_loss = g_diver * 10
            g_optimizer.zero_grad()
            # print(netG.final_conv[2].weight.grad)
            g_loss.backward()
            # print()
            g_optimizer.step()

            g_loss_epoch1 += g_norm.item()
            g_loss_epoch2 += g_diver.item()
            gloss_epoch += g_loss.item()

            
            netG.eval()
            target_model.train()

            # if n_iter % 5 != -1:
            #     continue

            grad = netG(train_imgs)
            neighbor_train_imgs = torch.clamp(train_imgs + grad, 0., 1.)
            neighbor_train_imgs = neighbor_train_imgs.detach()

            # train_imgs.requires_grad = True
            # neighbor_train_imgs = pgd_attack(train_imgs, train_labels).detach()

            with torch.no_grad():
                # neighbor_train_imgs = torch.clamp(train_imgs + torch.randn_like(train_imgs) * 8/255, 0., 1.)
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
            neighbor_mse_loss = F.mse_loss(delta_teacher - delta_student, torch.zeros_like(delta_student).to(device))


            # loss_model = F.cross_entropy(student_preds, train_labels)
            logits_label = torch.argmax(student_preds, dim=1)
            success_rate += torch.sum(logits_label == train_labels)/train_labels.shape[0]

            # loss_model = alpha * student_loss + (1 - alpha) * ditillation_loss + neighbor_ditillation_loss * 3
            loss_model = alpha * student_loss + (1 - alpha) * ditillation_loss + neighbor_mse_loss * 3 + neighbor_ditillation_loss * 3
            # loss_model = alpha * student_loss + (1 - alpha) * ditillation_loss

            loss_epoch += loss_model.item()
            loss_epoch1 += student_loss.item()
            loss_epoch2 += ditillation_loss.item()
            loss_epoch3 += neighbor_ditillation_loss.item()
            optimizer.zero_grad()
            loss_model.backward()
            optimizer.step()

        avg_model_loss = loss_epoch / num_patch
        avg_gloss = gloss_epoch / num_patch  # 统计整个epoch的平均生成器loss
        avg_student_loss = loss_epoch1 / num_patch
        avg_ditillation_loss = loss_epoch2 / num_patch
        avg_neighbor_ditillation_loss = loss_epoch3 / num_patch
        avg_g_norm = g_loss_epoch1 / num_patch
        avg_g_diver = g_loss_epoch2 / num_patch
        model_loss_curve.append(avg_model_loss)
        gloss_curve.append(avg_gloss)
        student_loss_curve.append(avg_student_loss)
        ditillation_loss_curve.append(avg_ditillation_loss)
        neighbor_ditillation_loss_curve.append(avg_neighbor_ditillation_loss)
        g_norm_curve.append(avg_g_norm)
        g_diver_curve.append(avg_g_diver)
        print('loss in epoch %d: [student] %f [ditillation] %f [neighbor] %f' % (epoch, avg_student_loss, avg_ditillation_loss, avg_neighbor_ditillation_loss))
        print('\tG loss in epoch %d: [norm] %f [diver] %f' % (epoch, avg_g_norm, avg_g_diver))
       
        # 计算验证损失（validation loss）
        target_model.eval()
        with torch.no_grad():
            for val_imgs, val_labels in DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=1):
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                val_preds = target_model(val_imgs)
                val_loss = student_loss_fn(val_preds, val_labels)
                val_loss_epoch += val_loss.item() * val_imgs.size(0)
                val_num += val_imgs.size(0)
        avg_val_loss = val_loss_epoch / val_num
        if epoch == 0:
            val_loss_curve = []
        val_loss_curve.append(avg_val_loss)
        print(f'Validation loss in epoch {epoch}: {avg_val_loss:.4f}')

    # save model
    # 选择loss最小的epoch
    min_model_loss = min(model_loss_curve)
    min_model_epoch = model_loss_curve.index(min_model_loss)
    # gloss取最小值的epoch
    min_gloss = min(gloss_curve)
    min_gloss_epoch = gloss_curve.index(min_gloss)
    print(f"Target Model Loss最低的epoch: {min_model_epoch+1}, loss={min_model_loss:.4f}")
    print(f"Generator Loss最小的epoch: {min_gloss_epoch+1}, loss={min_gloss:.4f}")
    targeted_model_file_name = './SVHN_target_modelC_1order_loss3.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()

    # 保存最优generator生成的扰动图和原图（已注释，按需恢复）
    # print(f"保存最优generator(epoch={min_gloss_epoch+1})生成的扰动图...")
    # sample_loader = DataLoader(svhn_test, batch_size=32, shuffle=True)
    # sample_imgs, sample_labels = next(iter(sample_loader))
    # sample_imgs = sample_imgs.to(device)
    # sample_labels = sample_labels.to(device)
    # idx = np.random.choice(sample_imgs.shape[0], 16, replace=False)
    # # 原图
    # save_images_grid(sample_imgs[idx], 'final_original_images.png')
    # # PGD扰动
    # sample_imgs.requires_grad_()  # 确保PGD攻击能计算梯度
    # pgd_imgs = pgd_attack(sample_imgs, sample_labels)
    # pgd_imgs = torch.clamp(pgd_imgs, 0., 1.)
    # save_images_grid(pgd_imgs[idx], 'final_pgd_images.png')
    # # 生成器扰动
    # with torch.no_grad():
    #     gen_imgs = torch.clamp(sample_imgs + netG(sample_imgs), 0., 1.)
    #     save_images_grid(gen_imgs[idx], 'final_generator_images.png')

    # 计算G Norm和G Diver的validation loss
    g_norm_val_curve = []
    g_diver_val_curve = []
    for epoch in range(epochs):
        print("epoch %d validation G Norm and G Diver loss..." % epoch)
        val_g_norm = 0
        val_g_diver = 0
        val_num = 0
        target_model.eval()
        netG.eval()
        with torch.no_grad():
            for val_imgs, val_labels in DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=1):
                val_imgs = val_imgs.to(device)
                grad = netG(val_imgs)
                neighbor_imgs = torch.clamp(val_imgs + grad, 0., 1.)
                teacher_preds = teacher_model(neighbor_imgs)
                teacher_x_preds = teacher_model(val_imgs)
                student_preds = target_model(neighbor_imgs)
                student_x_preds = target_model(val_imgs)
                delta_teacher = F.softmax(teacher_preds, dim=1) - F.softmax(teacher_x_preds, dim=1)
                delta_student = F.softmax(student_preds, dim=1) - F.softmax(student_x_preds, dim=1)
                g_norm = torch.clamp_min(grad.view(grad.size(0), -1).norm(2, dim=1), 1).mean()
                g_diver = -F.mse_loss(delta_teacher - delta_student, torch.zeros_like(delta_student).to(device))
                val_g_norm += g_norm.item() * val_imgs.size(0)
                val_g_diver += g_diver.item() * val_imgs.size(0)
                val_num += val_imgs.size(0)
        g_norm_val_curve.append(val_g_norm / val_num)
        g_diver_val_curve.append(val_g_diver / val_num)

    # 绘制G Norm Loss曲线（train/val）
    plt.figure(figsize=(8,5))
    plt.plot(g_norm_curve, label='Train G Norm Loss', color='red')
    plt.plot(g_norm_val_curve, label='Validation G Norm Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('G Norm Loss Curve (Train/Validation)')
    plt.legend()
    plt.grid(True)
    plt.savefig('distillation_loss_curve_1order_gnorm.png')
    plt.show()

    # 绘制G Diver Loss曲线（train/val）
    plt.figure(figsize=(8,5))
    plt.plot(g_diver_curve, label='Train G Diver Loss', color='purple')
    plt.plot(g_diver_val_curve, label='Validation G Diver Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('G Diver Loss Curve (Train/Validation)')
    plt.legend()
    plt.grid(True)
    plt.savefig('distillation_loss_curve_1order_gdiver.png')
    plt.show()

    # SVHN test dataset
    test_dataloader = DataLoader(svhn_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img), 1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('accuracy in testing set: %f\n'%(num_correct.item() / len(svhn_test)))