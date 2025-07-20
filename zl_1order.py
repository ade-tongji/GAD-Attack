import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
from models import  MNIST_target_net, MNISTClassifier, MNISTClassifierA, MNISTClassifierC, MNISTClassifierD


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
            # in_ch = min(ngf * mult, max_channel) * 2
            in_ch = min(ngf * mult, max_channel)
            out_ch = min(int(ngf * mult / 2), max_channel)
            setattr(self, 'up_sample_%d' % i,
                    nn.Sequential(activation,
                                  nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
                                  norm_layer(out_ch)))

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
    use_cuda = True
    image_nc = 3
    batch_size = 256

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # 加载SVHN数据集
    import torchvision
    from models import ResNet18_SVHN
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
    fgsm_attack = FGSM(target_model, attack_config)


    epochs = 100
    success_rate = 0
    n_iter = 0
    for epoch in range(epochs):
        loss_epoch = 0
        loss_epoch1 = 0
        loss_epoch2 = 0
        loss_epoch3 = 0
        g_loss_epoch1 = 0
        g_loss_epoch2 = 0
        num_patch = 0
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
            g_norm = torch.clamp_min(grad.view(batch_size, -1).norm(2, dim=1), 2).mean()
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

            
            netG.eval()
            target_model.train()

            # if n_iter % 5 != -1:
            #     continue

            grad = netG(train_imgs)
            neighbor_train_imgs = torch.clamp(train_imgs + grad, 0., 1.)
            neighbor_train_imgs = neighbor_train_imgs.detach()

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

            loss_model = alpha * student_loss + (1 - alpha) * ditillation_loss + neighbor_ditillation_loss * 3
            # loss_model = alpha * student_loss + (1 - alpha) * ditillation_loss + neighbor_mse_loss * 3 + neighbor_ditillation_loss * 3
            # loss_model = alpha * student_loss + (1 - alpha) * ditillation_loss

            loss_epoch += loss_model.item()
            loss_epoch1 += student_loss.item()
            loss_epoch2 += ditillation_loss.item()
            loss_epoch3 += neighbor_ditillation_loss.item()
            optimizer.zero_grad()
            loss_model.backward()
            optimizer.step()

        print('loss in epoch %d: [student] %f [ditillation] %f [neighbor] %f' % (epoch, loss_epoch1/num_patch, loss_epoch2/num_patch, loss_epoch3/num_patch))
        print('\tG loss in epoch %d: [norm] %f [diver] %f' % (epoch, g_loss_epoch1/num_patch, g_loss_epoch2/num_patch))

    # save model
    targeted_model_file_name = './SVHN_target_modelC_dynamic_loss3.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()

    # SVHN test dataset
    test_dataloader = DataLoader(svhn_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img), 1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('accuracy in testing set: %f\n'%(num_correct.item() / len(svhn_test)))