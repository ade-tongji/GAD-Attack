import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import ResNet18_SVHN
teacher_weight_path = "./svhn_resnet18_weights_adv.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_teacher_model(weight_path):
    model = ResNet18_SVHN(num_classes=10).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

teacher_model = load_teacher_model(teacher_weight_path)

import torchattacks
from tabulate import tabulate

# 配置
batch_size = 256

# 1. 加载模型

model_paths = {
    "static": "./SVHN_target_modelC_static_loss3.pth",
    "dynamic": "./SVHN_target_modelC_dynamic_loss3.pth",
    "first_order": "./SVHN_target_modelC_1order_loss3.pth"
}

def load_model(weight_path):
    model = ResNet18_SVHN(num_classes=10).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

models_dict = {k: load_model(v) for k, v in model_paths.items()}


# 2. 加载SVHN测试集
test_dataset = datasets.SVHN('./dataset', split='test', transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 3. 定义攻击方法
def get_attacks(model):
    return {
        "FGSM": torchattacks.FGSM(model, eps=8 / 255),
        "PGD": torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7),
        "BIM": torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=7),
        "MIFGSM": torchattacks.MIFGSM(model, eps=8 / 255, steps=7),
        "NIFGSM": torchattacks.NIFGSM(model, eps=8 / 255, steps=7),
        "VNIFGSM": torchattacks.VNIFGSM(model, eps=8 / 255, steps=7),
        "PIFGSM++": torchattacks.PIFGSMPP(model, max_epsilon= 8 / 255, num_iter_set=7),
    }


# 4. 测试函数
# 白盒攻击：直接对teacher_model攻击和评测
def test_attack_whitebox(model, attack, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack(images, labels)
        outputs = model(adv_images)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 1 - correct / total

# 黑盒攻击：用surrogate_model生成对抗样本，用victim_model(teacher)评测
def test_attack_blackbox(surrogate_model, attack, victim_model, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack(images, labels)
        outputs = victim_model(adv_images)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 1 - correct / total

# 5. 循环测试

if __name__ == "__main__":
    # 统一收集所有模型和攻击方式
    model_names = ["WhiteBox"] + list(models_dict.keys())
    attack_names = list(get_attacks(teacher_model).keys())
    # 结果字典：{模型名: {攻击名: 攻击成功率}}
    result_dict = {name: {} for name in model_names}


    # 教师模型白盒攻击
    teacher_attacks = get_attacks(teacher_model)
    for atk_name, attack in teacher_attacks.items():
        acc = test_attack_whitebox(teacher_model, attack, test_loader)
        result_dict["WhiteBox"][atk_name] = f"{acc:.4f}"

    # 学生模型黑盒攻击：用学生模型生成对抗样本，用teacher_model评测
    for model_name, model in models_dict.items():
        attacks = get_attacks(model)
        for atk_name, attack in attacks.items():
            acc = test_attack_blackbox(model, attack, teacher_model, test_loader)
            result_dict[model_name][atk_name] = f"{acc:.4f}"

    # 构造表格
    table = []
    for model_name in model_names:
        row = [model_name]
        for atk_name in attack_names:
            row.append(result_dict[model_name].get(atk_name, "-"))
        table.append(row)

    headers = ["Model"] + attack_names
    print("\n准确率统计表：")
    print(tabulate(table, headers=headers, tablefmt="grid"))
