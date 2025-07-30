import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import ResNet18_SVHN
import torchattacks
from tabulate import tabulate
import os

def main():
    # --- 配置 ---
    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_weight_path = "./svhn_resnet18_weights_adv.pth"
    
    print(f"Using device: {device}")
    
    # --- 加载教师模型 ---
    if not os.path.exists(teacher_weight_path):
        print(f"错误：未找到教师模型权重 '{teacher_weight_path}'。请先运行 train_resnet18_svhn_acc.py。")
        return
        
    print("正在加载教师模型...")
    teacher_model = ResNet18_SVHN(num_classes=10).to(device)
    teacher_model.load_state_dict(torch.load(teacher_weight_path, map_location=device))
    teacher_model.eval()
    print("教师模型加载完成。")

    # --- 定义所有待测试的替代模型路径 ---
    model_paths = {
        
        # "SGM": "./SVHN_target_model_SGM.pth",
        "static": "./SVHN_target_modelC_static_loss3.pth",
        "dynamic(PGD)": "./SVHN_target_modelC_dynamic(PGD)_loss3.pth",
        "dynamic(Gaussian)": "./SVHN_target_modelC_dynamic(Gaussian)_loss3.pth",
        "dynamic(FGSM)": "./SVHN_target_modelC_dynamic(FGSM)_loss3.pth",
        "FOD": "./SVHN_target_modelC_1order_loss3.pth",
        "ODD": "./SVHN_target_model_ODD.pth",
    }
    
    # 过滤掉不存在的权重文件，并给出明确提示
    existing_model_paths = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            existing_model_paths[name] = path
        else:
            print(f"警告：模型 '{name}' 的权重文件 '{path}' 未找到，将跳过此模型的测试。")
    
    if not existing_model_paths:
        print("错误：没有任何有效的替代模型权重文件可供测试。请先运行相应的训练脚本。")
        return

    # --- 加载替代模型 ---
    print("\n正在加载替代模型...")
    models_dict = {k: ResNet18_SVHN(num_classes=10).to(device) for k in existing_model_paths.keys()}
    for name, path in existing_model_paths.items():
        models_dict[name].load_state_dict(torch.load(path, map_location=device))
        models_dict[name].eval()
        print(f" - {name} 模型加载完成。")

    # --- 数据加载 ---
    test_dataset = datasets.SVHN('./dataset', split='test', transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # --- 定义攻击方法 ---
    def get_attacks(model):
        return {
            "FGSM": torchattacks.FGSM(model, eps=8 / 255),
            "PGD": torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7),
            "MIFGSM": torchattacks.MIFGSM(model, eps=8 / 255, steps=7),
            "BIM": torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=7),
            "NIFGSM": torchattacks.NIFGSM(model, eps=8 / 255, steps=7),
            "VNIFGSM": torchattacks.VNIFGSM(model, eps=8 / 255, steps=7),
            "PIFGSM++": torchattacks.PIFGSMPP(model, max_epsilon=8 / 255, num_iter_set=7),
        }

    # --- 测试函数 ---
    def test_attack(surrogate_model, attack, victim_model, loader):
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

    # --- 执行测试 ---
    print("\n开始进行攻击评测...")
    model_names = ["WhiteBox"] + list(existing_model_paths.keys())
    attack_names = list(get_attacks(teacher_model).keys())
    result_dict = {name: {} for name in model_names}

    # 1. 白盒攻击基准
    print("正在测试 WhiteBox 基准...")
    teacher_attacks = get_attacks(teacher_model)
    for atk_name, attack in teacher_attacks.items():
        succ_rate = test_attack(teacher_model, attack, teacher_model, test_loader)
        result_dict["WhiteBox"][atk_name] = f"{succ_rate:.4f}"
    
    # 2. 黑盒攻击
    for model_name, surrogate_model in models_dict.items():
        print(f"正在测试 {model_name} 模型的黑盒攻击能力...")
        surrogate_attacks = get_attacks(surrogate_model)
        for atk_name, attack in surrogate_attacks.items():
            succ_rate = test_attack(surrogate_model, attack, teacher_model, test_loader)
            result_dict[model_name][atk_name] = f"{succ_rate:.4f}"
            
    print("所有评测完成。")
    
    # --- 构造并打印结果表 ---
    table = []
    headers = ["Model"] + attack_names
    
    for model_name in model_names:
        row = [model_name]
        for atk_name in attack_names:
            row.append(result_dict[model_name].get(atk_name, "N/A"))
        table.append(row)
        
    print("\n黑盒攻击成功率对比表:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()