import torch
import torchvision
import torchvision.transforms as transforms

# 只加载数据用于计算，不进行归一化
transform_calc = transforms.ToTensor()
trainset_calc = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_calc)
trainloader_calc = torch.utils.data.DataLoader(trainset_calc, batch_size=1000, shuffle=False, num_workers=2)

mean = torch.zeros(3)
std = torch.zeros(3)
total_images_count = 0

print("Calculating mean and std for SVHN...")
for images, _ in trainloader_calc:
    batch_samples = images.size(0) # batch size (the last batch can be smaller)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_images_count += batch_samples

mean /= total_images_count
std /= total_images_count

print(f"Calculated Mean: {mean}")
print(f"Calculated Std: {std}")
