## Step 1
训练ResNet18模型：
```
python train_resnet18_svhn_acc.py
```

## Step 2
进行蒸馏：
```
python zl_1order.py
```

## Step 3
攻击测试：
```
python test_blackbox_attack.py
```