"""
绘制PyTorch CNN和MyTorch MLP的loss曲线对比图
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def read_loss_file(filepath):
    """读取loss文件"""
    epochs = []
    train_losses = []
    test_losses = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()[1:]  # 跳过表头
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    epochs.append(int(parts[0]))
                    train_losses.append(float(parts[1]))
                    test_losses.append(float(parts[2]))
    except FileNotFoundError:
        print(f"文件不存在: {filepath}")
        return None, None, None
    
    return epochs, train_losses, test_losses

# 读取两个模型的loss数据
pytorch_epochs, pytorch_train, pytorch_test = read_loss_file('MINIST/pytorch_cnn_loss.txt')
mytorch_epochs, mytorch_train, mytorch_test = read_loss_file('app/mytorch_mlp_loss.txt')

# 图1: 训练Loss对比
fig1 = plt.figure(figsize=(8, 6))
if pytorch_train is not None:
    plt.plot(pytorch_epochs, pytorch_train, '-', label='PyTorch', linewidth=2.5, color='#1f77b4')
if mytorch_train is not None:
    plt.plot(mytorch_epochs, mytorch_train, '-', label='MyTorch', linewidth=2.5, color='#ff7f0e')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11, framealpha=0.9)
plt.grid(False)
plt.tight_layout()
plt.savefig('train_loss_comparison.png', dpi=300, bbox_inches='tight')
print("训练loss对比图已保存为: train_loss_comparison.png")
plt.close()

# 图2: 测试Loss对比
fig2 = plt.figure(figsize=(8, 6))
if pytorch_test is not None:
    plt.plot(pytorch_epochs, pytorch_test, '-', label='PyTorch', linewidth=2.5, color='#1f77b4')
if mytorch_test is not None:
    plt.plot(mytorch_epochs, mytorch_test, '-', label='MyTorch', linewidth=2.5, color='#ff7f0e')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11, framealpha=0.9)
plt.grid(False)
plt.tight_layout()
plt.savefig('test_loss_comparison.png', dpi=300, bbox_inches='tight')
print("测试loss对比图已保存为: test_loss_comparison.png")
plt.close()

# 打印统计信息
if pytorch_train is not None and mytorch_train is not None:
    print("\n" + "="*60)
    print("Loss统计信息")
    print("="*60)
    print(f"PyTorch CNN - 最终训练Loss: {pytorch_train[-1]:.6f}, 最终测试Loss: {pytorch_test[-1]:.6f}")
    print(f"MyTorch MLP - 最终训练Loss: {mytorch_train[-1]:.6f}, 最终测试Loss: {mytorch_test[-1]:.6f}")
    print("="*60)
