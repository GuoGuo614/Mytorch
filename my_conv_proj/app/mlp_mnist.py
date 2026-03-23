import sys
sys.path.append("..")
import mytorch as torch
import mytorch.nn as nn
from mytorch.data import DataLoader
from mytorch.data.datasets import MNISTDataset
import numpy as np
import time
import argparse

np.random.seed(1)


class MLP(nn.Module):
    """
    多层感知机（MLP）- 纯全连接神经网络
    输入: 28x28 = 784
    Hidden1: 512
    Hidden2: 256
    Output: 10
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)
        
        self.fc4 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Input: (batch_size, 784)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x


def train(args, model, train_loader, optimizer, epoch):
    """训练函数"""
    model.train()
    loss_fn = nn.SoftmaxLoss()
    
    epoch_start_time = time.time()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.reset_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        # 统计训练准确率
        pred = torch.ops.argmax(output, axis=1)
        pred_numpy = pred.numpy()
        target_numpy = target.numpy()
        correct = np.sum(pred_numpy == target_numpy)
        total_correct += correct
        total_samples += data.shape[0]
        total_loss += loss.numpy() * data.shape[0]
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * data.shape[0], len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.numpy()))
            if args.dry_run:
                break
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc, epoch_time


def test(model, test_loader):
    """测试函数"""
    model.eval()
    loss_fn = nn.SoftmaxLoss()
    
    test_start_time = time.time()
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        output = model(data)
        test_loss += loss_fn(output, target).numpy() * data.shape[0]
        pred = torch.ops.argmax(output, axis=1)
        pred_numpy = pred.numpy()
        target_numpy = target.numpy()
        correct += np.sum(pred_numpy == target_numpy)
    
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    test_time = time.time() - test_start_time
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time: {:.2f}s\n'.format(
        test_loss, int(correct), len(test_loader.dataset),
        100. * test_acc, test_time))
    
    return test_loss, test_acc, test_time


def main():
    """主函数"""
    # Training settings
    parser = argparse.ArgumentParser(description='MyTorch MLP MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--quick-test', action='store_true',
                        help='quick test mode: use 1000 samples, 2 epochs')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # 加载数据集
    print("Loading MNIST dataset...")
    train_dataset = MNISTDataset(
        "../data/MNIST/raw/train-images-idx3-ubyte.gz",
        "../data/MNIST/raw/train-labels-idx1-ubyte.gz"
    )
    test_dataset = MNISTDataset(
        "../data/MNIST/raw/t10k-images-idx3-ubyte.gz",
        "../data/MNIST/raw/t10k-labels-idx1-ubyte.gz"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # 快速测试模式：只使用部分数据
    if args.quick_test:
        print("🚀 Quick Test Mode: Using 1000 training samples, 500 test samples, 2 epochs")
        # 创建子集
        train_indices = np.arange(min(1000, len(train_dataset)))
        test_indices = np.arange(min(500, len(test_dataset)))
        
        # 简化：直接修改dataset
        train_dataset.images = train_dataset.images[train_indices]
        train_dataset.labels = train_dataset.labels[train_indices]
        test_dataset.images = test_dataset.images[test_indices]
        test_dataset.labels = test_dataset.labels[test_indices]
        
        # 重新创建loader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        
        # 修改epoch数
        args.epochs = 2
    
    # 创建模型
    model = MLP()
    
    # 统计模型参数量
    num_params = sum(p.numpy().size for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # 使用SGD优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': [],
        'test_time': []
    }
    
    best_test_acc = 0.0
    total_start_time = time.time()
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    # 初始化loss记录文件
    with open('mytorch_mlp_loss.txt', 'w') as f:
        f.write("epoch\ttrain_loss\ttest_loss\n")
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, epoch_time = train(args, model, train_loader, optimizer, epoch)
        test_loss, test_acc, test_time = test(model, test_loader)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        history['test_time'].append(test_time)
        
        # 更新最佳准确率
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # 打印epoch总结
        print(f"Epoch {epoch} Summary: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}, Time={epoch_time:.2f}s")
        
        # 保存loss到文件
        with open('mytorch_mlp_loss.txt', 'a') as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{test_loss:.6f}\n")
        
        # 学习率衰减
        if epoch % 1 == 0:
            optimizer.lr = optimizer.lr * args.gamma
    
    total_time = time.time() - total_start_time
    
    # 打印最终统计
    print("\n" + "="*80)
    print("Training Complete - Final Statistics")
    print("="*80)
    print(f"Total Training Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average Time per Epoch: {np.mean(history['epoch_time']):.2f}s")
    print(f"Average Test Time per Epoch: {np.mean(history['test_time']):.2f}s")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f} ({100*history['train_acc'][-1]:.2f}%)")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.4f} ({100*history['test_acc'][-1]:.2f}%)")
    print(f"Best Test Accuracy: {best_test_acc:.4f} ({100*best_test_acc:.2f}%)")
    print(f"Train-Test Gap: {history['train_acc'][-1] - history['test_acc'][-1]:.4f}")
    print(f"Model Parameters: {num_params:,}")
    print("="*80)
    
    if args.save_model:
        print("Model saving not implemented in MyTorch")


if __name__ == '__main__':
    main()
