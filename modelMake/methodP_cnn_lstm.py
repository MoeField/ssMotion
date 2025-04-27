import os
import gc
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import random

RNG = 42 #random.randint(0, 10000)  # 随机种子
#print(f"Random Seed: {RNG}")
MODEL_NAME_CNNLSTM = "cnn_lstm-m2" # 模型名称
BATCH_SIZE = 32     # 批量大小
TRAIN_RATIO = 0.8   # 训练集划分比例
NUM_EPOCHS = 48     # 训练总周期数
INIT_LR = 0.001     # 初始学习率

# 模型参数
POOL_SIZE = 2
HIDDEN_SIZE = 256
KENERAL_SIZE = 3
CONV_CONV = 48
CONV_LSTM = 64
LSTM_LSTM = 128

ZERO_DIVISION = 0   # 零除数保护

class SensorDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        # 沿用时序结构但排除时间戳
        for action_name in os.listdir(root_dir):
            for csv_file in os.listdir(os.path.join(root_dir, action_name)):
                if not csv_file.endswith('.csv'):
                    continue
                csv_file = os.path.join(root_dir, action_name, csv_file)
                df = pd.read_csv(csv_file)
                # 修改点：排除时间戳列（假设时间戳是第一列）
                tensor = torch.FloatTensor(df.iloc[:, 1:].values)  # 使用iloc跳过第一列
                self.samples.append((tensor, action_name))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, input_size=9, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, CONV_CONV, kernel_size=KENERAL_SIZE, padding=1),    # 32个通道
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE),
            nn.Conv1d(CONV_CONV, CONV_LSTM, kernel_size=KENERAL_SIZE, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE)
        )
        self.lstm = nn.LSTM(CONV_LSTM, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, LSTM_LSTM),
            nn.ReLU(),
            nn.Linear(LSTM_LSTM, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # Conv1d需要通道在前
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # 恢复时序维度
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])

def train_model(train_loader, test_loader, num_epochs:int=48):
        # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM(num_classes=len(le.classes_)).to(device)
    
    # 新增训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    best_acc = 0.0
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 新增训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 验证阶段
        model.eval()
        test_loss = 0.0
        t_correct = 0
        t_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                t_total += labels.size(0)
                t_correct += predicted.eq(labels).sum().item()

        # 记录训练和验证损失
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)
        val_losses.append(test_loss / len(test_loader))
        val_accuracies.append(100 * t_correct / t_total)
        
        # 动态调整学习率
        scheduler.step(100 * t_correct / t_total)
        
        # 保存最佳模型
        current_acc = 100 * t_correct / t_total
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), f'./models/{MODEL_NAME_CNNLSTM}.pth')
        
        # 输出训练进度
        print(
            f'Epoch [{epoch+1}/{num_epochs}] | Loss: {train_loss/len(train_loader):.4f}'
            f' | Acc: {100*correct/total:.2f}%'
            f' | Val Loss: {test_loss/len(test_loader):.4f}'
            f' | Val Acc: {current_acc:.2f}%'
        )
        
        # 清理内存
        del inputs, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    # 最终评估
    model.load_state_dict(torch.load(f'./models/{MODEL_NAME_CNNLSTM}.pth',weights_only=True))
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_, zero_division=ZERO_DIVISION))

    # 保存完整模型
    torch.save(model.state_dict(), f'./models/{MODEL_NAME_CNNLSTM}_v.pth')    
    # 在训练代码最后补充
    from joblib import dump
    dump(le, f'./models/{MODEL_NAME_CNNLSTM}_label_encoder.joblib')

    return train_losses, train_accuracies, val_losses, val_accuracies

import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
def plot_learning_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    绘制学习曲线
    """
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_losses, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"./imgs/学习曲线_{MODEL_NAME_CNNLSTM}.png", dpi=300, transparent=True)
    plt.close()

    print(f"Learning curves saved as ./imgs/学习曲线_{MODEL_NAME_CNNLSTM}.png")


# 沿用SVM的标签编码和评估逻辑
if __name__ == "__main__":
    # 初始化数据集
    full_dataset = SensorDataset("./trainData")
    le = LabelEncoder()
    labels = [sample[1] for sample in full_dataset.samples]
    le.fit(labels)
    
    # 转换标签为编码值（新增关键步骤）
    full_dataset.samples = [
        (tensor, le.transform([label])[0]) 
        for tensor, label in full_dataset.samples
    ]
    
    # 划分训练测试集（添加随机种子保证可复现）
    train_size = int(TRAIN_RATIO * len(full_dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, len(full_dataset) - train_size],
        generator=torch.Generator().manual_seed(RNG)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    train_losses , train_accuracies, val_losses, val_accuracies = train_model(
        train_loader, test_loader, num_epochs=NUM_EPOCHS
    )
    
    # 绘制学习曲线
    plot_learning_curves(train_losses, train_accuracies, val_losses, val_accuracies)


if __name__ == "__main__" and False: #
    #显示网络结构
    from torchsummary import summary
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM(num_classes=10).to(device)
    
    # 打印模型摘要
    #summary(model, input_size=(100, 9))  

    import torch
    import torch.nn as nn
    from torchviz import make_dot

    # 生成随机输入
    x = torch.randn(1, 100, 9).to(device)  # 假设输入序列长度为 100，特征数为 9
    y = model(x)

    # 生成计算图
    dot = make_dot(y, params=dict(model.named_parameters()))
    #dot.attr(rankdir='LR')
    dot.render('cnn_lstm_graph', format='svg', cleanup=True)


    # 导出模型为 ONNX 格式
    torch.onnx.export(model, x, 'cnn_lstm.onnx', export_params=True)