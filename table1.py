from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split



pi = torch.tensor(np.pi)

def gaussian_nll(mean, variance, y):
    return torch.mean((torch.log(2*pi*variance) * 0.5) + ((0.5 * (y - mean) ** 2) / variance))


class GaussianMultiLayerPerceptron(nn.Module):  # this one
    
    def __init__(self, input_dim, output_dim=2):
        
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 50)
        self.fc2 = nn.Linear(50, self.output_dim)
        self.relu = nn.ReLU()
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        mean, variance = torch.split(x, 1, dim=1)
        mean = mean.squeeze(-1)
        variance = variance.squeeze(-1)

        variance = F.softplus(variance) + 1e-6 #Positive constraint
        return mean, variance

def init_network(input_dim, output_dim):
    model = GaussianMultiLayerPerceptron(input_dim, output_dim)

    # 2. 对模型做Kaiming随机初始化（核心：每次调用生成不同的随机权重）
    def _kaiming_init(m):
        if isinstance(m, nn.Linear):
            # kaiming_normal_ 是随机初始化，每次调用生成不同权重
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    # 3. 应用初始化
    model.apply(_kaiming_init)

    return model 
    

# train one network
def train_one_network(model, train_loader, epochs=40, lr=0.1):
    """
    训练单个网络
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        epochs: 训练轮数
        lr: 学习率
    Returns:
        model: 训练好的模型
        train_losses: 训练损失历史
        val_losses: 验证损失历史（如果有验证集）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    
    
    for epoch in range(epochs):
        # train
        model.train()
        epoch_train_loss = 0.0
        epoch_train_RMSE = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            mean, variance = model(batch_x)
            loss = gaussian_nll(mean, variance, batch_y)
            #RMSE = torch.sqrt(torch.mean((mean - batch_y) ** 2))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            #epoch_train_RMSE += RMSE
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        #train_RMSE.append(RMSE)
        #print(f'avg_train_RMSE_every_epoch is {epoch_train_RMSE/num_batches}')       
    
    return model, train_losses



def train_ensemble_models(n_networks=5, input_dim=None, train_loader= None, epochs=40):
    if input_dim is None:
        # 如果没指定input_dim，从train_loader获取
        for batch_x, _ in train_loader:
            input_dim = batch_x.shape[1]
            break
    
    ensemble_models = []
    training_history = []
    for i in range(n_networks):
        print(f"\n======= training NO. {i+1}/{n_networks} network =======")
        
        # initialize
        model = init_network(input_dim, output_dim=2)
        
        # train
        trained_model, train_losses = train_one_network(
            model, train_loader, epochs
        )
        
        ensemble_models.append(trained_model)
        training_history.append({
            'network_id': i,
            'train_losses': train_losses,
        })
        
        print(f"No.{i+1} network is trained, the training loss: {train_losses[-1]:.4f}")
    
    return ensemble_models, training_history

# 使用示例
# train_loader = splits[0]['train']
# test_loader = splits[0]['test']

# for batch_x, batch_y in train_loader:
#     print(f"Batch shape: {batch_x.shape}, {batch_y.shape}")
#     break




def evaluate_ensemble_with_details(ensemble_models, test_loader, scaler_y=None, y_std=None):
    """
    更详细的集成模型评估，包括每个模型的单独性能和反标准化
    
    Args:
        ensemble_models: 训练好的模型列表
        test_loader: 测试数据加载器
        scaler_y: 用于反标准化目标值的scaler（可选）
    
    Returns:
        results: 包含各种评估指标的字典
    """
    device = next(ensemble_models[0].parameters()).device
    #criterion = torch.nn.MSELoss()
    
    all_targets = []
    all_ensemble_mean = []
    all_ensemble_var = []
    #individual_preds = [[] for _ in range(len(ensemble_models))]
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_means = []
            batch_variances = []
            
            # 记录真实值
            all_targets.append(batch_y.cpu())
            
            # 每个模型单独预测
            for i, model in enumerate(ensemble_models):
                mean, variance = model(batch_x)
                batch_means.append(mean)
                batch_variances.append(variance)
            
            # 集成预测（取平均）
            #batch_ensemble_pred = torch.stack([p[-1] for p in individual_preds]).mean(dim=0)
            #all_ensemble_preds.append(batch_ensemble_pred)
            # 计算当前batch的集成预测（5个模型的平均）
            batch_ensemble_mean = torch.stack(batch_means).mean(dim=0)
            batch_ensemble_var = torch.stack(batch_variances)
            batch_ensemble_var = (batch_ensemble_var + torch.stack(batch_means)**2).mean(dim=0) - batch_ensemble_mean**2
            
            # 收集当前batch的集成预测
            all_ensemble_mean.append(batch_ensemble_mean.cpu())
            all_ensemble_var.append(batch_ensemble_var.cpu())

    
    # 拼接所有batch
    all_targets = torch.cat(all_targets, dim=0)
    all_ensemble_mean = torch.cat(all_ensemble_mean, dim=0)  # 变成一个张量
    all_ensemble_var = torch.cat(all_ensemble_var, dim=0)
    
    #individual_preds = [torch.cat(preds, dim=0) for preds in individual_preds]
    
    # # 1. 计算集成模型的损失
    # ensemble_loss = criterion(all_ensemble_preds, all_targets).item()
    
    # # 2. 计算每个单独模型的损失
    # individual_losses = []
    # for preds in individual_preds:
    #     loss = criterion(preds, all_targets).item()
    #     individual_losses.append(loss)
    
    # 3. 需要反标准化到原始尺度
    if scaler_y is not None:
        all_targets_orig = scaler_y.inverse_transform(all_targets.numpy().reshape(-1, 1)).ravel()
        all_ensemble_mean_orig = scaler_y.inverse_transform(all_ensemble_mean.cpu().numpy().reshape(-1, 1)).ravel()
        #all_ensemble_var_orig = scaler_y.inverse_transform(all_ensemble_var.cpu().numpy().reshape(-1, 1)).ravel()
        y_std = scaler_y.scale_[0]
        all_ensemble_var_orig = all_ensemble_var.cpu().numpy() * (y_std ** 2)  # 方差乘以标准差的平方
        #all_ensemble_var_orig = torch.tensor(all_ensemble_var_orig, dtype=torch.float32)
        #print(f'all_targets_orig is {all_targets_orig}')
        #print(f'all_ensemble_preds_orig is {all_ensemble_preds_orig}')
        
        
        # 计算原始尺度的RMSE
        ensemble_rmse_orig = np.sqrt(np.mean((all_ensemble_mean_orig - all_targets_orig) ** 2))
        # 计算原始尺度的NLL
        ensemble_nll_orig = gaussian_nll(all_ensemble_mean_orig, all_ensemble_var_orig, all_targets_orig)
    else:
        ensemble_rmse_orig = None
        all_targets_orig = None
        #all_ensemble_preds_orig = None
    
    results = {
        # 'ensemble_loss': ensemble_loss,
        # 'individual_losses': individual_losses,
        # 'mean_individual_loss': np.mean(individual_losses),
        # 'std_individual_loss': np.std(individual_losses),
        'ensemble_rmse_original': ensemble_rmse_orig,
        'ensemble_nll_original': ensemble_nll_orig,
        'all_targets': all_targets.numpy(),
        'all_ensemble_mean': all_ensemble_mean.numpy(),
        'all_targets_original': all_targets_orig,
        'all_ensemble_mean_original': all_ensemble_mean_orig,
        'all_ensemble_var_original': all_ensemble_var_orig

    }
    
    return results



def main(datasets='Boston', n_splits=None):
    
    # # boston = fetch_openml(data_id=531, as_frame=True, parser='pandas')
    # # X = boston.data.values.astype(np.float32)
    # # y = boston.target.values.astype(np.float32)
    
     
    # # concrete_compressive_strength = fetch_ucirepo(id=165) 
    # # X = concrete_compressive_strength.data.features.values.astype(np.float32)
    # # y = concrete_compressive_strength.data.targets.values.astype(np.float32)

    # # combined_cycle_power_plant = fetch_ucirepo(id=294) 
    # # X = combined_cycle_power_plant.data.features.values.astype(np.float32) 
    # # y = combined_cycle_power_plant.data.targets.values.astype(np.float32) 

    # # df = pd.read_csv('winequality-red.csv', sep=';')
    # # X = df.iloc[:, :-1].values.astype(np.float32)  # 所有列除了最后一列作为特征
    # # y = df.iloc[:, -1].values.astype(np.float32)   # 最后一列作为目标（质量分数）

    # df = pd.read_csv('yacht_hydrodynamics.data', sep=r'\s+', header=None)
    # X = df.iloc[:, :-1].values.astype(np.float32)  # 前6列是特征
    # y = df.iloc[:, -1].values.astype(np.float32)   # 最后一列是目标

    # # df = pd.read_csv('protein.csv', sep=',')
    # # X = df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']].values.astype(np.float32)
    # # y = df['RMSD'].values.astype(np.float32)

    # # df = pd.read_excel('ENB2012_data.xlsx')
    # # feature_columns = df.columns[:8].tolist()  # X1-X8
    # # target_columns = df.columns[8:].tolist()   # Y1
    # # # 转换为numpy数组
    # # X = df[feature_columns].values.astype(np.float32)
    # # y = df[target_columns].values.astype(np.float32)  # 两个目标变量



    """处理指定的数据集"""
    
    # 根据数据集名称加载数据
    if datasets == 'Boston':
        print(f"正在处理 Boston 数据集")
        data = fetch_openml(data_id=531, as_frame=True, parser='pandas')
        X = data.data.values.astype(np.float32)
        y = data.target.values.astype(np.float32)
        n_splits = n_splits or 20
        
    elif datasets == 'Concrete':
        print(f"正在处理 Concrete 数据集")
        data = fetch_ucirepo(id=165)
        X = data.data.features.values.astype(np.float32)
        y = data.data.targets.values.astype(np.float32)
        n_splits = n_splits or 20
        
    elif datasets == 'PowerPlant':
        print(f"正在处理 PowerPlant 数据集")
        data = fetch_ucirepo(id=294)
        X = data.data.features.values.astype(np.float32)
        y = data.data.targets.values.astype(np.float32)
        n_splits = n_splits or 20
        
    elif datasets == 'WineQuality':
        print(f"正在处理 WineQuality 数据集")
        df = pd.read_csv('winequality-red.csv', sep=';')
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)
        n_splits = n_splits or 20
        
    elif datasets == 'Yacht':
        print(f"正在处理 Yacht 数据集")
        df = pd.read_csv('yacht_hydrodynamics.data', sep=r'\s+', header=None)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)
        n_splits = n_splits or 20
        
    elif datasets == 'Protein':
        print(f"正在处理 Protein 数据集")
        df = pd.read_csv('protein.csv', sep=',')
        X = df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']].values.astype(np.float32)
        y = df['RMSD'].values.astype(np.float32)
        n_splits = n_splits or 5  # Protein 默认5折
        
    elif datasets == 'ENB':
        print(f"正在处理 ENB 数据集")
        df = pd.read_excel('ENB2012_data.xlsx')
        feature_columns = df.columns[:8].tolist()
        target_columns = df.columns[8:].tolist()
        X = df[feature_columns].values.astype(np.float32)
        y = df[target_columns].values.astype(np.float32)
        n_splits = n_splits or 20
        
    else:
        raise ValueError(f"未知数据集: {datasets}")

    # # 生成20个随机90/10划分
    splits = []
    np.random.seed(42)
    # n_splits = 20

    for i in range(n_splits):
        # 随机划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42 + i
        )
        
        # 标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        y_std = scaler_y.scale_[0]  # 保存标准差供后续使用
        
        # 转换为tensor并创建dataloader
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        splits.append({
            'train': train_loader,
            'test': test_loader,
            'scaler_y': scaler_y  # 保存scaler用于反标准化
        })
    
    final_results = []

    for i in range(n_splits):
        ensemble_models, training_history = train_ensemble_models(n_networks=5, input_dim=None, train_loader= splits[i]['train'], epochs=40)
        
        # 详细评估 - 包含更多信息
        results = evaluate_ensemble_with_details(
            ensemble_models,
            test_loader=splits[i]['test'],
            scaler_y=splits[i]['scaler_y'],
            y_std=y_std
        )
        final_results.append(results)
    
        print("\n=== 集成模型评估结果 ===")
        print(f'the RMSE loss of this fold over 5 networks is {results['ensemble_rmse_original']}')
        print(f'the nll loss of this fold over 5 networks is {results['ensemble_nll_original']}')

    rmse_folds = []
    nll_folds = []
    for i in range(n_splits):
        rmse_folds.append(final_results[i]['ensemble_rmse_original'])
        nll_folds.append(final_results[i]['ensemble_nll_original'])

    print(f'the overall mean of RMSE is {np.mean(rmse_folds)}')
    print(f'the overall std error of RMSE is {np.sqrt(np.var(rmse_folds))}')
    print(f'the overall mean of NLL is {np.mean(nll_folds)}')
    print(f'the std error of NLL is {np.sqrt(np.var(nll_folds))}')


if __name__ == "__main__":
    main(datasets='Boston')