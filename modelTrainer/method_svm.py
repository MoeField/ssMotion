import os
import gc
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

RNG = 42  # 随机种子
MODEL_NAME_SVM = "svm_iac2"

# 1. 数据加载与预处理
def load_data(root_dir):
    """
    加载文件夹结构数据：
    trainData/
        ├── action1/
        │   ├── sample1.csv
        │   └── ...
        ├── action2/
        └── ...
    """
    features = []
    labels = []
    
    # 遍历每个动作文件夹
    for action_name in os.listdir(root_dir):
        action_dir = os.path.join(root_dir, action_name)
        
        # 遍历每个样本文件
        for sample_file in os.listdir(action_dir):
            if sample_file.endswith('.csv'):
                file_path = os.path.join(action_dir, sample_file)
                
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 2. 特征工程（关键步骤）
                sample_features = extract_features(df)
                
                features.append(sample_features)
                labels.append(action_name)
    
    return np.array(features), np.array(labels)

def extract_features(df):
    """从80行时序数据中提取特征"""
    features = []
    
    # 对每个传感器通道提取统计特征
    sensors = [
        'accX',      'accY',     'accZ', 
        'gyroX',     'gyroY',    'gyroZ',
        'angleX',    'angleY',   'height',
    ]
    
    for col in sensors:
        #阈值处理
        if col in ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']:
            df[col] = df[col].apply(lambda x: 0 if abs(x) < 5 else x)# 阈值处理，将绝对值小于0.1的设为0

        # 时域特征
        ts = df[col].values
        features += [
            np.mean(ts),        # 平均值
            np.std(ts),         # 标准差
            #np.max(ts)-np.min(ts),  # 极差
            np.percentile(ts, 75),  # 75%分位数
            #zero_crossing_rate(ts)  # 过零率
        ]
        
        # 频域特征
        fft_vals = np.fft.rfft(ts)
        fft_abs = np.abs(fft_vals)
        features += [
            np.max(fft_abs),    # 最大频率分量幅值
            np.mean(fft_abs[1:5])  # 低频能量（排除直流分量）
        ]
    
    # 添加运动学组合特征
    acc_norm = np.linalg.norm(df[['accX', 'accY', 'accZ']], axis=1)
    features += [
        np.mean(acc_norm), 
        np.std(acc_norm),
        np.max(acc_norm)
    ]
    
    # 增加时间窗口滑动特征
    window_size = 10
    for col in sensors:
        rolling_mean = df[col].rolling(window=window_size).mean().dropna()
        features += [
            np.mean(rolling_mean),
            np.std(rolling_mean)
        ]
    
    # 增加差分特征
    for col in sensors:
        diffs = np.diff(df[col].values)
        features += [
            np.mean(diffs),
            np.std(diffs)
        ]
    
    # 添加新数据特有的特征处理
    if 'height' in df.columns:
        # 新增高度变化率特征
        height_diff = np.diff(df['height'].values)
        features += [
            np.mean(height_diff),
            np.std(height_diff)
        ]
    
    # 添加运动模式检测
    acc_std = np.std(df[['accX', 'accY', 'accZ']], axis=0)
    features += list(acc_std)
    
    # 移除低频能量比等冗余特征
    # 增加特征相关性阈值（相关系数>0.7的特征自动合并）
    corr_threshold = 0.7
    sensor_corr = df[sensors].corr().abs()
    upper = sensor_corr.where(np.triu(np.ones(sensor_corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    
    return features

def zero_crossing_rate(signal):
    """计算过零率"""
    return ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)

###################################################################
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用非GUI后端
# 新增字体配置
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置黑体 
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.model_selection import LearningCurveDisplay
# 绘制学习曲线
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1):
    plt.figure(figsize=(8, 6))
    LearningCurveDisplay.from_estimator(
        estimator,
        X,y,cv=cv,
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./imgs/学习曲线{title}.png", dpi=300, transparent=True)

# 在模型评估阶段添加
from sklearn.calibration import calibration_curve

# 修改校准曲线绘制逻辑
def plot_calibration_curve(y_true, y_prob, classes):
    plt.figure(figsize=(12, 8))
    for i in range(len(classes)):
        prob_true, prob_pred = calibration_curve(
            (y_true == i).astype(int), 
            y_prob[:, i],
            n_bins=10
        )
        plt.plot(prob_pred, prob_true, marker='o', label=classes[i])
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Calibration Curve (OvR)")
    plt.savefig("./imgs/calibration_curve.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import classification_report

    import numpy as np
    
    # 步骤1：加载数据
    X, y = load_data("./trainData")
    #UniQueTypes
    print("Unique labels:", np.unique(y))
    
    #print("Data loaded successfully.")

    # 步骤2：标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    #print("Labels encoded successfully.")

    # 步骤3：创建处理管道
    # 修改处理管道
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # 保留95%方差
        ('svm', SVC(kernel='rbf', probability=True))
    ])
    #print("Pipeline created successfully.")

    # 步骤4：训练与参数搜索
    # 7. 训练与评估
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RNG    #
    )
    #print("Data split into training and test sets.")

    # 首先进行GridSearchCV找最优参数（保持原代码）
    # 修改参数网格增加正则化选项
    """
    param_grid = {
        #'svm__C': [0.1, 1, 10],
        #'svm__gamma': ['scale', 'auto'],
        #'svm__kernel': ['rbf']
        'svm__C': [0.01, 0.1, 1, 10],  # 扩大搜索范围
        'svm__gamma': ['scale', 0.1, 0.01, 0.001],  # 更精细的gamma值
        'svm__class_weight': ['balanced']  # 处理类别不平衡
    }
    #print("Parameter grid defined.")
    """
    param_grid = {
    'svm__C': np.logspace(-3, 1, 5),  # 缩小搜索范围 [0.001, 0.01, 0.1, 1, 10]
    'svm__gamma': np.concatenate((np.logspace(-3, 0, 4), [0.5])),  # 减少离散值
    'svm__class_weight': ['balanced'],
    'svm__kernel': ['rbf'], # 'poly'],  # 增加核函数选择
    #'svm__max_iter': [10000]  # 确保收敛
    'svm__shrinking': [True, False]  # 新增正则化选项
}

    grid_search = GridSearchCV(
        pipeline,       # 使用完整的Pipeline
        param_grid,     # 搜索的参数网格
        cv=5,           # 5折交叉验证
        n_jobs=-1,      # 使用所有可用CPU核心
        verbose=1       # 输出详细信息
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    gc.collect()

    # 阶段一：绘制最佳模型的学习曲线
    plot_learning_curve(best_model, "SVM LearnCurve", X_train, y_train, cv=5)
    # 阶段二：输出模型在测试集上的表现
    print("Best model score on test set: {:.3f}".format(best_model.score(X_test, y_test)))
    # 输出最终分类报告
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    # 在模型预测后调用时传入类别标签
    y_prob = best_model.predict_proba(X_test)  # 获取概率预测
    plot_calibration_curve(y_test, y_prob, le.classes_)

    # 步骤5：保存模型
    from joblib import dump
    dump(best_model, f'./models/{MODEL_NAME_SVM}_pipeline.joblib')  # 保存完整的Pipeline
    dump(le, f'./models/{MODEL_NAME_SVM}_label_encoder.joblib')  # 单独保存标签编码器
    print("Models saved successfully.")
    # 新增嵌套交叉验证
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold
    nested_scores = cross_val_score(
        grid_search, X_train, y_train, 
        cv=StratifiedKFold(n_splits=3),
        scoring='balanced_accuracy')
    print(f"Nested CV Score: {np.mean(nested_scores):.3f} ± {np.std(nested_scores):.3f}")

