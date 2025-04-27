from sklearn.preprocessing import LabelEncoder
from joblib import load
import torch
import numpy as np
import pandas as pd

import os
import sys
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from modelTrainer.method_svm import extract_features, MODEL_NAME_SVM
from modelTrainer.methodP_cnn_lstm import CNNLSTM, MODEL_NAME_CNNLSTM

class SVMActionPredictor:
    def __init__(self):
        """初始化时加载模型和标签编码器"""
        self.pipeline = load(f'./models/{MODEL_NAME_SVM}_pipeline.joblib')
        self.le = load(f'./models/{MODEL_NAME_SVM}_label_encoder.joblib')
    
    def predict(self, data: pd.DataFrame)->(str,float):
        raw_features = extract_features(data)
        features = np.array(raw_features).reshape(1, -1)
        probas = self.pipeline.predict_proba(features)
        pred_idx = self.pipeline.predict(features)[0]
        confidence = np.max(probas)
        return (self.le.inverse_transform([pred_idx])[0], float(confidence))

class CnnLstmActionPredictor:
    def __init__(self):
        """初始化时加载模型和标签编码器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载标签编码器
        self.le = load(f'./models/{MODEL_NAME_CNNLSTM}_label_encoder.joblib')
        # 初始化并加载模型
        self.model = CNNLSTM(num_classes=len(self.le.classes_)).to(self.device)
        self.model.load_state_dict(torch.load(
            f'./models/{MODEL_NAME_CNNLSTM}.pth', 
            map_location=self.device, 
            weights_only=True
        ))
        self.model.eval()

    def predict(self, df: pd.DataFrame)->(str,float):
        """执行预测的类方法"""
        # 预处理输入数据
        input_tensor = torch.FloatTensor(df.iloc[:, 1:].values).unsqueeze(0)
        
        # 执行预测
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        return (self.le.inverse_transform(predicted.cpu().numpy())[0], float(confidence.item()))

# 使用示例
if __name__ == "__main__":
    import os
    
    # 初始化预测器（只加载一次模型）
    sPredictor = SVMActionPredictor()
    clPredictor = CnnLstmActionPredictor()
    
    test_dir = "./trainData_badQuality"
    #test_dir = "./trainData"

    for action_name in os.listdir(test_dir):
        action_dir = os.path.join(test_dir, action_name)
        print(f"Testing action: {action_name}")
        
        for sample_file in os.listdir(action_dir):
            if sample_file.endswith('.csv'):
                file_path = os.path.join(action_dir, sample_file)
                pdData=pd.read_csv(file_path)
                pred_action = None
                cfd = 0.0
                #SVM
                pred_action, cfd = sPredictor.predict(pdData)
                print(
                    "\t "+f"file: {sample_file},"+f"  \tSVM:\t\t{cfd}\t"+
                    f"{'Good' if pred_action==action_name else 'BadPredict:    '+ pred_action}"
                )
                #CNN-LSTM
                pred_action, cfd = clPredictor.predict(pdData)
                print(
                    "\t "+f"file: {sample_file},"+f"  \tCNN-LSTM:\t{cfd}\t"+
                    f"{'Good' if pred_action==action_name else 'BadPredict:   '+ pred_action}"
                )