# 文件名: run_evaluation.py
# 描述: 专门用于模型评估和结果报告

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix

def evaluate_model_performance(X, y, selected_indices):
    """
    (修改版)
    使用5折交叉验证和L2逻辑回归，评估所选特征的
    Accuracy, AUC, ...
    同时返回 *训练集* 的指标以诊断过拟合/欠拟合。
    """
    
    if len(selected_indices) == 0:
        # ... (这部分不变) ...
        return {"Accuracy": 0, "AUC": 0, "Sensitivity": 0, "Specificity": 0, "F1-Macro": 0, "Train_AUC": 0, "Train_Accuracy": 0}
        
    X_subset = X[:, selected_indices]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # [修改] 增加 "train_" 相关的指标
    metrics = {
        "acc": [], "auc": [], "sens": [], "spec": [], "f1": [],
        "acc_train": [], "auc_train": []  # <--- 新增
    }

    valid_folds = 0 

    for train_idx, test_idx in skf.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if len(np.unique(y_test)) < 2:
            continue
        
        valid_folds += 1
        
        model = LogisticRegressionCV(
            Cs=10, cv=3, penalty='l2', solver='liblinear', 
            random_state=42, max_iter=1000,
            # class_weight='balanced'  # <--- 保持这个（来自上次对话的修改）
        )
        model.fit(X_train, y_train)
        
        # --- 验证集 (Test) 指标 (原逻辑) ---
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics["acc"].append(accuracy_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred, average='macro'))
        metrics["auc"].append(roc_auc_score(y_test, y_proba))
        metrics["sens"].append(recall_score(y_test, y_pred, pos_label=1, average='binary'))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP)
        metrics["spec"].append(specificity)

        # --- [!!! 核心新增 !!!] ---
        # --- 训练集 (Train) 指标 ---
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        
        metrics["acc_train"].append(accuracy_score(y_train, y_pred_train))
        metrics["auc_train"].append(roc_auc_score(y_train, y_proba_train))
        # --- [新增结束] ---

    if valid_folds == 0:
        # ... (这部分不变) ...
        return {"Accuracy": 0, "AUC": 0, "Sensitivity": 0, "Specificity": 0, "F1-Macro": 0, "Train_AUC": 0, "Train_Accuracy": 0}

    # [修改] 计算平均值时加入 "train_" 指标
    avg_results = {
        "Accuracy": np.mean(metrics["acc"]),
        "AUC": np.mean(metrics["auc"]),
        "Sensitivity": np.mean(metrics["sens"]),
        "Specificity": np.mean(metrics["spec"]),
        "F1-Macro": np.mean(metrics["f1"]),
        "Train_Accuracy": np.mean(metrics["acc_train"]), # <--- 新增
        "Train_AUC": np.mean(metrics["auc_train"])       # <--- 新增
    }
    
    return avg_results

def print_summary_table(all_results, all_selected_indices, execution_times=None):
    """
    (修改版)
    打印最终的对比表格，加入训练集指标和运行时间。
    """
    if execution_times is None:
        execution_times = {}

    print("\n" + "#"*90)
    print("### 最终实验对比总结 ###")
    print("#"*90)
    
    # [修改] 增加 Time(s) 列
    header = f"{'Method':<12} | {'K':<4} | {'Time(s)':<8} | {'AUC':<10} | {'Train_AUC':<10} | {'Accuracy':<10} | {'Train_Acc':<10} | {'Sensitivity':<11} | {'Specificity':<11} | {'F1-Macro':<10}"
    print(header)
    print("-" * len(header))
    
    sorted_methods = sorted(all_results.items(), key=lambda item: item[1].get('Accuracy', 0), reverse=True)
    
    for method_name, metrics in sorted_methods:
        k = len(all_selected_indices.get(method_name, []))
        time_taken = execution_times.get(method_name, 0.0) # 获取时间
        
        # [修改] 打印时间列
        print(f"{method_name:<12} | {k:<4} | "
              f"{time_taken:<8.2f} | "                   # <--- 新增：保留2位小数的时间
              f"{metrics.get('AUC', 0):<10.4f} | "
              f"{metrics.get('Train_AUC', 0):<10.4f} | "       
              f"{metrics.get('Accuracy', 0):<10.4f} | "
              f"{metrics.get('Train_Accuracy', 0):<10.4f} | "  
              f"{metrics.get('Sensitivity', 0):<11.4f} | "
              f"{metrics.get('Specificity', 0):<11.4f} | "
              f"{metrics.get('F1-Macro', 0):<10.4f}")
              
    print("#"*90)
