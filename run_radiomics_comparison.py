# 文件名: run_radiomics_comparison.py
# 描述: 主脚本，用于数据清理、特征选择和调用评估
# 依赖: run_evaluation.py, CDGAFS.py, fisher_score.py, pymrmr

import pandas as pd
import numpy as np
import warnings
import time
import os
import sys

# 机器学习和 mRMR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE  # <--- 新增 RFE
import pymrmr

# --- 导入你项目中的核心功能 ---
from CDGAFS import cdgafs_feature_selection
from fisher_score import compute_fisher_score
from run_evaluation import evaluate_model_performance, print_summary_table

# 忽略所有警告
warnings.filterwarnings('ignore')

# ===================================================================
# 1. 数据加载与清理模块
# ===================================================================
def load_and_clean_radiomics(csv_path, label_col_name):
    print(f"--- 步骤 1: 加载并清理数据: {csv_path} ---")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"!!! 致命错误: 找不到文件 '{csv_path}'。")
        sys.exit(1)

    if label_col_name not in data.columns:
        print(f"!!! 致命错误: 在CSV中找不到标签列 '{label_col_name}'。")
        sys.exit(1)

    y_raw = data[label_col_name].values
    unique_labels = np.unique(y_raw)
    if len(unique_labels) == 2:
        class_0_label = np.min(unique_labels)
        y = np.where(y_raw == class_0_label, 0, 1)
        print(f"    - 标签已转换为 0 和 1。")
    else:
        print(f"!!! 致命错误: 标签列必须包含2个唯一的类别。找到: {unique_labels}")
        sys.exit(1)
    
    id_col = 'ID' if 'ID' in data.columns else 'Patient_ID'
    diag_cols = [col for col in data.columns if col.startswith('diagnostics_')]
    cols_to_drop = [id_col, label_col_name] + diag_cols
    
    X_df = data.drop(columns=cols_to_drop, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number])
    feature_names = X_df.columns.tolist()
    
    # 填充 NaN
    if X_df.isna().any().any():
        imputer = SimpleImputer(strategy='mean')
        X_unscaled = imputer.fit_transform(X_df)
    else:
        X_unscaled = X_df.values
    
    # 移除低方差特征
    stds = np.std(X_unscaled, axis=0)
    variance_threshold = 1e-6 
    good_indices = np.where(stds > variance_threshold)[0]
    
    X_unscaled = X_unscaled[:, good_indices]
    feature_names = [feature_names[i] for i in good_indices]
    
    print(f"    - 清理后剩余特征数: {len(feature_names)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)
    
    X_df_filled = pd.DataFrame(X_unscaled, columns=feature_names) 
    
    return X_scaled, X_df_filled, y, feature_names


# ===================================================================
# 2. 特征选择器模块
# ===================================================================

def select_features_cdafs(X, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA):
    print("\n--- 正在运行: CDGAFS (社区检测 + 遗传算法) ---")
    start_time = time.time()

    # 调用 GA 流程
    (selected_indices, 
     _, _, _, _) = cdgafs_feature_selection(
        X=X, 
        y=y, 
        gene_list=feature_names, 
        theta=THETA, 
        omega=GA_OMEGA, 
        population_size=GA_POPULATION_SIZE,
        w_bio_boost=0.0,
        pre_filter_top_n=None,
        graph_type='pearson_only'
    )
        
    elapsed = time.time() - start_time
    print(f"--- CDGAFS GA 阶段完成。耗时: {elapsed:.2f} 秒。GA 选出 {len(selected_indices)} 个特征。---")
        
    # --- [核心改进] 智能剪枝 ---
    # 如果 GA 选出的特征多于 K_FEATURES，使用 RFE (Wrapper) 而不是 Fisher Score (Filter) 进行剪枝。
    # Fisher Score 是单变量的，会破坏 GA 找到的组合优势；RFE 能保留组合优势。
    if len(selected_indices) > K_FEATURES:
         print(f"    - [优化] 使用 RFE (递归特征消除) 从 {len(selected_indices)} 精选到 {K_FEATURES} 个...")
         
         # 使用带 balanced 权重的 LR 进行 RFE
         estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
         selector = RFE(estimator, n_features_to_select=K_FEATURES, step=1)
         
         # 注意：这里只传入 GA 选中的那些列
         X_ga_selected = X[:, selected_indices]
         selector.fit(X_ga_selected, y)
         
         # 获取 RFE 选中的掩码
         rfe_support = selector.support_
         
         # 映射回原始索引
         # selected_indices 是列表，先转 array
         selected_indices = np.array(selected_indices)[rfe_support]
         
         print(f"    - RFE 剪枝完成。")
             
    elif len(selected_indices) == 0:
        print("    - !!! 警告: CDGAFS 未选出任何特征。")
        return []

    return selected_indices

def select_features_mrmr(X_df, y, K_FEATURES):
    print("\n--- 正在运行: mRMR ---")
    start_time = time.time()
    
    X_df_discrete = X_df.copy()
    n_bins = 10
    
    for col in X_df_discrete.columns:
        try:
            X_df_discrete[col] = pd.qcut(X_df_discrete[col], q=n_bins, labels=False, duplicates='drop')
        except Exception:
            X_df_discrete[col] = 0
            
    X_df_discrete['label'] = y
    
    try:
        selected_feature_names = pymrmr.mRMR(X_df_discrete, 'MIQ', K_FEATURES)
        if 'label' in selected_feature_names:
            selected_feature_names.remove('label')

        name_to_index_map = {name: i for i, name in enumerate(X_df.columns)}
        selected_indices = [name_to_index_map[name] for name in selected_feature_names]
        
        print(f"--- mRMR 完成。选出 {len(selected_indices)} 个特征。---")
        return selected_indices
    except Exception as e:
        print(f"!!! mRMR 运行失败: {e}")
        return []

def select_features_lasso(X, y, K_FEATURES):
    print("\n--- 正在运行: LASSO (L1) with Cross-Validation ---")
    # 使用 LogisticRegressionCV 自动寻找最佳 C
    model = LogisticRegressionCV(
        cv=5, 
        penalty='l1', 
        solver='liblinear', 
        class_weight='balanced', # 同样加上 balanced 以便公平对比
        random_state=42, 
        max_iter=3000,
        scoring='roc_auc'
    )
    model.fit(X, y)
    
    coefficients = model.coef_[0]
    non_zero_indices = np.where(np.abs(coefficients) > 1e-6)[0]
        
    if len(non_zero_indices) == 0:
        return []
    
    sorted_indices = sorted(non_zero_indices, key=lambda i: np.abs(coefficients[i]), reverse=True)
    selected_indices = sorted_indices[:K_FEATURES] 
    
    print(f"--- LASSO-CV 完成。最佳 C: {model.C_[0]:.4f}。选出 {len(selected_indices)} 个特征。")
    return selected_indices

# ===================================================================
# 3. 主程序
# ===================================================================
def main():
    # --- 参数设置 ---
    CSV_PATH = '/data/qh_20T_share_file/lct/CT67/ovarian_features_with_label.csv'
    LABEL_COLUMN = 'label'
    
    # 建议：如果想比 LASSO 更准，尝试保留更多特征，或者让 K=8 与 LASSO 保持一致进行公平对比
    # 这里设定 K=8，看 CDGAFS 能否在同等数量下打败 LASSO
    K_FEATURES = 8 
    
    GA_POPULATION_SIZE = 100 
    GA_OMEGA = 0.15 # 稍微调大一点，给 RFE 更多选择空间
    THETA = 0.9

    print("#"*70)
    print("### 开始运行放射组学特征选择与评估对比实验 (优化版) ###")
    print("#"*70)
    
    X_scaled, X_df_filled, y, feature_names = load_and_clean_radiomics(CSV_PATH, LABEL_COLUMN)
    if X_scaled is None: return
        
    all_selected_indices = {}
    
    # 1. CDGAFS (优化版)
    all_selected_indices['CDGAFS'] = select_features_cdafs(X_scaled, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA)
    
    # 2. mRMR
    all_selected_indices['mRMR'] = select_features_mrmr(X_df_filled.copy(), y, K_FEATURES)

    # 3. LASSO
    all_selected_indices['LASSO'] = select_features_lasso(X_scaled, y, K_FEATURES)    
    
    # 评估
    print("\n" + "#"*70)
    print("### 开始评估性能 (L2 LR, 5-CV) ###")
    print("#"*70)
    
    all_results = {}
    for method_name, indices in all_selected_indices.items():
        print(f"\n--- (评估) {method_name} ---")
        if len(indices) == 0: continue
        results = evaluate_model_performance(X_scaled, y, indices)
        all_results[method_name] = results
        for metric, value in results.items():
            print(f"        - {metric:<12}: {value:.4f}")

    print_summary_table(all_results, all_selected_indices)

if __name__ == "__main__":
    main()