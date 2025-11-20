# 文件名: run_radiomics_comparison.py
# 描述: 主脚本 (已整理：统一计时逻辑，移除 try-except)

import pandas as pd
import numpy as np
import warnings
import time
import os
import sys
import radMLBench

# 机器学习和 mRMR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE 
import pymrmr

# --- 导入核心功能 ---
from CDGAFS import cdgafs_feature_selection
from fisher_score import compute_fisher_score
from run_evaluation import evaluate_model_performance, print_summary_table

# 忽略所有警告
warnings.filterwarnings('ignore')

# ===================================================================
# 1. 数据加载与清理模块 (保持不变)
# ===================================================================
def clean_radiomics_df(data, label_col_name):
    print(f"    - [处理中] 原始数据形状: {data.shape}")

    if label_col_name not in data.columns:
        print(f"!!! 致命错误: 找不到标签列 '{label_col_name}'。")
        sys.exit(1)

    y_raw = data[label_col_name].values
    unique_labels = np.unique(y_raw)
    if len(unique_labels) == 2:
        class_0_label = np.min(unique_labels)
        y = np.where(y_raw == class_0_label, 0, 1)
    else:
        print(f"!!! 致命错误: 标签列必须包含2个唯一的类别。找到: {unique_labels}")
        sys.exit(1)
    
    id_cols = [col for col in data.columns if 'ID' in col or 'id' in col] 
    diag_cols = [col for col in data.columns if col.startswith('diagnostics_')]
    cols_to_drop = id_cols + [label_col_name] + diag_cols
    
    X_df = data.drop(columns=cols_to_drop, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number])
    feature_names = X_df.columns.tolist()
    
    if X_df.isna().any().any():
        imputer = SimpleImputer(strategy='mean')
        X_unscaled = imputer.fit_transform(X_df)
    else:
        X_unscaled = X_df.values
    
    stds = np.std(X_unscaled, axis=0)
    variance_threshold = 1e-6 
    good_indices = np.where(stds > variance_threshold)[0]
    
    if len(good_indices) == 0:
        print("!!! 错误: 所有特征方差均为0，无法继续。")
        sys.exit(1)

    X_unscaled = X_unscaled[:, good_indices]
    feature_names = [feature_names[i] for i in good_indices]
    
    print(f"    - 清理后剩余特征数: {len(feature_names)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)
    X_df_filled = pd.DataFrame(X_scaled, columns=feature_names) 
    
    return X_scaled, X_df_filled, y, feature_names

def load_and_clean_radiomics(csv_path, label_col_name):
    # 【核心修改】pandas.read_csv 默认支持读取压缩文件 (.gz)，无需额外处理。
    # 只要文件路径正确，它就能自动解压和读取。
    print(f"--- 加载本地文件: {csv_path} ---")
    
    # 尝试加载数据，使用 compression='gzip' 是可选的，但可以增加健壮性
    # 假设标签列总是 'label' 或 'Target'。
    try:
        data = pd.read_csv(csv_path, compression='gzip' if csv_path.endswith('.gz') else 'infer')
    except Exception as e:
        print(f"!!! 致命错误: 读取文件 {csv_path} 失败: {e}")
        sys.exit(1)
        
    return clean_radiomics_df(data, label_col_name)

# ===================================================================
# 2. 特征选择器模块 
# ===================================================================

def select_features_cdafs(X, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, pruning_method='RFE'):
    print(f"\n--- 正在运行: CDGAFS (剪枝方法: {pruning_method}) ---")

    # 调用 GA 流程
    (selected_indices, _, _, _, _) = cdgafs_feature_selection(
        X=X, y=y, gene_list=feature_names, theta=THETA, omega=GA_OMEGA, 
        population_size=GA_POPULATION_SIZE, w_bio_boost=0.0, 
        pre_filter_top_n=None, graph_type='pearson_only'
    )
    
    # [修改] 移除了 elapsed 打印，只保留逻辑信息
    print(f"    - GA 阶段结束，初步选中 {len(selected_indices)} 个特征。")
        
    if len(selected_indices) > K_FEATURES:
        if pruning_method == 'RFE':
            print(f"    - [优化] 使用 RFE 从 {len(selected_indices)} 精选到 {K_FEATURES} 个...")
            estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
            selector = RFE(estimator, n_features_to_select=K_FEATURES, step=1)
            
            X_ga_selected = X[:, selected_indices]
            selector.fit(X_ga_selected, y)
            
            rfe_support = selector.support_
            selected_indices = np.array(selected_indices)[rfe_support]
            print(f"    - RFE 剪枝完成。")

        elif pruning_method == 'FISHER':
            print(f"    - [优化] 使用 Fisher Score 从 {len(selected_indices)} 精选到 {K_FEATURES} 个...")
            X_ga_selected = X[:, selected_indices]
            scores_on_subset = compute_fisher_score(X_ga_selected, y)
            top_subset_indices = np.argsort(scores_on_subset)[-K_FEATURES:]
            selected_indices_array = np.array(selected_indices)
            selected_indices = selected_indices_array[top_subset_indices]
            print(f"    - Fisher Score 剪枝完成。")
            
    elif len(selected_indices) == 0:
        print("    - !!! 警告: CDGAFS 未选出任何特征。")
        return []

    return selected_indices

def select_features_mrmr(X_df, y, K_FEATURES):
    print("\n--- 正在运行: mRMR ---")
    
    X_df_discrete = X_df.copy()
    n_bins = 10
    
    for col in X_df_discrete.columns:
        X_df_discrete[col] = pd.qcut(X_df_discrete[col], q=n_bins, labels=False, duplicates='drop')
            
    X_df_discrete['label'] = y
    selected_feature_names = pymrmr.mRMR(X_df_discrete, 'MIQ', K_FEATURES)
    
    if 'label' in selected_feature_names:
        selected_feature_names.remove('label')

    name_to_index_map = {name: i for i, name in enumerate(X_df.columns)}
    selected_indices = [name_to_index_map[name] for name in selected_feature_names]
    
    print(f"    - mRMR 选中 {len(selected_indices)} 个特征。")
    return selected_indices


def select_features_lasso_fixed_k(X, y, K_FEATURES):
    print(f"\n--- 正在运行: LASSO-Fixed-K (目标 K={K_FEATURES}) ---")
    # 使用 LogisticRegressionCV 自动寻找最佳 C (L1惩罚)
    model = LogisticRegressionCV(
        cv=5, 
        penalty='l1', 
        solver='liblinear', 
        class_weight='balanced', 
        random_state=42, 
        max_iter=3000,
        scoring='roc_auc'
    )
    model.fit(X, y)
    
    # 使用找到的最佳 C 对应的系数
    coefficients = model.coef_[0]
    non_zero_indices = np.where(np.abs(coefficients) > 1e-6)[0]
    
    if len(non_zero_indices) == 0:
        print("!!! LASSO-Fixed-K 未选出任何特征。")
        return []
    
    # 排序并截断到 K_FEATURES
    sorted_indices = sorted(non_zero_indices, 
                        key=lambda i: np.abs(coefficients[i]), 
                        reverse=True)
    selected_indices = sorted_indices[:K_FEATURES] 
    
    print(f"    - LASSO-Fixed-K 完成。最佳 C: {model.C_[0]:.4f}。从 {len(non_zero_indices)} 个非零特征中截取前 {len(selected_indices)} 个。")
    return selected_indices


def select_features_lasso_cv(X, y):
    print("\n--- 正在运行: LASSO-CV (自动特征数量) ---")
    # 使用 LogisticRegressionCV 自动寻找最佳 C (L1惩罚)
    model = LogisticRegressionCV(
        cv=5, 
        penalty='l1', 
        solver='liblinear', 
        class_weight='balanced', 
        random_state=42, 
        max_iter=3000,
        scoring='roc_auc'
    )
    model.fit(X, y)
    
    coefficients = model.coef_[0]
    non_zero_indices = np.where(np.abs(coefficients) > 1e-6)[0]
    
    if len(non_zero_indices) == 0:
        print("!!! LASSO-CV 未选出任何特征。")
        return []
    
    print(f"    - LASSO-CV 完成。最佳 C: {model.C_[0]:.4f}。自动选出 {len(non_zero_indices)} 个特征。")
    # 返回所有非零特征的索引 (即自动确定的 K)
    return non_zero_indices.tolist()

def select_features_rfe_only(X, y, K_FEATURES):
    print("\n--- 正在运行: RFE-Only (递归特征消除) ---")

    estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42, max_iter=3000)
    selector = RFE(estimator, n_features_to_select=K_FEATURES, step=1)

    print(f"    - 正在从 {X.shape[1]} 个特征中精选 {K_FEATURES} 个...")
    selector.fit(X, y) 

    selected_indices = np.where(selector.support_)[0]

    print(f"    - RFE-Only 选中 {len(selected_indices)} 个特征。")
    return selected_indices

# ===================================================================
# 3. 统一执行与评估模块 (已整理：统一负责计时)
# ===================================================================
def run_analysis_on_dataset(X_scaled, X_df_filled, y, feature_names, 
                            K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, 
                            dataset_title):
    """
    通用函数：接收处理好的数据，运行所有特征选择方法并评估。
    """
    print(f"\n{'-'*30} 正在分析: {dataset_title} {'-'*30}")
    print(f"    - 样本数: {X_scaled.shape[0]}, 特征数: {X_scaled.shape[1]}")
    
    all_selected_indices = {}
    execution_times = {}  # 存储运行时间
    
    # --- 1. CDGAFS ---
    start_time = time.time() # [计时点]
    all_selected_indices['CDGAFS'] = select_features_cdafs(
        X_scaled, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, pruning_method='RFE'
    )
    elapsed = time.time() - start_time
    execution_times['CDGAFS'] = elapsed
    print(f"    >>> CDGAFS 执行完毕，耗时: {elapsed:.2f} 秒") # [统一打印]

    # --- 2. mRMR ---
    start_time = time.time()
    all_selected_indices['mRMR'] = select_features_mrmr(X_df_filled.copy(), y, K_FEATURES)
    elapsed = time.time() - start_time
    execution_times['mRMR'] = elapsed
    print(f"    >>> mRMR 执行完毕，耗时: {elapsed:.2f} 秒")

    # --- 3. LASSO ---
    start_time = time.time()
    # 调用新的固定K函数
    # all_selected_indices['LASSO-Fixed-K'] = select_features_lasso_fixed_k(X_scaled, y, K_FEATURES)
    # 调用新的自动K函数，不传入 K_FEATURES
    all_selected_indices['LASSO-CV'] = select_features_lasso_cv(X_scaled, y)
    elapsed = time.time() - start_time
    execution_times['LASSO'] = elapsed
    print(f"    >>> LASSO 执行完毕，耗时: {elapsed:.2f} 秒")
    
    # # --- 4. RFE-Only ---
    start_time = time.time()
    all_selected_indices['RFE-Only'] = select_features_rfe_only(X_scaled, y, K_FEATURES)
    elapsed = time.time() - start_time
    execution_times['RFE-Only'] = elapsed
    print(f"    >>> RFE-Only 执行完毕，耗时: {elapsed:.2f} 秒")

    # --- 5. 评估与制表 ---
    print(f"\n>>> {dataset_title} 的最终评估结果 <<<")
    all_results = {}
    for method_name, indices in all_selected_indices.items():
        if indices is None or len(indices) == 0: 
            continue
        
        results = evaluate_model_performance(X_scaled, y, indices)
        all_results[method_name] = results

    if all_results:
        # 记得传入 execution_times
        print_summary_table(all_results, all_selected_indices, execution_times)
    else:
        print(f"警告: {dataset_title} 没有产生任何有效结果。")

# ===================================================================
# 4. 主程序
# ===================================================================
def main():
    # 任务 1: 本地数据 (Ovarian Data)
    LOCAL_CSV_PATH = '/data/qh_20T_share_file/lct/CT67/ovarian_features_with_label.csv'
    LOCAL_LABEL_COL = 'label'
    
    # 【新增/修改】配置本地保存的 radMLBench 数据集路径和标签
    # 假设您的文件路径结构为：PUBLIC_DATASET_DIR / ds_name.gz
    PUBLIC_DATASET_DIR = '/data/qh_20T_share_file/lct/CT67/dataset' # 根据您的截图，这是包含 .gz 文件的目录
    # 【通用配置】radMLBench 公开数据集的标签列名都是 'Target'
    PUBLIC_LABEL_COL = 'Target'
    # 【只需修改这里来更换数据集】
    # public_datasets 列表中的名称必须与您本地保存的 .gz 文件名（不含扩展名）一致。
    public_datasets = ['UPENN-GBM'] # 'C4KC-KiTS', 'BraTS-2021'

    K_FEATURES = 50
    GA_POPULATION_SIZE = 10
    GA_OMEGA = 0.5
    THETA = 0.9

    print("#"*70)
    print(f"### 开始运行实验：本地数据 + 公开基准测试 (K={K_FEATURES}) ###")
    print("#"*70)
    
    # 任务 1: 本地数据
    print(f"\n\n>>> [任务 1] 加载本地数据... <<<")
    local_data = load_and_clean_radiomics(LOCAL_CSV_PATH, LOCAL_LABEL_COL)
    
    X, X_df, y, f_names = local_data
    run_analysis_on_dataset(X, X_df, y, f_names, 
                            K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, 
                            dataset_title="Local Ovarian Data")

    # 任务 2: 公开数据集 (修改为自动化路径构建)
    for ds_name in public_datasets:
        print(f"\n\n>>> [任务 2] 加载公开数据集: {ds_name}... <<<")
        
        # 【核心修改逻辑】根据数据集名称和通用格式自动构建路径
        file_name = f"{ds_name}.gz"
        local_path = os.path.join(PUBLIC_DATASET_DIR, file_name)
        label_col = PUBLIC_LABEL_COL # 使用通用标签列名
        
        # 检查文件是否存在
        if not os.path.exists(local_path):
             print(f"!!! 致命警告: 文件未找到 - {local_path}。跳过此数据集。")
             continue

        # 复用 load_and_clean_radiomics 加载本地 .gz 文件
        rad_data = load_and_clean_radiomics(local_path, label_col)
        X, X_df, y, f_names = rad_data
            
        run_analysis_on_dataset(X, X_df, y, f_names, 
                                K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, 
                                dataset_title=ds_name)

if __name__ == "__main__":
    main()
