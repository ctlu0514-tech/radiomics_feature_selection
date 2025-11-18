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
def select_features_cdafs(X, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, pruning_method='RFE'):
    # --- 修改了打印，显示当前使用的是哪种剪枝方法 ---
    print(f"\n--- 正在运行: CDGAFS (剪枝方法: {pruning_method}) ---")
    start_time = time.time()

    # 调用 GA 流程 (这部分保持不变)
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
        
    # --- [!!! 2. 核心修改：使用 'if/elif' 来切换剪枝逻辑 !!!] ---
    if len(selected_indices) > K_FEATURES:
        
        # 选项 A: RFE (你的原始逻辑)
        if pruning_method == 'RFE':
            print(f"    - [优化] 使用 RFE (递归特征消除) 从 {len(selected_indices)} 精选到 {K_FEATURES} 个...")
            
            # (RFE 逻辑保持不变)
            estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
            selector = RFE(estimator, n_features_to_select=K_FEATURES, step=1)
            
            X_ga_selected = X[:, selected_indices]
            selector.fit(X_ga_selected, y)
            
            rfe_support = selector.support_
            selected_indices = np.array(selected_indices)[rfe_support]
            
            print(f"    - RFE 剪枝完成。")

        # 选项 B: Fisher Score (你要求的新逻辑)
        elif pruning_method == 'FISHER':
            print(f"    - [优化] 使用 Fisher Score (过滤法) 从 {len(selected_indices)} 精选到 {K_FEATURES} 个...")
            
            # 1. 获取 GA 选中特征的子集数据
            X_ga_selected = X[:, selected_indices]
            
            # 2. 在这个子集上计算 Fisher Score
            #    (我们复用脚本顶部的 'compute_fisher_score' 导入)
            scores_on_subset = compute_fisher_score(X_ga_selected, y)
            
            # 3. 获取子集中分数最高的 K_FEATURES 个特征的 *索引*
            #    np.argsort 默认从小到大排序，所以我们取最后 K 个
            top_subset_indices = np.argsort(scores_on_subset)[-K_FEATURES:]
            
            # 4. 将这些 "子集索引" 映射回 "原始索引"
            selected_indices_array = np.array(selected_indices)
            selected_indices = selected_indices_array[top_subset_indices]
            print(f"    - Fisher Score 剪枝完成。")
            
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

# def select_features_lasso(X, y, K_FEATURES):
    print("\n--- 正在运行: LASSO (L1) ---")
    start_time = time.time()
    # --- 修改开始 ---
    # 我们不再使用 LogisticRegressionCV，而是使用 LogisticRegression
    # 我们手动设置 C=1.0 (一个标准的、相对较弱的惩罚)。
    # 你可以尝试 C=1.0, C=5.0, C=10.0
    # C 越大 = 惩罚越弱 = 特征越多

    model = LogisticRegression(
        C=1.0,  # <-- 手动设置 C 值
        penalty='l1', 
        solver='liblinear', 
        random_state=42, 
        max_iter=1000
    )
    # --- 修改结束 ---

    model.fit(X, y)
    coefficients = model.coef_[0]

    non_zero_indices = np.where(np.abs(coefficients) > 1e-5)[0]
    
    if len(non_zero_indices) == 0:
        print("!!! LASSO 将所有特征系数都惩罚为0。")
        print("    - 提示：尝试在 'select_features_lasso' 函数中增大 C 的值 (例如 C=5.0 或 C=10.0)")
        return []

    # 你的排序逻辑是完美的
    sorted_indices = sorted(non_zero_indices, 
                        key=lambda i: np.abs(coefficients[i]), 
                        reverse=True)

    # 你的截取逻辑也是完美的
    # 如果 C=1.0 产生了 (例如) 80 个特征, 
    # sorted_indices[:K_FEATURES] 将会正确截取前 35 个
    selected_indices = sorted_indices[:K_FEATURES] 

    elapsed = time.time() - start_time
    print(f"--- LASSO 完成。耗时: {elapsed:.2f} 秒。")
    print(f"    - L1 (C=1.0) 找到了 {len(non_zero_indices)} 个非零特征。")
    print(f"    - 已按系数大小排序并截取前 {len(selected_indices)} 个。---")
    return selected_indices

def select_features_rfe_only(X, y, K_FEATURES):
    """
    独立的 RFE 特征选择器。
    它会从 *所有* 特征开始，递归消除直到剩下 K_FEATURES 个。
    """
    print("\n--- 正在运行: RFE-Only (递归特征消除) ---")
    start_time = time.time()

    # 1. 创建 RFE 的 "评估器" 
    # 我们使用与 CDGAFS 剪枝时相同的评估器，以保证公平对比
    estimator = LogisticRegression(
        solver='liblinear', 
        class_weight='balanced', 
        random_state=42,
        max_iter=3000  # 对于大数据集，可能需要更多迭代
    )

    # 2. 初始化 RFE
    # 目标是直接从所有特征中选出 K_FEATURES 个
    selector = RFE(
        estimator, 
        n_features_to_select=K_FEATURES, 
        step=1  # 每次迭代移除1个最差特征
    )

    # 3. 在 *全部* 数据上执行 RFE
    # 警告: 如果 X.shape[1] 很大 (例如 > 2000)，这一步会非常耗时！
    print(f"    - 正在从 {X.shape[1]} 个特征中精选 {K_FEATURES} 个...")

    selector.fit(X, y) # 在完整的 X_scaled 上运行

    # 4. 获取选中的特征索引
    selected_indices = np.where(selector.support_)[0]

    elapsed = time.time() - start_time
    print(f"--- RFE-Only 完成。耗时: {elapsed:.2f} 秒。选出 {len(selected_indices)} 个特征。---")

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
    K_FEATURES = 50
    
    GA_POPULATION_SIZE = 10
    GA_OMEGA = 0.5 # 稍微调大一点，给 RFE 更多选择空间
    THETA = 0.9

    print("#"*70)
    print("### 开始运行放射组学特征选择与评估对比实验 (优化版) ###")
    print("#"*70)
    
    X_scaled, X_df_filled, y, feature_names = load_and_clean_radiomics(CSV_PATH, LABEL_COLUMN)
    if X_scaled is None: return
        
    all_selected_indices = {}
    
    # 1. CDGAFS (优化版)
    all_selected_indices['CDGAFS'] = select_features_cdafs(
        X_scaled, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, 
        pruning_method='RFE'  # <--- 明确传入 'RFE'还是 ‘FISHER’
    )

    # 2. mRMR
    # all_selected_indices['mRMR'] = select_features_mrmr(X_df_filled.copy(), y, K_FEATURES)

    # 3. LASSO
    # all_selected_indices['LASSO'] = select_features_lasso(X_scaled, y, K_FEATURES)    

    # 4. 【新增】RFE-Only
    # all_selected_indices['RFE-Only'] = select_features_rfe_only(X_scaled, y, K_FEATURES)
    
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
