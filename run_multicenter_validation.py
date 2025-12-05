# 文件名: run_multicenter_validation.py
# 描述: 多中心验证流程 (性能优化版：加速 mRMR 和 RFE)

import pandas as pd
import numpy as np
import warnings
import time
import sys
import os

# --- 机器学习库 ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.feature_selection import RFE

# --- 尝试导入扩展算法 ---
try:
    from CDGAFS import cdgafs_feature_selection
    HAS_CDGAFS = True
except ImportError:
    HAS_CDGAFS = False
    # print("警告: 未找到 CDGAFS，将跳过该方法。")

# 【优化 1】切换为高效的 mrmr_classif
try:
    from mrmr import mrmr_classif
    HAS_MRMR = True
except ImportError:
    HAS_MRMR = False
    print("警告: 未找到 mrmr-selection 库 (pip install mrmr-selection)。将跳过 mRMR。")

warnings.filterwarnings('ignore')

# ===================================================================
# 1. 数据加载与划分模块 (保持不变)
# ===================================================================
def load_and_split_data(csv_path, train_center_names, label_col='label', center_col='Center_Source', id_col='Sample_ID'):
    print(f"\n--- [1] 加载与拆分数据: {os.path.basename(csv_path)} ---")
    
    if isinstance(train_center_names, str):
        train_center_names = [train_center_names]
        
    if not os.path.exists(csv_path):
        print(f"!!! 错误: 文件不存在 {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    missing_cols = [col for col in [label_col, center_col, id_col] if col not in df.columns]
    if missing_cols:
        print(f"!!! 错误: CSV中缺失列: {missing_cols}")
        sys.exit(1)

    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    
    non_feature_cols = [id_col, center_col, label_col]
    feature_cols = [c for c in df.columns if c not in non_feature_cols and not c.startswith('diagnostics_')]
    
    unique_centers = df[center_col].unique()
    print(f"    - 总样本: {len(df)}")
    print(f"    - 特征数: {len(feature_cols)}")
    print(f"    - 中心列表: {unique_centers}")

    for tc in train_center_names:
        if tc not in unique_centers:
            print(f"!!! 错误: 指定的训练中心 '{tc}' 在数据中不存在。")
            sys.exit(1)
        
    train_df = df[df[center_col].isin(train_center_names)]
    
    X_train_raw, X_intval_raw, y_train, y_intval = train_test_split(
        train_df[feature_cols], 
        train_df[label_col],
        test_size=0.3, 
        stratify=train_df[label_col],
        random_state=42
    )
    
    datasets = {
        'Train': (X_train_raw, y_train.values),
        'IntVal': (X_intval_raw, y_intval.values)
    }
    
    external_centers = [c for c in unique_centers if c not in train_center_names]
    
    for ext_c in external_centers:
        ext_df = df[df[center_col] == ext_c]
        datasets[f"Ext_{ext_c}"] = (ext_df[feature_cols], ext_df[label_col].values)
        print(f"    - 外部验证 ({ext_c}): {len(ext_df)} 例")
        
    print(f"    - 训练集涵盖中心: {train_center_names} (共 {len(train_df)} 例)")

    return datasets, feature_cols

# ===================================================================
# 2. 严谨的预处理 (保持不变)
# ===================================================================
def preprocess_securely(datasets):
    print(f"\n--- [2] 预处理 (标准化 & 填补) ---")
    X_train_df, y_train = datasets['Train']
    
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    imputer.fit(X_train_df)
    X_train_imp = imputer.transform(X_train_df)
    
    stds = np.std(X_train_imp, axis=0)
    good_idx = np.where(stds > 1e-6)[0]
    X_train_imp = X_train_imp[:, good_idx]
    
    scaler.fit(X_train_imp)
    
    processed_datasets = {}
    for name, (X_df, y) in datasets.items():
        X_imp = imputer.transform(X_df)
        X_filt = X_imp[:, good_idx]
        X_scaled = scaler.transform(X_filt)
        processed_datasets[name] = (X_scaled, y)
        
    all_cols = datasets['Train'][0].columns
    final_features = [all_cols[i] for i in good_idx]
    
    print(f"    - 预处理完成。保留特征数: {len(final_features)}")
    return processed_datasets, final_features

# ===================================================================
# 3. 特征选择核心模块 (关键优化部分)
# ===================================================================
def run_feature_selection(X_train, y_train, feature_names, method, K):
    print(f"\n--- [3] 运行特征选择: {method} (Target K={K}) ---")
    start_t = time.time()
    selected_idx = []
    
    # --- 方法 1: LASSO (自带高效实现，保持不变) ---
    if method == 'LASSO':
        clf = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', scoring='roc_auc', class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        coefs = np.abs(clf.coef_[0])
        indices = np.argsort(coefs)[::-1]
        selected_idx = indices[:K]

    # --- 方法 2: RFE (优化：步长加速) ---
    elif method == 'RFE':
        # 【优化】step=0.05 表示每轮剔除 5% 的特征，比 step=1 快几十倍
        estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        
        # 如果特征特别多，用比例 step；如果特征少，用整数 step
        n_feats = X_train.shape[1]
        step_val = 0.05 if n_feats > 500 else 1
        
        selector = RFE(estimator, n_features_to_select=K, step=step_val)
        selector.fit(X_train, y_train)
        selected_idx = np.where(selector.support_)[0]

    # --- 方法 3: mRMR (优化：使用 mrmr_classif) ---
    elif method == 'mRMR':
        if not HAS_MRMR:
            print("    [跳过] 缺少 mrmr-selection 库")
            return []
        
        try:
            # mrmr_classif 需要 DataFrame 输入
            X_df = pd.DataFrame(X_train, columns=feature_names)
            y_series = pd.Series(y_train)
            
            # 【优化】使用 mrmr_classif 直接选择，无需手动离散化
            selected_features = mrmr_classif(X=X_df, y=y_series, K=K, show_progress=False)
            
            # 将选出的特征名转回索引
            name_to_idx = {name: i for i, name in enumerate(feature_names)}
            selected_idx = [name_to_idx[f] for f in selected_features if f in name_to_idx]
            
        except Exception as e:
            print(f"    mRMR 运行出错: {e}")
            selected_idx = []

    # --- 方法 4: CDGAFS ---
    elif method == 'CDGAFS':
        if not HAS_CDGAFS:
            print("    [跳过] 缺少 CDGAFS 库")
            return []
        try:
            (sel_idx, _, _, _, _) = cdgafs_feature_selection(
                X=X_train, y=y_train, gene_list=feature_names,
                theta=0.9, omega=0.6, population_size=100,
                w_bio_boost=0.0, graph_type='pearson_only'
            )
            # 剪枝
            if len(sel_idx) > K:
                print(f"    CDGAFS 选中 {len(sel_idx)} 个，RFE 剪枝至 {K}...")
                est = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
                rfe = RFE(est, n_features_to_select=K, step=1)
                rfe.fit(X_train[:, sel_idx], y_train)
                selected_idx = np.array(sel_idx)[rfe.support_]
            else:
                selected_idx = sel_idx
        except Exception as e:
            print(f"    CDGAFS 运行出错: {e}")
            selected_idx = []
    
    selected_idx = np.array(selected_idx, dtype=int)
    
    # 兜底
    if len(selected_idx) == 0:
        print(f"    警告: {method} 未选中任何特征，回退到方差筛选。")
        vars = np.var(X_train, axis=0)
        selected_idx = np.argsort(vars)[::-1][:K]

    print(f"    - 最终选中特征数: {len(selected_idx)} (耗时 {time.time()-start_t:.1f}s)")
    return selected_idx

# ===================================================================
# 4. 模型评估 (保持不变)
# ===================================================================
def evaluate_model(datasets, selected_idx):
    results = {}
    X_train, y_train = datasets['Train']
    X_train_sel = X_train[:, selected_idx]
    
    clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    clf.fit(X_train_sel, y_train)
    
    for name, (X_full, y_true) in datasets.items():
        X_sel = X_full[:, selected_idx]
        y_pred = clf.predict(X_sel)
        y_prob = clf.predict_proba(X_sel)[:, 1]
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5
            
        acc = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results[name] = {'AUC': auc, 'ACC': acc, 'Sens': sens, 'Spec': spec}
    return results

# ===================================================================
# 5. 主函数
# ===================================================================
def main():
    # --- 配置 ---
    CSV_FILE = '/data/qh_20T_share_file/lct/CT67/Merged_All_Centers_Strict_Harmonized.csv'
    TRAIN_CENTERS = ['ShiZhongXin'] 
    
    METHODS = ['CDGAFS','LASSO', 'RFE', 'mRMR'] 
    K_FEATURES = 12
    
    # --- 执行 ---
    datasets_raw, feature_names_raw = load_and_split_data(CSV_FILE, TRAIN_CENTERS)
    
    datasets, feature_names = preprocess_securely(datasets_raw)
    X_train, y_train = datasets['Train']
    
    print(f"\n{'='*60}")
    print(f"多中心验证开始 (K={K_FEATURES})")
    print(f"{'='*60}")
    
    final_summary = []
    
    for method in METHODS:
        sel_idx = run_feature_selection(X_train, y_train, feature_names, method, K_FEATURES)
        
        selected_names = [feature_names[i] for i in sel_idx]
        print(f"    特征列表: {selected_names}")
        
        scores = evaluate_model(datasets, sel_idx)
        
        print(f"\n>>> {method} 结果 <<<")
        header = f"{'Dataset':<20} | {'AUC':<8} | {'ACC':<8} | {'Sens':<8} | {'Spec':<8}"
        print(header)
        print("-" * len(header))
        
        row = {'Method': method, 'Features': len(sel_idx)}
        for ds_name, res in scores.items():
            print(f"{ds_name:<20} | {res['AUC']:.4f}   | {res['ACC']:.4f}   | {res['Sens']:.4f}   | {res['Spec']:.4f}")
            row[f"{ds_name}_AUC"] = res['AUC']
            row[f"{ds_name}_ACC"] = res['ACC']
        
        final_summary.append(row)
        print("-" * 65)

    pd.DataFrame(final_summary).to_csv('multicenter_validation_comparison.csv', index=False)
    print("\n完成。结果已保存至 multicenter_validation_comparison.csv")

if __name__ == "__main__":
    main()
