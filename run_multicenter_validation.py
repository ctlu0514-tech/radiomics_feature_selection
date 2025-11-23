# 文件名: run_multicenter_validation_v2.py
# 描述: 多中心验证流程 (含 LASSO, RFE, mRMR, CDGAFS)
# 特性: 严格防泄露 (Split -> Impute/Scale on Train -> Apply to Others)

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
    print("警告: 未找到 CDGAFS，将跳过该方法。")

try:
    import pymrmr
    HAS_MRMR = True
except ImportError:
    HAS_MRMR = False
    print("警告: 未找到 pymrmr，将跳过 mRMR 方法 (pip install pymrmr)。")

warnings.filterwarnings('ignore')

# ===================================================================
# 1. 数据加载与划分模块
# ===================================================================
def load_and_split_data(csv_path, train_center_name, label_col='label', center_col='Center_Source', id_col='Sample_ID'):
    print(f"\n--- [1] 加载与拆分数据: {os.path.basename(csv_path)} ---")
    
    if not os.path.exists(csv_path):
        print(f"!!! 错误: 文件不存在 {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # 基础清洗
    missing_cols = [col for col in [label_col, center_col, id_col] if col not in df.columns]
    if missing_cols:
        print(f"!!! 错误: CSV中缺失列: {missing_cols}")
        sys.exit(1)

    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    
    # 提取特征列
    non_feature_cols = [id_col, center_col, label_col]
    feature_cols = [c for c in df.columns if c not in non_feature_cols and not c.startswith('diagnostics_')]
    
    unique_centers = df[center_col].unique()
    print(f"    - 总样本: {len(df)}")
    print(f"    - 特征数: {len(feature_cols)}")
    print(f"    - 中心列表: {unique_centers}")

    # 验证训练中心
    if train_center_name not in unique_centers:
        print(f"!!! 错误: 训练中心 '{train_center_name}' 不存在。")
        sys.exit(1)
        
    # 提取训练中心并拆分 (7:3)
    train_center_df = df[df[center_col] == train_center_name]
    X_train_raw, X_intval_raw, y_train, y_intval = train_test_split(
        train_center_df[feature_cols], 
        train_center_df[label_col],
        test_size=0.3, 
        stratify=train_center_df[label_col],
        random_state=42
    )
    
    datasets = {
        'Train': (X_train_raw, y_train.values),
        'IntVal': (X_intval_raw, y_intval.values)
    }
    
    # 提取外部验证中心
    external_centers = [c for c in unique_centers if c != train_center_name]
    for ext_c in external_centers:
        ext_df = df[df[center_col] == ext_c]
        datasets[f"Ext_{ext_c}"] = (ext_df[feature_cols], ext_df[label_col].values)
        print(f"    - 外部验证 ({ext_c}): {len(ext_df)} 例")

    return datasets, feature_cols

# ===================================================================
# 2. 严谨的预处理 (Fit on Train ONLY)
# ===================================================================
def preprocess_securely(datasets):
    print(f"\n--- [2] 预处理 (标准化 & 填补) ---")
    X_train_df, y_train = datasets['Train']
    
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    # 1. Fit on TRAIN
    imputer.fit(X_train_df)
    X_train_imp = imputer.transform(X_train_df)
    
    # 方差过滤
    stds = np.std(X_train_imp, axis=0)
    good_idx = np.where(stds > 1e-6)[0]
    X_train_imp = X_train_imp[:, good_idx]
    
    scaler.fit(X_train_imp)
    
    # 2. Transform ALL
    processed_datasets = {}
    for name, (X_df, y) in datasets.items():
        X_imp = imputer.transform(X_df)
        X_filt = X_imp[:, good_idx]
        X_scaled = scaler.transform(X_filt)
        processed_datasets[name] = (X_scaled, y)
        
    # 更新特征名列表
    all_cols = datasets['Train'][0].columns
    final_features = [all_cols[i] for i in good_idx]
    
    print(f"    - 预处理完成。保留特征数: {len(final_features)}")
    return processed_datasets, final_features

# ===================================================================
# 3. 特征选择核心模块 (含 LASSO, RFE, mRMR, CDGAFS)
# ===================================================================
def run_feature_selection(X_train, y_train, feature_names, method, K):
    print(f"\n--- [3] 运行特征选择: {method} (Target K={K}) ---")
    start_t = time.time()
    selected_idx = []
    
    # --- 方法 1: LASSO ---
    if method == 'LASSO':
        clf = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', scoring='roc_auc', class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        coefs = np.abs(clf.coef_[0])
        indices = np.argsort(coefs)[::-1]
        selected_idx = indices[:K]

    # --- 方法 2: RFE (Recursive Feature Elimination) ---
    elif method == 'RFE':
        # 使用逻辑回归作为基估计器
        estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        selector = RFE(estimator, n_features_to_select=K, step=1)
        selector.fit(X_train, y_train)
        selected_idx = np.where(selector.support_)[0]

    # --- 方法 3: mRMR ---
    elif method == 'mRMR':
        if not HAS_MRMR:
            print("    [跳过] 缺少 pymrmr 库")
            return []
        
        # mRMR 需要 DataFrame 格式，且通常需要离散化处理才能有效工作
        # 构造临时 DataFrame
        df_temp = pd.DataFrame(X_train, columns=feature_names)
        df_temp.insert(0, 'label', y_train)
        
        # 离散化 (Discretization): 转换为 10 个 bin 的整数
        # 注意：这只用于特征选择，不改变原始用于训练的 X_train
        for col in feature_names:
            try:
                df_temp[col] = pd.qcut(df_temp[col], q=10, labels=False, duplicates='drop')
            except ValueError:
                # 如果这一列数值太单一无法分箱，就置为0
                df_temp[col] = 0
                
        # 运行 mRMR (使用 MIQ 方法: Mutual Information Quotient)
        try:
            selected_names = pymrmr.mRMR(df_temp, 'MIQ', K)
            # 将选出的特征名映射回索引
            name_to_idx = {name: i for i, name in enumerate(feature_names)}
            selected_idx = [name_to_idx[n] for n in selected_names if n in name_to_idx]
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
            # 如果 CDGAFS 选出的特征多于 K，使用 RFE 进行二次剪枝
            if len(sel_idx) > K:
                print(f"    CDGAFS 选中 {len(sel_idx)} 个，正在 RFE 剪枝至 {K}...")
                est = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
                rfe = RFE(est, n_features_to_select=K, step=1)
                rfe.fit(X_train[:, sel_idx], y_train)
                selected_idx = np.array(sel_idx)[rfe.support_]
            else:
                selected_idx = sel_idx
        except Exception as e:
            print(f"    CDGAFS 运行出错: {e}")
            selected_idx = []
    
    # 统一转为 numpy array 并确保非空
    selected_idx = np.array(selected_idx, dtype=int)
    
    # 兜底机制：如果选出的特征为0，强行选方差最大的前 K 个
    if len(selected_idx) == 0:
        print(f"    警告: {method} 未选中任何特征，回退到方差筛选。")
        vars = np.var(X_train, axis=0)
        selected_idx = np.argsort(vars)[::-1][:K]

    print(f"    - 最终选中特征数: {len(selected_idx)} (耗时 {time.time()-start_t:.1f}s)")
    return selected_idx

# ===================================================================
# 4. 模型评估
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
    CSV_FILE = '/data/qh_20T_share_file/lct/CT67/Merged_All_Centers.csv'
    TRAIN_CENTER = 'FuYi'
    
    # 这里加入了所有方法
    METHODS = ['CDGAFS','LASSO', 'RFE', 'mRMR'] 
    K_FEATURES = 12
    
    # --- 执行 ---
    datasets_raw, feature_names_raw = load_and_split_data(CSV_FILE, TRAIN_CENTER)
    datasets, feature_names = preprocess_securely(datasets_raw)
    X_train, y_train = datasets['Train']
    
    print(f"\n{'='*60}")
    print(f"多中心验证 (Train: {TRAIN_CENTER}, K={K_FEATURES})")
    print(f"{'='*60}")
    
    final_summary = []
    
    for method in METHODS:
        sel_idx = run_feature_selection(X_train, y_train, feature_names, method, K_FEATURES)
        
        # 打印选中的特征
        selected_names = [feature_names[i] for i in sel_idx]
        print(f"    特征列表: {selected_names}")
        
        # 评估
        scores = evaluate_model(datasets, sel_idx)
        
        # 打印结果
        print(f"\n>>> {method} 结果 <<<")
        print(f"{'Dataset':<20} | {'AUC':<8} | {'ACC':<8} | {'Sens':<8} | {'Spec':<8}")
        print("-" * 65)
        
        row = {'Method': method, 'Features': len(sel_idx)}
        for ds_name, res in scores.items():
            print(f"{ds_name:<20} | {res['AUC']:.4f}   | {res['ACC']:.4f}   | {res['Sens']:.4f}   | {res['Spec']:.4f}")
            row[f"{ds_name}_AUC"] = res['AUC']
            row[f"{ds_name}_ACC"] = res['ACC']
        
        final_summary.append(row)
        print("-" * 65)

    # 保存
    pd.DataFrame(final_summary).to_csv('multicenter_validation_comparison.csv', index=False)
    print("\n完成。结果已保存。")

if __name__ == "__main__":
    main()