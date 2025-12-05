# 文件名: run_covbat_python.py
# 描述: 严谨版多中心数据协调 (修复版：增加填补和常量去除)

import pandas as pd
import numpy as np
import os
import sys
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# 导入修改过的 covbat 模块
try:
    import covbat
except ImportError:
    print("错误: 未找到 covbat.py 文件。")
    sys.exit(1)

def main():
    # ================= 配置区域 =================
    INPUT_CSV = '/data/qh_20T_share_file/lct/CT67/Merged_All_Centers_With_New.csv'
    OUTPUT_CSV = '/data/qh_20T_share_file/lct/CT67/Merged_All_Centers_Strict_Harmonized.csv'

    CENTER_COL = 'Center_Source'
    LABEL_COL = 'label'
    ID_COL = 'Sample_ID'

    # 定义参考批次 (训练集)
    REF_BATCHES = ['ShiZhongXin'] 
    # ===========================================

    print(f"--- [1] 读取数据: {os.path.basename(INPUT_CSV)} ---")
    if not os.path.exists(INPUT_CSV):
        print(f"文件不存在: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    
    # 检查参考中心
    unique_centers = df[CENTER_COL].unique()
    for ref in REF_BATCHES:
        if ref not in unique_centers:
            print(f"!!! 错误: 参考中心 '{ref}' 未找到！")
            sys.exit(1)

    # ================= 数据预处理 (关键修复步骤) =================
    print("\n--- [2] 数据预处理 (填补 & 去除常量) ---")
    
    # 1. 分离特征
    non_feature_cols = [ID_COL, CENTER_COL, LABEL_COL]
    # 排除 diagnostics 列
    feature_cols = [c for c in df.columns if c not in non_feature_cols and not c.startswith('diagnostics_')]
    
    X = df[feature_cols].values
    print(f"    原始特征数: {len(feature_cols)}")

    # 2. 填补空值 (Imputation)
    # CovBat 不接受 NaN，必须先填补
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 3. 移除低方差/常量特征 (Variance Threshold)
    # 如果某个特征方差为0，会导致 CovBat 内部标准化除以0，产生 NaN 崩溃
    selector = VarianceThreshold(threshold=1e-6) # 阈值设为极小值，去除完全不变的列
    X_cleaned = selector.fit_transform(X_imputed)
    
    # 获取保留下来的特征名
    valid_indices = selector.get_support(indices=True)
    kept_feature_names = [feature_cols[i] for i in valid_indices]
    
    removed_count = len(feature_cols) - len(kept_feature_names)
    print(f"    移除常量/空列: {removed_count}")
    print(f"    剩余特征数: {len(kept_feature_names)}")

    if len(kept_feature_names) == 0:
        print("错误: 所有特征都被移除了！请检查数据。")
        sys.exit(1)

    # ================= 准备 CovBat 输入 =================
    # 使用清洗后的数据构建 DataFrame
    # CovBat 需要 (n_features, n_samples)
    data_for_covbat = pd.DataFrame(X_cleaned.T, index=kept_feature_names, columns=df.index)
    
    batch_series = df[CENTER_COL]
    
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    model_df = df[[LABEL_COL]].copy()

    # ================= 运行 CovBat =================
    print(f"\n--- [3] 开始运行 CovBat (严谨模式) ---")
    print(f"    参考标准: {REF_BATCHES}")

    try:
        data_harmonized_df = covbat.covbat(
            data=data_for_covbat, 
            batch=batch_series, 
            model=model_df, 
            numerical_covariates=[LABEL_COL],
            n_pc=0,
            ref_batch=REF_BATCHES 
        )
        print("    >>> 协调完成！")
        
    except Exception as e:
        print(f"\n!!! CovBat 运行崩溃: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ================= 保存结果 =================
    print(f"\n--- [4] 保存结果 ---")
    
    # 构造最终 DataFrame
    # 1. 取出元数据列
    meta_df = df[non_feature_cols]
    
    # 2. 取出协调后的特征 (转置回 n_samples x n_features)
    # 注意：列名必须对应清洗后的特征名
    harmonized_features = pd.DataFrame(data_harmonized_df.T.values, columns=kept_feature_names)
    
    # 3. 合并
    df_final = pd.concat([meta_df, harmonized_features], axis=1)
    
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print(f"成功保存至: {OUTPUT_CSV}")
    print(f"形状: {df_final.shape}")

if __name__ == "__main__":
    main()
