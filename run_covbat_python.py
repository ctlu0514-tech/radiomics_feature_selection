# 文件名: run_covbat_python.py
# 描述: 直接使用 Python 版 CovBat 对多中心影像组学数据进行协调 (修复版)

import pandas as pd
import numpy as np
import os
import sys
from sklearn.impute import SimpleImputer

# 导入您刚才保存的 covbat 模块
try:
    import covbat
except ImportError:
    print("错误: 未找到 covbat.py 文件。")
    print("请将 covbat.py 放入当前目录。")
    sys.exit(1)

def main():
    # --- 1. 配置路径 ---
    INPUT_CSV = '/data/qh_20T_share_file/lct/CT67/Merged_All_Centers.csv'
    OUTPUT_CSV = '/data/qh_20T_share_file/lct/CT67/Merged_All_Centers_CovBat_Python.csv'

    CENTER_COL = 'Center_Source'  # 批次变量 (Batch)
    LABEL_COL = 'label'           # 保护变量 (生物学差异)
    ID_COL = 'Sample_ID'

    print(f"--- 读取数据: {os.path.basename(INPUT_CSV)} ---")
    if not os.path.exists(INPUT_CSV):
        print(f"文件不存在: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    # --- 2. 数据准备 ---
    metadata_cols = [CENTER_COL, LABEL_COL, ID_COL]
    diag_cols = [c for c in df.columns if c.startswith('diagnostics_')]
    all_meta_cols = metadata_cols + diag_cols
    
    features_df = df.drop(columns=[c for c in all_meta_cols if c in df.columns], errors='ignore')
    features_df = features_df.select_dtypes(include=[np.number]) 
    feature_names = features_df.columns.tolist()
    
    print(f"   -> 特征数量: {len(feature_names)}")

    # 填补 NaN
    imputer = SimpleImputer(strategy='mean')
    data_np = imputer.fit_transform(features_df)
    
    # 移除常量/零方差特征
    stds = np.std(data_np, axis=0)
    non_const_idx = np.where(stds > 1e-6)[0]
    data_clean = data_np[:, non_const_idx]
    feature_names_clean = [feature_names[i] for i in non_const_idx]
    
    print(f"   -> 清洗后特征: {len(feature_names_clean)} (移除了 {len(feature_names) - len(feature_names_clean)} 个常量)")

    # --- 3. 格式转换 ---
    # A. 构造 data (转置: n_features x n_samples)
    data_for_covbat = pd.DataFrame(data_clean.T, index=feature_names_clean, columns=df.index)
    
    # B. 构造 batch
    batch_series = df[CENTER_COL]
    
    # C. 构造 model (协变量)
    # 【关键修改】：保持 label 为数值型 (int)，不要转 str
    model_df = df[[LABEL_COL]].copy()
    model_df[LABEL_COL] = model_df[LABEL_COL].astype(int)

    print("\n--- 开始运行 CovBat (Python版) ---")
    print(f"   -> Batch: {CENTER_COL}")
    print(f"   -> Numerical Covariates: {LABEL_COL} (作为数值型变量保护)")

    # 调用 covbat
    # 【关键修改】：numerical_covariates=[LABEL_COL]
    # 这样 covbat.py 会把它当作数字列处理，避免拼接字符串导致报错
    data_harmonized_df = covbat.covbat(
        data=data_for_covbat, 
        batch=batch_series, 
        model=model_df, 
        numerical_covariates=[LABEL_COL], 
        n_pc=0 
    )
    
    print("   -> 校正完成！")

    # --- 4. 结果重组与保存 ---
    # 转置回来
    data_harmonized = data_harmonized_df.T.values
    
    df_harmonized = pd.DataFrame(data_harmonized, columns=feature_names_clean)
    
    meta_df = df[all_meta_cols].reset_index(drop=True)
    df_final = pd.concat([meta_df, df_harmonized], axis=1)
    
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n成功! 文件已保存至: {OUTPUT_CSV}")
    print("下一步: 请运行 check_batch_effect_fixed.py 检查这个新文件的效果。")

if __name__ == "__main__":
    main()