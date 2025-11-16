import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed

def calculate_fitness(chromosomes, X, y, similarity_matrix, n_jobs=-1):
    """
    计算种群适应度 - 升级版
    改进点：
    1. 使用 LogisticRegression 替代 KNN，与最终评估模型对齐。
    2. 增加 class_weight='balanced' 解决特异度(Specificity)过低的问题。
    3. 使用 3-Fold 交叉验证替代单次 Split，提升泛化能力。
    4. 使用 AUC 作为核心指标。
    """

    # 定义单个染色体处理函数
    def _process_chromosome(chromosome):
        chromosome_arr = np.array(chromosome)
        selected_mask = chromosome_arr.astype(bool)
        selected_features = np.where(selected_mask)[0]
      
        # 如果没有选中任何特征，适应度为0
        if len(selected_features) == 0:
            return 0.0
        
        # 提取子集
        X_sub = X[:, selected_features]
      
        # --- [核心改进 1] ---
        # 使用逻辑回归 (liblinear 适合小样本)，并开启类别平衡
        clf = LogisticRegression(
            solver='liblinear', 
            class_weight='balanced',  # <--- 关键！修复 Specificity=0.18 的核心参数
            random_state=42, 
            max_iter=1000
        )

        # --- [核心改进 2] ---
        # 使用 3折交叉验证获取更稳健的 AUC 分数
        try:
            # scoring 可以是 'roc_auc' 或 'accuracy'，医学数据建议优先 'roc_auc'
            cv_scores = cross_val_score(clf, X_sub, y, cv=3, scoring='roc_auc')
            main_score = cv_scores.mean()
        except ValueError:
            # 极少数情况（如某折中只有一类样本）可能报错
            main_score = 0.0

        # --- 冗余度惩罚 (保持原逻辑) ---
        n = len(selected_features)
        if n < 2:
            denominator = 0.0
        else:
            # 从全局相似性矩阵中提取子矩阵
            sub_sim = similarity_matrix[selected_features, :][:, selected_features]
            # 只计算上三角部分的和 (不含对角线)
            triu_idx = np.triu_indices_from(sub_sim, k=1)
            total_sim = sub_sim[triu_idx].sum()
            denominator = (2 * total_sim) / (n * (n - 1))
        
        # --- [核心改进 3] ---
        # 调整权重 w。
        # w=0.9 表示我们需要 90% 的性能 + 10% 的去冗余。
        # 如果发现特征太多且高度相关，可以降低 w (如 0.8)
        w = 0.9  
        fitness = (w * main_score) + ((1 - w) * (1 - denominator))

        return fitness

    # 并行计算
    return Parallel(n_jobs=n_jobs)(delayed(_process_chromosome)(chr) for chr in chromosomes)