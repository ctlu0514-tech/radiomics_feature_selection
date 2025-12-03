# 文件: calculate_fitness.py

import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score  <--- 移除这一行
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# +++ 添加这两个导入 +++
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

def calculate_fitness_old(chromosomes, X, y, similarity_matrix, n_jobs=-1):
    """
    参数：
    - chromosomes: 染色体列表 [n_chromosomes, n_features]
    - X: 已标准化的特征矩阵 [n_samples, n_features]
    - y: 目标标签 [n_samples]
    - similarity_matrix: 预计算的全局特征相似性矩阵 [n_features, n_features]
    - n_jobs: 并行任务数（-1使用全部核心）
  
    返回：
    - fitness_values: 适应度值列表 [n_chromosomes]
    """

    # 定义单个染色体处理函数
    def _process_chromosome(chromosome):
        chromosome_arr = np.array(chromosome)
        selected_mask = chromosome_arr.astype(bool)
        selected_features = np.where(selected_mask)[0]
      
        X_sub = X[:, selected_features]
      
        X_train, X_val, y_train, y_val = train_test_split(
            X_sub, y, 
            test_size=0.3, 
            stratify=y, 
            random_state=42  # 保持固定随机种子保证可重复性
        )
      
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        ca = knn.score(X_val, y_val)

        #从全局相似性矩阵中提取仅包含当前染色体选中特征的子矩阵
        sub_sim = similarity_matrix[selected_features, :][:, selected_features]
        #生成上三角索引
        triu_idx = np.triu_indices_from(sub_sim, k=1)
        total_sim = sub_sim[triu_idx].sum()
      
        n = len(selected_features)
        denominator = (2 * total_sim) / (n * (n - 1))
        fitness = ca / (denominator + 1e-9)
        return fitness

    # 使用并行处理替代原有循环
    return Parallel(n_jobs=n_jobs)(delayed(_process_chromosome)(chr) for chr in chromosomes)
