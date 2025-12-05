import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from normalize_scores import normalize_scores  # 归一化函数

def construct_pearson_only_graph(X, theta):
    """
    仅使用皮尔逊相关系数构建特征图。
    这是用于消融实验的对照组函数。
    """
    print("\n--- 构建仅基于皮尔逊的图 ---")
    # 计算 Pearson 相关系数矩阵
    pearson_matrix = np.corrcoef(X, rowvar=False)
    abs_corr = np.abs(pearson_matrix)
    
    # 使用你项目中的归一化方法
    pearson_norm = normalize_scores(abs_corr)
    
    # 复制一份用于过滤
    graph_matrix = pearson_norm.copy()
    
    # 根据阈值过滤边
    num_edges_before = np.count_nonzero(graph_matrix) // 2
    graph_matrix[graph_matrix < theta] = 0
    num_edges_after = np.count_nonzero(graph_matrix) // 2
    print(f"根据阈值 {theta} 过滤的边数: {num_edges_before - num_edges_after}")

    np.fill_diagonal(graph_matrix, 0)
    graph_matrix = np.nan_to_num(graph_matrix, nan=0)
    
    G = nx.from_numpy_array(graph_matrix)
    print("皮尔逊图构建完成:")
    print(G)
    return G
