"""
FCM.py

模糊C均值聚类（Fuzzy C-Means, FCM）实现

"""

import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    rand_score,
)
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# 数据集配置
# ==========================================================


# path        : 数据集路径
# clusters    : 聚类簇数
# label_col   : 标签列索引(None表示无标签)
# id_col      : ID列索引(None表示无ID列)
# header      : 表头行(0表示第一行为表头，None表示无表头)
# sep         : 分隔符
# skiprows    : 跳过前几行
# comment     : 注释符(None表示无注释符)

Dataset = namedtuple(
    "Dataset",
    [
        "path",
        "clusters",
        "label_col",
        "id_col",
        "header",
        "sep",
        "skip_rows",
        "comment"
    ]
)

DATASETS = {
    # ==========================
    # UCI数据集
    # ==========================

    # 最后一列是标签，英文逗号作为分割符，无特征名表头
    "iris": Dataset("dataset/uci/iris.data", 3, -1, None, None, ",", 0, None),
    # 最后一列是标签，空格作为分割符，无特征名表头
    "seeds": Dataset("dataset/uci/seeds_dataset.txt", 3, -1, None, None,  r"\s+", 0, None),
    # 最后一列是标签，英文逗号作为分割符，无特征名表头
    "glass": Dataset("dataset/uci/glass.data", 6, -1, 0, None, ",", 0, None),
    # 第一列是标签，英文逗号作为分割符，无特征名表头
    "wine": Dataset("dataset/uci/wine.data", 3, 0, None, None, ",", 0, None),
    # 第一列是ID，第二列是标签，英文逗号作为分割符，无特征名表头
    "wdbc": Dataset("dataset/uci/wdbc.data", 2, 1, 0, None, ",", 0, None),
    # 第一列是ID，第二列是标签，英文逗号作为分割符，无特征名表头
    "ecoli": Dataset("dataset/uci/ecoli.data", 8, -1, 0, None, r"\s+", 0, None),

    # ==========================
    # 人工合成数据集
    # ==========================

    # 空格分隔，最后一列是标签
    "flame": Dataset("dataset/synthetic/Flame.txt", 2, -1, None, None, ",", 0, None),
    "jain": Dataset("dataset/synthetic/Jain.txt", 2, -1, None, None, ",", 0, None),
    "spiral": Dataset("dataset/synthetic/Spiral.txt", 3, -1, None, None, ",", 0, None),
    "panelB": Dataset("dataset/synthetic/panelB.txt", 3, None, None, None, ",", 0, None),
    "panelC": Dataset("dataset/synthetic/panelC.txt", 3, None, None, None, ",", 0, None),
    "r15": Dataset("dataset/synthetic/R15.txt", 15, -1, None, None, ",", 0, None),
    "d31": Dataset("dataset/synthetic/D31.txt", 31, -1, None, None, ",", 0, None),
    "aggregation": Dataset("dataset/synthetic/Aggregation.txt", 7, -1, None, None, ",", 0, None),
    "compound": Dataset("dataset/synthetic/Compound.txt", 6, -1, None, None, ",", 0, None),
    "pathbased": Dataset("dataset/synthetic/Pathbased.txt", 3, -1, None, None, ",", 0, None),
}


# ==========================================================
# 数据加载
# ==========================================================

# ==========================================================
# 数据加载
# ==========================================================

def load_dataset(dataset_name):
    """
    加载数据集

    Parameters
    ----------
    dataset_name : str
        数据集名称

    Returns
    -------
    features : ndarray
        特征矩阵

    labels_true : list or None
        真实标签

    num_clusters : int
        聚类簇数
    """

    config = DATASETS[dataset_name]
    df = pd.read_csv(config.path, sep=config.sep, header=config.header, skiprows=config.skip_rows, comment=config.comment)
    # 提取真实标签
    if config.label_col is not None:
        labels_true = df.iloc[:, config.label_col].tolist()
    else:
        labels_true = None

    # 构造需要删除的列(包括标签列、id列)
    drop_cols = []

    if config.id_col is not None:
        drop_cols.append(df.columns[config.id_col])

    if config.label_col is not None:
        drop_cols.append(df.columns[config.label_col])

    # 得到特征矩阵
    if drop_cols:
        # 一次性删除，防止错乱
        features = df.drop(columns=drop_cols).values
    else:
        features = df.values

    return features, labels_true, config.clusters

# ==========================================================
# FCM核心算法
# ==========================================================

def initialize_membership_matrix(n, c, random_state=42):
    """
    初始化隶属度矩阵
    """
    rng = np.random.default_rng(random_state)
    membership_matrix = rng.random((n, c))
    membership_matrix /= np.sum(membership_matrix, axis=1, keepdims=True)
    return membership_matrix


def calculate_cluster_center(features, membership_matrix, c, m):
    """
    根据隶属度矩阵计算聚类中心
    """
    n = len(features)
    cluster_centers = []
    cluster_mem_val_list = list(zip(*membership_matrix))
    for j in range(c):
        x = cluster_mem_val_list[j]
        x_raised = [e ** m for e in x]
        denominator = sum(x_raised)
        temp_num = []
        for i in range(n):
            data_point = features[i]
            prod = [x_raised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]
        cluster_centers.append(center)
    return np.array(cluster_centers)


def update_membership_value(features, membership_matrix, cluster_centers, c, m):
    """
    更新隶属度矩阵
    """
    n = len(features)
    power = 2.0 / (m - 1)

    for i in range(n):
        x = features[i]
        distances = np.array([
            np.linalg.norm(x - cluster_centers[j])
            for j in range(c)
        ])
        distances = np.maximum(distances, 1e-10)

        for j in range(c):
            den = sum([
                (distances[j] / distances[k]) ** power
                for k in range(c)
            ])
            membership_matrix[i][j] = 1.0 / den

    return membership_matrix


def get_clusters(membership_matrix):
    """
    获取聚类标签
    """
    return np.argmax(membership_matrix, axis=1)


def fuzzy_c_means_clustering(features, num_clusters, tol=1e-5, fuzzifier=2.0, max_iter=100):
    """
    FCM聚类主函数

    Parameters
    ----------
    features : ndarray
              特征矩阵(符号：X)
    num_clusters : int
                聚类簇数(符号：c)
    tol : float, default=1e-5
          终止阈值容差tolerance(符号：epsilon)
    fuzzifier : float, default=2
                模糊化因子fuzzifier(符号：m)
    max_iter : int
               最大迭代数(符号：T)

    Returns
    -------
    cluster_labels : ndarray
                    聚类标签

    cluster_centers : list
                  聚类中心

    membership_matrix : int
        隶属度矩阵
    """

    n_samples = len(features)  # 样本总数(符号：n)
    start = time.time()
    membership_matrix = initialize_membership_matrix(n_samples, num_clusters)
    cluster_centers = np.zeros((num_clusters, features.shape[1]))  # 提前初始化，避免警告
    iteration = 0
    while iteration <= max_iter:
        cluster_centers = calculate_cluster_center(features, membership_matrix, num_clusters, fuzzifier)
        old_membership_mat = membership_matrix.copy()
        membership_matrix = update_membership_value(features, membership_matrix, cluster_centers, num_clusters, fuzzifier)
        if np.linalg.norm(membership_matrix - old_membership_mat) < tol:
            break
        iteration += 1
    cluster_labels = get_clusters(membership_matrix)
    elapsed_time = time.time() - start

    print(f"总计迭代次数: {iteration};   " f"耗时: {elapsed_time:.4f} s")

    return cluster_labels, cluster_centers, membership_matrix


# ==========================================================
# 标签对齐
# ==========================================================

def align_labels(labels_true, labels_pred):
    """
    使用匈牙利算法进行标签匹配
    """
    cm = confusion_matrix(labels_true, labels_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {}
    for true_label, pred_label in zip(row_ind, col_ind):
        mapping[pred_label] = true_label

    labels_pred_aligned = np.array([
        mapping[label]
        for label in labels_pred
    ])

    return labels_pred_aligned


# ==========================================================
# 聚类评价指标
# ==========================================================

def clustering_indicators(labels_true, labels_pred):
    """
    计算聚类评价指标
    """

    if isinstance(labels_true[0], str):
        # 真实标签转化为数字标签
        labels_true = LabelEncoder().fit_transform(labels_true)

    labels_pred_aligned = align_labels(labels_true, labels_pred)

    f1 = f1_score(labels_true, labels_pred_aligned, average="macro")
    accuracy = accuracy_score(labels_true, labels_pred_aligned)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ri = rand_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)

    return f1, accuracy, nmi, ri, ari


# ==========================================================
# 聚类结果可视化
# ==========================================================

def draw_cluster(features, cluster_centers, labels_pred):
    """
    绘制聚类结果散点图
    """

    features = np.asarray(features)
    cluster_centers = np.asarray(cluster_centers)

    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        cluster_centers_2d = pca.transform(cluster_centers)
    else:
        features_2d = features
        cluster_centers_2d = cluster_centers

    plt.figure(figsize=(8, 6))
    # 做散点图
    plt.scatter(features_2d[:, 0], features_2d[:, 1], marker='o', c='black', s=7)  # 原图
    plt.show()
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_pred, cmap="nipy_spectral", s=7, marker="o")

    plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], marker="x", color="m", s=30)
    # 设置x和y坐标轴刻度的标签字体和字号
    # plt.xticks(fontproperties='Times New Roman', fontsize=10.5)
    # plt.yticks(fontproperties='Times New Roman', fontsize=10.5)
    # plt.xlabel("x - label", fontdict={'family': 'Times New Roman', 'size': 10.5}, loc="right")
    # plt.ylabel("y - label", fontdict={'family': 'Times New Roman', 'size': 10.5}, loc="top")
    plt.title("FCM Clustering Result")

    plt.show()


# ==========================================================
# 主程序
# ==========================================================

if __name__ == "__main__":
    # 初始化本次数据集名称
    _dataset_name = "iris"
    # 加载数据集
    _features, _labels_true, _num_clusters = load_dataset(_dataset_name)
    # 输出样本数n_samples和属性数n_attributes(维度dimensions)
    print("加载数据集:   "f"样本数量={_features.shape[0]}  " f"属性数量(维度)={_features.shape[1]}")

    _tol = 1e-5
    _fuzzifier = 2.0
    _max_iter = 100
    print("开始fuzzy-c-means:   " f"c={_num_clusters}  " f"n={len(_features)}   " f"epsilon={_tol}  " f"m={_fuzzifier}  " f"T={_max_iter} ")
    # 进行FCM算法
    _labels_pred, _cluster_centers, _membership_matrix = fuzzy_c_means_clustering(_features, _num_clusters, _tol, _fuzzifier, _max_iter)

    # 开始计数聚类指标
    if _labels_true is not None:
        # 有标签：计算聚类指标
        F1, ACC, NMI, RI, ARI = clustering_indicators(_labels_true, _labels_pred)
        print("聚类指标: " f"F1={F1:.6f}  " f"ACC={ACC:.6f}  " f"NMI={NMI:.6f}  " f"RI={RI:.6f}  " f"ARI={ARI:.6f}")
    else:
        # 数据集无标签：无法计算聚类指标
        print("Dataset without labels，unable to calculate clustering index")
    # 输出散点图
    draw_cluster(_features, _cluster_centers, _labels_pred)
